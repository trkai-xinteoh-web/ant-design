"""
Silkscreen Detection for PCB Images using OpenCV and LabelMe JSON
Processes multiple labeled images with existing annotations and detects silkscreen regions.

Usage:
    python silkscreen_detection.py --input /path/to/input --output /path/to/output [--config config.json]
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SilkscreenColor(Enum):
    """Common silkscreen color ranges (BGR format)"""
    WHITE = {'lower': (200, 200, 200), 'upper': (255, 255, 255)}
    YELLOW = {'lower': (0, 200, 200), 'upper': (100, 255, 255)}
    RED = {'lower': (0, 0, 100), 'upper': (100, 100, 255)}


@dataclass
class DetectionConfig:
    """Configuration for silkscreen detection"""
    # Color detection range (BGR)
    color_lower: Tuple[int, int, int] = (200, 200, 200)  # Default: white
    color_upper: Tuple[int, int, int] = (255, 255, 255)
    
    # Morphological operations
    kernel_size: int = 5
    morph_iterations: int = 2
    
    # Contour filtering
    min_contour_area: float = 50.0  # Minimum area in pixels
    max_contour_area: float = 50000.0  # Maximum area in pixels
    
    # Edge detection fallback
    use_edge_detection: bool = True
    edge_threshold1: int = 50
    edge_threshold2: int = 150
    
    # Shape approximation
    epsilon_percentage: float = 0.02  # Percentage of contour perimeter
    

class SilkscreenDetector:
    """Detects and labels silkscreen regions in PCB images"""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the detector with configuration.
        
        Args:
            config: DetectionConfig object with detection parameters
        """
        self.config = config or DetectionConfig()
        self.image_size = None
    
    def load_labelme_json(self, json_path: str) -> Dict:
        """
        Load LabelMe JSON annotation file.
        
        Args:
            json_path: Path to LabelMe JSON file
            
        Returns:
            Dictionary containing LabelMe annotation data
        """
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            raise
    
    def save_labelme_json(self, json_path: str, data: Dict) -> None:
        """
        Save LabelMe JSON annotation file.
        
        Args:
            json_path: Path to save LabelMe JSON file
            data: Dictionary containing LabelMe annotation data
        """
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved annotations to {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON {json_path}: {e}")
            raise
    
    def create_exclusion_mask(self, image_height: int, image_width: int, 
                             shapes: List[Dict]) -> np.ndarray:
        """
        Create a binary mask of already-labeled regions.
        
        Args:
            image_height: Height of the image
            image_width: Width of the image
            shapes: List of shape dictionaries from LabelMe JSON
            
        Returns:
            Binary mask where labeled regions are white (255)
        """
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for shape in shapes:
            shape_type = shape.get('shape_type', 'polygon')
            points = shape.get('points', [])
            
            if not points:
                continue
            
            # Convert points to numpy array format
            pts = np.array(points, dtype=np.int32)
            
            try:
                if shape_type == 'polygon':
                    cv2.fillPoly(mask, [pts], 255)
                elif shape_type == 'rectangle':
                    # Rectangle points: [[x1, y1], [x2, y2]]
                    if len(pts) >= 2:
                        pt1 = tuple(pts[0])
                        pt2 = tuple(pts[1])
                        cv2.rectangle(mask, pt1, pt2, 255, -1)
                elif shape_type == 'circle':
                    # Circle: points contain center and a point on circle
                    if len(pts) >= 2:
                        center = tuple(pts[0])
                        radius = int(np.linalg.norm(pts[1] - pts[0]))
                        cv2.circle(mask, center, radius, 255, -1)
            except Exception as e:
                logger.warning(f"Error processing shape {shape.get('label')}: {e}")
                continue
        
        return mask
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply exclusion mask to image (set masked regions to black).
        
        Args:
            image: Original image
            mask: Binary exclusion mask
            
        Returns:
            Masked image with labeled regions set to black
        """
        # Invert mask: we want to keep areas NOT in mask
        mask_inv = cv2.bitwise_not(mask)
        
        if len(image.shape) == 3:  # Color image
            masked_image = cv2.bitwise_and(image, image, mask=mask_inv)
        else:  # Grayscale
            masked_image = cv2.bitwise_and(image, mask_inv)
        
        return masked_image
    
    def detect_silkscreen_by_color(self, masked_image: np.ndarray) -> np.ndarray:
        """
        Detect silkscreen regions using color thresholding.
        
        Args:
            masked_image: Image with existing labels masked out
            
        Returns:
            Binary image with detected silkscreen regions
        """
        # Create color range for silkscreen
        lower = np.array(self.config.color_lower, dtype=np.uint8)
        upper = np.array(self.config.color_upper, dtype=np.uint8)
        
        # Threshold image by color
        binary = cv2.inRange(masked_image, lower, upper)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.config.kernel_size, self.config.kernel_size)
        )
        
        # Close operation: fill small holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 
                                 iterations=self.config.morph_iterations)
        
        # Open operation: remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                 iterations=self.config.morph_iterations)
        
        return binary
    
    def detect_silkscreen_by_edge(self, masked_image: np.ndarray) -> np.ndarray:
        """
        Detect silkscreen regions using edge detection (fallback method).
        
        Args:
            masked_image: Image with existing labels masked out
            
        Returns:
            Binary image with detected silkscreen regions
        """
        # Convert to grayscale if needed
        if len(masked_image.shape) == 3:
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = masked_image
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.config.edge_threshold1, 
                         self.config.edge_threshold2)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.kernel_size, self.config.kernel_size)
        )
        binary = cv2.dilate(edges, kernel, iterations=self.config.morph_iterations)
        
        return binary
    
    def extract_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from binary image.
        
        Args:
            binary_image: Binary image with detected regions
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_contour_area <= area <= self.config.max_contour_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def approximate_shape(self, contour: np.ndarray) -> np.ndarray:
        """
        Approximate contour to polygon with fewer points.
        
        Args:
            contour: Original contour
            
        Returns:
            Approximated contour points
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.config.epsilon_percentage * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx.reshape(-1, 2).tolist()
    
    def create_labelme_shape(self, polygon_points: List, label: str = "silkscreen") -> Dict:
        """
        Create a LabelMe shape dictionary from polygon points.
        
        Args:
            polygon_points: List of [x, y] coordinates
            label: Label name (default: "silkscreen")
            
        Returns:
            LabelMe shape dictionary
        """
        return {
            "label": label,
            "points": polygon_points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        }
    
    def detect_silkscreen(self, image_path: str, json_path: str, 
                         skip_existing: bool = False) -> Tuple[List[Dict], int]:
        """
        Detect silkscreen regions in image with existing annotations.
        
        Args:
            image_path: Path to PCB image
            json_path: Path to LabelMe JSON file
            skip_existing: If True, skip images that already have silkscreen labels
            
        Returns:
            Tuple of (detected shapes list, number of silkscreen regions found)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return [], 0
            
            self.image_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Load existing annotations
            labelme_data = self.load_labelme_json(json_path)
            existing_shapes = labelme_data.get('shapes', [])
            
            # Check for existing silkscreen labels
            if skip_existing:
                existing_silkscreen = [s for s in existing_shapes if s.get('label') == 'silkscreen']
                if existing_silkscreen:
                    logger.info(f"Skipping {image_path}: already has silkscreen labels")
                    return [], 0
            
            # Step 1 & 2: Create exclusion mask from existing labels
            mask = self.create_exclusion_mask(image.shape[0], image.shape[1], existing_shapes)
            
            # Step 3: Apply mask to image
            masked_image = self.apply_mask(image, mask)
            
            # Step 4: Detect silkscreen in unmasked regions
            binary = self.detect_silkscreen_by_color(masked_image)
            
            # Fallback to edge detection if color detection finds nothing
            if cv2.countNonZero(binary) < 100:
                if self.config.use_edge_detection:
                    logger.info(f"Color detection insufficient, using edge detection: {image_path}")
                    binary = self.detect_silkscreen_by_edge(masked_image)
            
            # Extract contours
            contours = self.extract_contours(binary)
            
            # Step 5: Create shape dictionaries for detected regions
            detected_shapes = []
            for contour in contours:
                polygon_points = self.approximate_shape(contour)
                if len(polygon_points) >= 3:  # Valid polygon needs at least 3 points
                    shape = self.create_labelme_shape(polygon_points, "silkscreen")
                    detected_shapes.append(shape)
            
            logger.info(f"Detected {len(detected_shapes)} silkscreen regions in {image_path}")
            
            return detected_shapes, len(detected_shapes)
            
        except Exception as e:
            logger.error(f"Error detecting silkscreen in {image_path}: {e}")
            return [], 0
    
    def update_json_with_detections(self, json_path: str, detected_shapes: List[Dict],
                                    keep_existing: bool = True) -> None:
        """
        Update LabelMe JSON with detected silkscreen regions.
        
        Args:
            json_path: Path to LabelMe JSON file
            detected_shapes: List of detected shape dictionaries
            keep_existing: If True, keep existing labels; if False, replace
        """
        try:
            labelme_data = self.load_labelme_json(json_path)
            
            if keep_existing:
                # Remove any existing silkscreen labels
                existing_shapes = [s for s in labelme_data.get('shapes', []) 
                                 if s.get('label') != 'silkscreen']
                # Add new detections
                labelme_data['shapes'] = existing_shapes + detected_shapes
            else:
                labelme_data['shapes'] = detected_shapes
            
            self.save_labelme_json(json_path, labelme_data)
            
        except Exception as e:
            logger.error(f"Error updating JSON {json_path}: {e}")
            raise


class SilkscreenProcessor:
    """Main processor for batch processing PCB images"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 config: Optional[DetectionConfig] = None):
        """
        Initialize the processor.
        
        Args:
            input_dir: Root input directory
            output_dir: Output directory for results
            config: Detection configuration
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or DetectionConfig()
        self.detector = SilkscreenDetector(self.config)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        logger.info(f"Initialized processor: input={input_dir}, output={output_dir}")
    
    def find_image_annotation_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find all image-annotation pairs in input directory (including subdirectories).
        
        Returns:
            List of tuples (image_path, json_path) for images with annotations
        """
        pairs = []
        
        # Search for JSON files recursively
        for json_path in self.input_dir.rglob('*.json'):
            # Find corresponding image file
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_found = False
            
            for ext in image_extensions:
                # Try same name with different extension
                image_path = json_path.with_suffix(ext)
                if image_path.exists():
                    pairs.append((image_path, json_path))
                    image_found = True
                    break
            
            if not image_found:
                logger.warning(f"No corresponding image found for {json_path}")
        
        logger.info(f"Found {len(pairs)} image-annotation pairs")
        return pairs
    
    def get_relative_output_path(self, file_path: Path) -> Path:
        """
        Get the relative path for output, preserving directory structure.
        
        Args:
            file_path: Original file path
            
        Returns:
            Relative output path
        """
        relative_path = file_path.relative_to(self.input_dir)
        return self.output_dir / relative_path
    
    def copy_image_to_output(self, image_path: Path, output_path: Path) -> None:
        """
        Copy image to output directory.
        
        Args:
            image_path: Source image path
            output_path: Destination image path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read and write image using OpenCV to ensure proper handling
        image = cv2.imread(str(image_path))
        if image is not None:
            cv2.imwrite(str(output_path), image)
            logger.info(f"Copied image to {output_path}")
        else:
            logger.error(f"Could not copy image: {image_path}")
    
    def process_all(self, skip_existing: bool = False) -> Dict:
        """
        Process all image-annotation pairs.
        
        Args:
            skip_existing: Skip images that already have silkscreen labels
            
        Returns:
            Dictionary with processing statistics
        """
        pairs = self.find_image_annotation_pairs()
        
        if not pairs:
            logger.warning("No image-annotation pairs found")
            return {
                'total': 0,
                'processed': 0,
                'skipped': 0,
                'detected': 0,
                'total_silkscreen': 0,
                'errors': 0
            }
        
        stats = {
            'total': len(pairs),
            'processed': 0,
            'skipped': 0,
            'detected': 0,
            'total_silkscreen': 0,
            'errors': 0
        }
        
        logger.info(f"Starting processing of {len(pairs)} image-annotation pairs")
        
        for idx, (image_path, json_path) in enumerate(pairs, 1):
            logger.info(f"[{idx}/{len(pairs)}] Processing {image_path.name}")
            
            try:
                # Detect silkscreen
                detected_shapes, num_detections = self.detector.detect_silkscreen(
                    str(image_path), str(json_path), skip_existing=skip_existing
                )
                
                if skip_existing and num_detections == 0:
                    # Check if this was skipped due to existing labels
                    labelme_data = self.detector.load_labelme_json(str(json_path))
                    existing_silkscreen = [s for s in labelme_data.get('shapes', []) 
                                         if s.get('label') == 'silkscreen']
                    if existing_silkscreen:
                        stats['skipped'] += 1
                        continue
                
                # Update JSON with detections
                if detected_shapes:
                    self.detector.update_json_with_detections(
                        str(json_path), detected_shapes, keep_existing=True
                    )
                    stats['detected'] += 1
                    stats['total_silkscreen'] += num_detections
                
                # Copy image and JSON to output
                output_image_path = self.get_relative_output_path(image_path)
                output_json_path = self.get_relative_output_path(json_path)
                
                self.copy_image_to_output(image_path, output_image_path)
                
                # Copy updated JSON
                output_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, 'r') as src:
                    json_data = json.load(src)
                with open(output_json_path, 'w') as dst:
                    json.dump(json_data, dst, indent=2)
                logger.info(f"Saved annotations to {output_json_path}")
                
                stats['processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                stats['errors'] += 1
                continue
        
        return stats


def load_config_from_file(config_path: str) -> DetectionConfig:
    """
    Load detection configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        DetectionConfig object
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Handle nested color configs
        if 'color_lower' in config_dict and isinstance(config_dict['color_lower'], dict):
            # Support both tuple and dict formats
            pass
        
        return DetectionConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return DetectionConfig()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Detect and label silkscreen regions in PCB images with existing annotations'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Root input directory containing labeled images and annotations')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for results')
    parser.add_argument('--config', '-c', default=None,
                       help='Configuration JSON file (optional)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have silkscreen labels')
    parser.add_argument('--color-lower', nargs=3, type=int, default=None,
                       help='Lower color range (B G R), e.g., 200 200 200')
    parser.add_argument('--color-upper', nargs=3, type=int, default=None,
                       help='Upper color range (B G R), e.g., 255 255 255')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = DetectionConfig()
    
    # Override color range if specified
    if args.color_lower:
        config.color_lower = tuple(args.color_lower)
    if args.color_upper:
        config.color_upper = tuple(args.color_upper)
    
    logger.info(f"Configuration: {asdict(config)}")
    
    # Process images
    try:
        processor = SilkscreenProcessor(args.input, args.output, config)
        stats = processor.process_all(skip_existing=args.skip_existing)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Processing Summary:")
        logger.info(f"  Total pairs: {stats['total']}")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Detected silkscreen: {stats['detected']}")
        logger.info(f"  Total silkscreen regions: {stats['total_silkscreen']}")
        logger.info(f"  Skipped (existing labels): {stats['skipped']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info("=" * 60)
        
        if stats['errors'] == 0 and stats['processed'] > 0:
            logger.info("Processing completed successfully!")
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
