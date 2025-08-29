#!/usr/bin/env python3
"""
Note Corner Detector using DocAligner

This script detects corners in note paper images using DocAligner library
and performs perspective transformation to a top-down view.

Requirements:
- pip install docaligner
- pip install opencv-python numpy matplotlib

Author: AI Assistant
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from typing import Tuple, Optional, List
import time

try:
    from docaligner import DocAligner
    DOCALIGNER_AVAILABLE = True
except ImportError:
    DOCALIGNER_AVAILABLE = False
    print("Warning: DocAligner not available. Install with: pip install docaligner-docsaid")

class NoteCornerDetectorDocAligner:
    """
    Corner detector using DocAligner library for document alignment.
    
    Features:
    - Automatic corner detection using DocAligner
    - Perspective transformation to top-down view
    - Corner ordering and validation
    - Visualization and debugging
    - Support for various image formats
    """
    
    def __init__(self, debug_mode: bool = False, cache_dir: str = None):
        """
        Initialize the DocAligner corner detector.
        
        Args:
            debug_mode: Enable debug output
            cache_dir: Directory for caching (not used in this implementation)
        """
        self.debug_mode = debug_mode
        self.cache_dir = cache_dir
        
        if not DOCALIGNER_AVAILABLE:
            raise ImportError("DocAligner is not available. Please install it with: pip install docaligner")
        
        # Initialize DocAligner
        try:
            self.docaligner = DocAligner()
            if self.debug_mode:
                print("DocAligner initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DocAligner: {e}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess the image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.debug_mode:
            height, width = image.shape[:2]
            print(f"Image loaded: {width}x{height} pixels")
        
        return image, image_rgb
    
    def detect_corners_docaligner(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Detect corners using DocAligner.
        
        Args:
            image: Input image (BGR format for OpenCV)
            
        Returns:
            Tuple of (corners, method_name)
        """
        try:
            if self.debug_mode:
                print("Starting DocAligner corner detection...")
                start_time = time.time()
            
            # Convert BGR to RGB for DocAligner
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect document boundaries using DocAligner
            # According to the GitHub repo: result = model(img) returns 4 corner points
            result = self.docaligner(image_rgb)
            print(result)
            
            if self.debug_mode:
                elapsed_time = time.time() - start_time
                print(f"DocAligner detection completed in {elapsed_time:.3f} seconds")
            
            # Extract corners from DocAligner result
            corners = self._extract_corners_from_docaligner(result)
            
            if self.debug_mode:
                print(f"DocAligner extracted {len(corners)} corners")
            
            return corners, "docaligner"
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in DocAligner detection: {e}")
            return np.array([]), "docaligner_error"
    
    def _extract_corners_from_docaligner(self, result) -> np.ndarray:
        """
        Extract corner coordinates from DocAligner result.
        
        Args:
            result: DocAligner detection result (numpy array of 4 corner points)
            
        Returns:
            Array of corner coordinates
        """
        try:
            # According to the GitHub repo, DocAligner returns corners directly as numpy array
            # result format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            if isinstance(result, np.ndarray):
                corners = result
            elif hasattr(result, 'corners'):
                corners = result.corners
            elif hasattr(result, 'contour'):
                # If result has contour, approximate to get corners
                contour = result.contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                corners = approx.reshape(-1, 2)
            elif hasattr(result, 'polygon'):
                corners = result.polygon.reshape(-1, 2)
            else:
                # Fallback: try to find the largest contour
                if hasattr(result, 'contours'):
                    contours = result.contours
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        corners = approx.reshape(-1, 2)
                    else:
                        corners = np.array([])
                else:
                    corners = np.array([])
            
            # Ensure we have the right number of corners
            if len(corners) > 4:
                # If more than 4 corners, try to simplify
                epsilon = 0.02 * cv2.arcLength(corners.astype(np.float32), True)
                approx = cv2.approxPolyDP(corners.astype(np.float32), epsilon, True)
                corners = approx.reshape(-1, 2)
            
            return corners.astype(np.float32)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting corners from DocAligner result: {e}")
            return np.array([])
    
    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in the sequence: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Array of 4 corner coordinates
            
        Returns:
            Ordered corner coordinates
        """
        if len(corners) != 4:
            return corners
        
        # Reshape to 2D array if needed
        if corners.ndim == 1:
            corners = corners.reshape(-1, 2)
        
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Separate corners above and below centroid
        top_corners = []
        bottom_corners = []
        
        for corner in corners:
            if corner[1] < centroid[1]:
                top_corners.append(corner)
            else:
                bottom_corners.append(corner)
        
        # Sort top corners (left to right)
        top_corners.sort(key=lambda x: x[0])
        
        # Sort bottom corners (left to right)
        bottom_corners.sort(key=lambda x: x[0])
        
        # Ensure we have exactly 2 top and 2 bottom corners
        if len(top_corners) == 2 and len(bottom_corners) == 2:
            ordered_corners = np.array([
                top_corners[0],      # Top-left
                top_corners[1],      # Top-right
                bottom_corners[1],   # Bottom-right
                bottom_corners[0]    # Bottom-left
            ])
        else:
            # Fallback: use original corners
            ordered_corners = corners
        
        return ordered_corners.astype(np.float32)
    
    def calculate_perspective_transform(self, corners: np.ndarray, target_size: Tuple[int, int] = (900, 1200)) -> Optional[np.ndarray]:
        """
        Calculate perspective transform matrix.
        
        Args:
            corners: Ordered corner coordinates (4 corners)
            target_size: Target image size (width, height) - 900x1200 for 22.5x29.7cm aspect ratio
            
        Returns:
            Perspective transform matrix or None if invalid
        """
        if len(corners) != 4:
            return None
        
        # Ensure corners are in the right format
        if corners.ndim == 1:
            corners = corners.reshape(-1, 2)
        
        # Define target corners for top-down view (22.5x29.7cm aspect ratio)
        target_corners = np.array([
            [0, 0],                    # Top-left
            [target_size[0], 0],       # Top-right
            [target_size[0], target_size[1]],  # Bottom-right
            [0, target_size[1]]        # Bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        try:
            if self.debug_mode:
                print(f"Source corners: {corners.tolist()}")
                print(f"Target corners: {target_corners.tolist()}")
            
            matrix = cv2.getPerspectiveTransform(
                corners.astype(np.float32), target_corners
            )
            
            if self.debug_mode:
                print(f"Perspective transform matrix calculated successfully")
            
            return matrix
        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating perspective transform: {e}")
            return None
    
    def apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray, 
                                 target_size: Tuple[int, int] = (900, 1200)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply perspective transform to correct the image.
        
        Args:
            image: Input image
            corners: Corner coordinates
            target_size: Target image size (900x1200 for 22.5x29.7cm aspect ratio)
            
        Returns:
            Tuple of (corrected_image, transform_matrix)
        """
        matrix = self.calculate_perspective_transform(corners, target_size)
        
        if matrix is None:
            return None, None
        
        try:
            # Apply transform
            corrected = cv2.warpPerspective(image, matrix, target_size)
            return corrected, matrix
        except Exception as e:
            if self.debug_mode:
                print(f"Error applying perspective transform: {e}")
            return None, None
    
    def visualize_corners(self, image: np.ndarray, corners: np.ndarray, method: str = "unknown") -> plt.Figure:
        """
        Create visualization of detected corners.
        
        Args:
            image: Input image (RGB format)
            corners: Detected corner coordinates
            method: Detection method name
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with corners
        ax1.imshow(image)
        if len(corners) > 0:
            # Draw corners
            for i, corner in enumerate(corners):
                ax1.plot(corner[0], corner[1], 'ro', markersize=10)
                ax1.text(corner[0] + 10, corner[1] + 10, f'{i+1}', 
                        color='red', fontsize=12, fontweight='bold')
            
            # Draw lines between corners if we have 4
            if len(corners) == 4:
                ordered_corners = self.order_corners(corners)
                for i in range(4):
                    pt1 = ordered_corners[i]
                    pt2 = ordered_corners[(i + 1) % 4]
                    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2)
        
        ax1.set_title(f'Original Image with Corners ({method})')
        ax1.axis('off')
        
        # Perspective corrected image
        if len(corners) == 4:
            corrected_image, _ = self.apply_perspective_transform(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), corners
            )
            if corrected_image is not None:
                corrected_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
                ax2.imshow(corrected_rgb)
                ax2.set_title('Perspective Corrected (Top-Down View)')
            else:
                ax2.text(0.5, 0.5, 'Transform Failed', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=16)
                ax2.set_title('Perspective Correction Failed')
        else:
            ax2.text(0.5, 0.5, f'Need 4 corners for transform\n(Found {len(corners)})', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Perspective Correction Not Available')
        
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _save_opencv_visualization(self, image: np.ndarray, corners: np.ndarray, 
                                 method: str, output_path: str):
        """
        Save OpenCV visualization (direct image with corners).
        
        Args:
            image: Input image (BGR format)
            corners: Corner coordinates
            method: Detection method
            output_path: Output file path
        """
        # Create a copy for drawing
        vis_image = image.copy()
        
        if len(corners) > 0:
            # Draw corners
            for i, corner in enumerate(corners):
                cv2.circle(vis_image, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), -1)
                cv2.putText(vis_image, str(i+1), (int(corner[0]) + 15, int(corner[1]) + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw lines between corners if we have 4
            if len(corners) == 4:
                ordered_corners = self.order_corners(corners)
                for i in range(4):
                    pt1 = tuple(map(int, ordered_corners[i]))
                    pt2 = tuple(map(int, ordered_corners[(i + 1) % 4]))
                    cv2.line(vis_image, pt1, pt2, (0, 255, 0), 3)
        
        # Save the image
        cv2.imwrite(output_path, vis_image)
    
    def process_image(self, image_path: str, save_visualization: bool = True) -> dict:
        """
        Process an image to detect corners and apply perspective transformation.
        
        Args:
            image_path: Path to the input image
            save_visualization: Whether to save visualization files
            
        Returns:
            Dictionary with processing results
        """
        try:
            if self.debug_mode:
                print(f"Processing image: {image_path}")
            
            # Load image
            image, image_rgb = self.load_image(image_path)
            image_width, image_height = image.shape[1], image.shape[0]
            
            if self.debug_mode:
                print(f"Image loaded: {image_width}x{image_height} pixels")
            
            # Detect corners using DocAligner
            corners, method = self.detect_corners_docaligner(image)
            
            if self.debug_mode:
                print(f"Corner detection completed using {method}")
                print(f"Corners detected: {len(corners)}")
            
            # Order corners if we have 4
            if len(corners) == 4:
                if self.debug_mode:
                    print(f"Original corners: {corners.tolist()}")
                
                ordered_corners = self.order_corners(corners)
                corners = ordered_corners
                
                if self.debug_mode:
                    print(f"Ordered corners: {corners.tolist()}")
                    print("Corners ordered for perspective transformation")
            
            # Calculate perspective transform
            perspective_matrix = None
            corrected_image = None
            
            if len(corners) == 4:
                if self.debug_mode:
                    print(f"Calculating perspective transform for corners: {corners.tolist()}")
                
                perspective_matrix = self.calculate_perspective_transform(corners)
                if perspective_matrix is not None:
                    corrected_image, _ = self.apply_perspective_transform(image, corners)
                    if self.debug_mode:
                        print("Perspective transformation matrix calculated")
                        print("Perspective correction applied successfully")
                else:
                    if self.debug_mode:
                        print("Failed to calculate perspective transformation matrix")
                        print("This might be due to collinear or invalid corner points")
            else:
                if self.debug_mode:
                    print(f"Perspective transformation not available (need 4 corners, found {len(corners)})")
            
            # Create visualization
            if save_visualization:
                # Create debug directory if it doesn't exist
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), '.debug')
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                    if self.debug_mode:
                        print(f"Created debug directory: {debug_dir}")
                
                # Get base filename without path and extension
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save OpenCV visualization (direct image with corners)
                opencv_output_path = os.path.join(debug_dir, f"{base_filename}_docaligner_corners_opencv.jpg")
                self._save_opencv_visualization(image, corners, method, opencv_output_path)
                print(f"\nOpenCV visualization saved to: {opencv_output_path}")
                
                # Create and save matplotlib visualization
                fig = self.visualize_corners(image_rgb, corners, method)
                matplotlib_output_path = os.path.join(debug_dir, f"{base_filename}_docaligner_corner_detection.png")
                fig.savefig(matplotlib_output_path, dpi=300, bbox_inches='tight')
                print(f"Matplotlib visualization saved to: {matplotlib_output_path}")
                
                # Show plot
                plt.show()
            
            # Final processing summary
            if self.debug_mode:
                print(f"\nProcessing Summary:")
                print(f"  Input image: {image_path}")
                print(f"  Image dimensions: {image_width} x {image_height} pixels")
                print(f"  Detection method: {method}")
                print(f"  Corners detected: {len(corners)}")
                if len(corners) > 0:
                    print(f"  Corner coordinates: {corners.tolist()}")
                    if len(corners) == 4:
                        print(f"  Perspective correction: Available")
                    else:
                        print(f"  Perspective correction: Not available (need 4 corners)")
                else:
                    print(f"  Perspective correction: Not available (no corners)")
                
                # Show debug directory information
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), '.debug')
                if os.path.exists(debug_dir):
                    debug_files = [f for f in os.listdir(debug_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    print(f"  Debug files saved: {len(debug_files)} files in {debug_dir}")
                else:
                    print(f"  Debug directory: {debug_dir} (will be created when saving)")
            
            return {
                'corners': corners,
                'method': method,
                'perspective_matrix': perspective_matrix,
                'corrected_image': corrected_image,
                'image_shape': image.shape,
                'detector_type': 'docaligner'
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error processing image: {e}")
            raise


def main():
    """Main function to run the DocAligner corner detector."""
    parser = argparse.ArgumentParser(
        description='Detect corners in note paper images using DocAligner library'
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization and save')
    
    args = parser.parse_args()
    
    try:
        # Check if DocAligner is available
        if not DOCALIGNER_AVAILABLE:
            print("Error: DocAligner is not available.")
            print("Please install it with: pip install docaligner")
            return 1
        
        # Create detector
        detector = NoteCornerDetectorDocAligner(debug_mode=args.debug)
        
        # Process image
        result = detector.process_image(
            args.image_path, 
            save_visualization=not args.no_viz
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Detector type: {result['detector_type']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
