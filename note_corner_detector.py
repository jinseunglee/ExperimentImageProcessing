#!/usr/bin/env python3
"""
Note Corner Detector

This script detects corners in note paper images for perspective correction
and grid analysis. It uses various computer vision techniques to identify
the four corners of squared note paper.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class NoteCornerDetector:
    def __init__(self, debug_mode=False):
        """
        Initialize the corner detector.
        
        Args:
            debug_mode (bool): Enable debug output and visualization
        """
        self.debug_mode = debug_mode
        
    def load_image(self, image_path):
        """Load and preprocess the image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large for processing
        height, width = image_rgb.shape[:2]
        if width > 1200 or height > 1200:
            scale = min(1200/width, 1200/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
        return image_rgb
    
    def preprocess_image(self, image):
        """
        Preprocess image for corner detection.
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_corners_harris(self, image, max_corners=100, quality=0.01, min_distance=10):
        """
        Detect corners using Harris corner detection.
        
        Args:
            image: Preprocessed grayscale image
            max_corners: Maximum number of corners to detect
            quality: Minimum quality of corner below which everyone is rejected
            min_distance: Minimum possible euclidean distance between corners
            
        Returns:
            Array of corner coordinates
        """
        corners = cv2.goodFeaturesToTrack(
            image, max_corners, quality, min_distance
        )
        
        if corners is not None:
            corners = np.int0(corners)
            return corners.reshape(-1, 2)
        else:
            return np.array([])
    
    def detect_corners_contours(self, image):
        """
        Detect corners using contour analysis.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Array of corner coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.array([])
        
        # Find the largest contour (assumed to be the note paper)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have 4 points, return them as corners
        if len(approx) == 4:
            return approx.reshape(-1, 2)
        
        # If not exactly 4 points, try to find the best 4 corners
        if len(approx) > 4:
            # Use convex hull to get the outer boundary
            hull = cv2.convexHull(approx)
            # Simplify to 4 points
            epsilon = 0.1 * cv2.arcLength(hull, True)
            simplified = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(simplified) == 4:
                return simplified.reshape(-1, 2)
        
        return np.array([])
    
    def order_corners(self, corners):
        """
        Order corners in clockwise order starting from top-left.
        
        Args:
            corners: Array of 4 corner coordinates
            
        Returns:
            Ordered array of corners [top-left, top-right, bottom-right, bottom-left]
        """
        if len(corners) != 4:
            return corners
        
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Separate corners into top and bottom based on y-coordinate
        top = []
        bottom = []
        
        for corner in corners:
            if corner[1] < centroid[1]:
                top.append(corner)
            else:
                bottom.append(corner)
        
        # Sort top corners by x-coordinate (left to right)
        top = sorted(top, key=lambda x: x[0])
        bottom = sorted(bottom, key=lambda x: x[0])
        
        # Return in order: top-left, top-right, bottom-right, bottom-left
        return np.array([top[0], top[1], bottom[1], bottom[0]])
    
    def detect_grid_corners(self, image):
        """
        Detect corners specifically for grid/note paper.
        
        Args:
            image: Input RGB image
            
        Returns:
            Ordered array of 4 corner coordinates
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        if self.debug_mode:
            print("Preprocessing completed")
        
        # Try Harris corner detection first
        harris_corners = self.detect_corners_harris(preprocessed)
        
        if self.debug_mode:
            print(f"Harris corners detected: {len(harris_corners)}")
        
        # Try contour-based detection
        contour_corners = self.detect_corners_contours(preprocessed)
        
        if self.debug_mode:
            print(f"Contour corners detected: {len(contour_corners)}")
        
        # Use the method that found exactly 4 corners
        if len(contour_corners) == 4:
            corners = contour_corners
            method = "contour"
        elif len(harris_corners) >= 4:
            # Select the 4 corners with highest response
            corners = harris_corners[:4]
            method = "harris"
        else:
            # Fallback: use all detected corners
            corners = np.vstack([harris_corners, contour_corners]) if len(harris_corners) > 0 and len(contour_corners) > 0 else np.vstack([harris_corners]) if len(harris_corners) > 0 else contour_corners
            method = "combined"
        
        if self.debug_mode:
            print(f"Using {method} method with {len(corners)} corners")
        
        # If we have exactly 4 corners, order them
        if len(corners) == 4:
            ordered_corners = self.order_corners(corners)
            return ordered_corners, method
        else:
            return corners, method
    
    def calculate_perspective_transform(self, corners, target_size=(800, 600)):
        """
        Calculate perspective transform matrix.
        
        Args:
            corners: Ordered corner coordinates
            target_size: Target image size (width, height)
            
        Returns:
            Perspective transform matrix
        """
        if len(corners) != 4:
            return None
        
        # Define target corners
        target_corners = np.array([
            [0, 0],                    # Top-left
            [target_size[0], 0],       # Top-right
            [target_size[0], target_size[1]],  # Bottom-right
            [0, target_size[1]]        # Bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32), target_corners
        )
        
        return matrix
    
    def apply_perspective_transform(self, image, corners, target_size=(800, 600)):
        """
        Apply perspective transform to correct the image.
        
        Args:
            image: Input image
            corners: Corner coordinates
            target_size: Target image size
            
        Returns:
            Corrected image
        """
        matrix = self.calculate_perspective_transform(corners, target_size)
        
        if matrix is None:
            return image
        
        # Apply transform
        corrected = cv2.warpPerspective(image, matrix, target_size)
        
        return corrected
    
    def visualize_corners(self, image, corners, method="unknown"):
        """
        Create visualization of detected corners.
        
        Args:
            image: Input image
            corners: Detected corner coordinates
            method: Detection method used
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original image with corners
        axes[0].imshow(image)
        axes[0].set_title(f'Original Image with Detected Corners\nMethod: {method}')
        axes[0].axis('off')
        
        if len(corners) > 0:
            # Draw corners
            for i, corner in enumerate(corners):
                axes[0].plot(corner[0], corner[1], 'ro', markersize=10)
                axes[0].text(corner[0] + 10, corner[1] + 10, f'C{i+1}', 
                            color='red', fontsize=12, fontweight='bold')
            
            # Draw lines connecting corners if we have 4
            if len(corners) == 4:
                for i in range(4):
                    pt1 = corners[i]
                    pt2 = corners[(i + 1) % 4]
                    axes[0].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2)
        
        # Preprocessed image
        preprocessed = self.preprocess_image(image)
        axes[1].imshow(preprocessed, cmap='gray')
        axes[1].set_title('Preprocessed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def process_image(self, image_path, save_visualization=True):
        """
        Main method to process an image and detect corners.
        
        Args:
            image_path: Path to input image
            save_visualization: Whether to save visualization
            
        Returns:
            Dictionary with results
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = self.load_image(image_path)
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Detect corners
        corners, method = self.detect_grid_corners(image)
        
        print(f"\nCorner Detection Results:")
        print(f"Method used: {method}")
        print(f"Corners detected: {len(corners)}")
        
        if len(corners) > 0:
            print(f"Corner coordinates:")
            for i, corner in enumerate(corners):
                print(f"  Corner {i+1}: ({corner[0]}, {corner[1]})")
        
        # Calculate perspective transform if we have 4 corners
        perspective_matrix = None
        corrected_image = None
        
        if len(corners) == 4:
            print(f"\nPerspective correction available")
            perspective_matrix = self.calculate_perspective_transform(corners)
            
            if perspective_matrix is not None:
                corrected_image = self.apply_perspective_transform(image, corners)
                print(f"Perspective transform matrix calculated")
        
        # Create visualization
        if save_visualization:
            fig = self.visualize_corners(image, corners, method)
            
            # Save visualization
            output_path = image_path.rsplit('.', 1)[0] + '_corner_detection.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
            
            # Show plot
            plt.show()
        
        return {
            'corners': corners,
            'method': method,
            'perspective_matrix': perspective_matrix,
            'corrected_image': corrected_image,
            'image_shape': image.shape
        }


def main():
    """Main function to run the corner detector."""
    parser = argparse.ArgumentParser(
        description='Detect corners in note paper images for perspective correction'
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization and save')
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = NoteCornerDetector(debug_mode=args.debug)
        
        # Process image
        results = detector.process_image(
            args.image_path, 
            save_visualization=not args.no_viz
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
