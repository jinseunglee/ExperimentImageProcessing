#!/usr/bin/env python3
"""
Note Corner Detector - OpenCV Version

This script detects corners in note paper images using analytical computer vision
techniques with OpenCV. It provides robust corner detection without external API dependencies.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class NoteCornerDetectorOpenCV:
    def __init__(self, debug_mode=False):
        """
        Initialize the OpenCV-based corner detector.
        
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
            corners = np.int32(corners)
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
    
    def detect_corners_edges(self, image):
        """
        Detect corners using edge detection and line intersection.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Array of corner coordinates
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find lines using Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return np.array([])
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            
            # Check if line is approximately horizontal (0° or 180°)
            if abs(angle_deg) < 10 or abs(angle_deg - 180) < 10:
                horizontal_lines.append((rho, theta))
            # Check if line is approximately vertical (90°)
            elif abs(angle_deg - 90) < 10:
                vertical_lines.append((rho, theta))
        
        # Find intersections of horizontal and vertical lines
        corners = []
        for h_line in horizontal_lines[:2]:  # Use top 2 horizontal lines
            for v_line in vertical_lines[:2]:  # Use top 2 vertical lines
                # Calculate intersection point
                h_rho, h_theta = h_line
                v_rho, v_theta = v_line
                
                # Convert polar to Cartesian coordinates
                h_a = np.cos(h_theta)
                h_b = np.sin(h_theta)
                v_a = np.cos(v_theta)
                v_b = np.sin(v_theta)
                
                # Solve intersection: h_a*x + h_b*y = h_rho, v_a*x + v_b*y = v_rho
                det = h_a * v_b - h_b * v_a
                if abs(det) > 1e-6:  # Check if lines are not parallel
                    x = (h_rho * v_b - h_b * v_rho) / det
                    y = (h_a * v_rho - h_rho * v_a) / det
                    
                    # Check if intersection is within image bounds
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        corners.append([int(x), int(y)])
        
        return np.array(corners)
    
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
        Detect corners using multiple OpenCV-based approaches.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (corners array, method description)
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        if self.debug_mode:
            print("Preprocessing completed")
        
        # Try multiple detection methods
        methods_results = []
        
        # Method 1: Harris corner detection
        harris_corners = self.detect_corners_harris(preprocessed)
        methods_results.append(("harris", harris_corners))
        
        if self.debug_mode:
            print(f"Harris corners detected: {len(harris_corners)}")
        
        # Method 2: Contour-based detection
        contour_corners = self.detect_corners_contours(preprocessed)
        methods_results.append(("contour", contour_corners))
        
        if self.debug_mode:
            print(f"Contour corners detected: {len(contour_corners)}")
        
        # Method 3: Edge-based detection
        edge_corners = self.detect_corners_edges(preprocessed)
        methods_results.append(("edge", edge_corners))
        
        if self.debug_mode:
            print(f"Edge-based corners detected: {len(edge_corners)}")
        
        # Select the best method that found exactly 4 corners
        best_corners = np.array([])
        best_method = "none"
        
        for method_name, corners in methods_results:
            if len(corners) == 4:
                best_corners = corners
                best_method = method_name
                break
        
        # If no method found exactly 4 corners, use the one with most corners
        if len(best_corners) != 4:
            max_corners = 0
            for method_name, corners in methods_results:
                if len(corners) > max_corners:
                    max_corners = len(corners)
                    best_corners = corners
                    best_method = method_name
        
        if self.debug_mode:
            print(f"Selected {best_method} method with {len(best_corners)} corners")
        
        # If we have exactly 4 corners, order them
        if len(best_corners) == 4:
            ordered_corners = self.order_corners(best_corners)
            return ordered_corners, best_method
        else:
            return best_corners, best_method
    
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
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
        
        # Edge detection visualization
        edges = cv2.Canny(preprocessed, 50, 150)
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Edge Detection')
        axes[2].axis('off')
        
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
        print(f"Using OpenCV-based analytical corner detection")
        
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
            output_path = image_path.rsplit('.', 1)[0] + '_opencv_corner_detection.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
            
            # Show plot
            plt.show()
        
        return {
            'corners': corners,
            'method': method,
            'perspective_matrix': perspective_matrix,
            'corrected_image': corrected_image,
            'image_shape': image.shape,
            'detector_type': 'opencv'
        }


def main():
    """Main function to run the OpenCV corner detector."""
    parser = argparse.ArgumentParser(
        description='Detect corners in note paper images using OpenCV (analytical approach)'
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization and save')
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = NoteCornerDetectorOpenCV(debug_mode=args.debug)
        
        # Process image
        results = detector.process_image(
            args.image_path, 
            save_visualization=not args.no_viz
        )
        
        print("\nProcessing completed successfully!")
        print(f"Detector type: {results['detector_type']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
