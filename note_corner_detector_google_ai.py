#!/usr/bin/env python3
"""
Note Corner Detector - Google Cloud AI Version

This script detects corners in note paper images using Google Cloud AI Vision API.
It provides advanced document analysis and corner detection through cloud-based AI services.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import requests
import base64
from typing import Optional, Tuple, List


class NoteCornerDetectorGoogleAI:
    def __init__(self, api_key: str, api_endpoint: str = None, debug_mode: bool = False):
        """
        Initialize the Google AI-based corner detector.
        
        Args:
            api_key (str): Google Cloud AI API key
            api_endpoint (str): Google Cloud AI API endpoint URL
            debug_mode (bool): Enable debug output and visualization
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint or "https://vision.googleapis.com/v1/images:annotate"
        self.debug_mode = debug_mode
        
        if not self.api_key:
            raise ValueError("Google Cloud AI API key is required")
    
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
    
    def encode_image_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string for API request.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, 'rb') as image_file:
            image_content = base64.b64encode(image_file.read()).decode('utf-8')
        return image_content
    
    def detect_corners_document_text(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Detect corners using Google Vision API document text detection.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (corners array, method description)
        """
        try:
            # Encode image
            image_content = self.encode_image_base64(image_path)
            
            # Prepare API request for document text detection
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_content
                        },
                        "features": [
                            {
                                "type": "DOCUMENT_TEXT_DETECTION",
                                "maxResults": 1
                            }
                        ]
                    }
                ]
            }
            
            # Make API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            if self.debug_mode:
                print("Making Google Vision API request for document text detection...")
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                if self.debug_mode:
                    print(f"Google API error: {response.status_code} - {response.text}")
                return np.array([]), f"api_error_{response.status_code}"
            
            # Parse response
            result = response.json()
            corners = self._extract_corners_from_document_text(result)
            
            if len(corners) == 4:
                return corners, "google_ai_document_text"
            else:
                return corners, "google_ai_document_text_partial"
                
        except Exception as e:
            if self.debug_mode:
                print(f"Document text detection error: {e}")
            return np.array([]), f"document_text_exception: {str(e)}"
    
    def detect_corners_object_localization(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Detect corners using Google Vision API object localization.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (corners array, method description)
        """
        try:
            # Encode image
            image_content = self.encode_image_base64(image_path)
            
            # Prepare API request for object localization
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_content
                        },
                        "features": [
                            {
                                "type": "OBJECT_LOCALIZATION",
                                "maxResults": 20
                            }
                        ]
                    }
                ]
            }
            
            # Make API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            if self.debug_mode:
                print("Making Google Vision API request for object localization...")
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                if self.debug_mode:
                    print(f"Google API error: {response.status_code} - {response.text}")
                return np.array([]), f"api_error_{response.status_code}"
            
            # Parse response
            result = response.json()
            corners = self._extract_corners_from_object_localization(result)
            
            if len(corners) == 4:
                return corners, "google_ai_object_localization"
            else:
                return corners, "google_ai_object_localization_partial"
                
        except Exception as e:
            if self.debug_mode:
                print(f"Object localization error: {e}")
            return corners, f"object_localization_exception: {str(e)}"
    
    def detect_corners_web_detection(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Detect corners using Google Vision API web detection.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (corners array, method description)
        """
        try:
            # Encode image
            image_content = self.encode_image_base64(image_path)
            
            # Prepare API request for web detection
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_content
                        },
                        "features": [
                            {
                                "type": "WEB_DETECTION",
                                "maxResults": 5
                            }
                        ]
                    }
                ]
            }
            
            # Make API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            if self.debug_mode:
                print("Making Google Vision API request for web detection...")
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                if self.debug_mode:
                    print(f"Google API error: {response.status_code} - {response.text}")
                return np.array([]), f"api_error_{response.status_code}"
            
            # Parse response
            result = response.json()
            corners = self._extract_corners_from_web_detection(result)
            
            if len(corners) == 4:
                return corners, "google_ai_web_detection"
            else:
                return corners, "google_ai_web_detection_partial"
                
        except Exception as e:
            if self.debug_mode:
                print(f"Web detection error: {e}")
            return np.array([]), f"web_detection_exception: {str(e)}"
    
    def _extract_corners_from_document_text(self, response_data: dict) -> np.ndarray:
        """
        Extract corner coordinates from document text detection response.
        
        Args:
            response_data: Response from Google Vision API
            
        Returns:
            Array of corner coordinates
        """
        corners = []
        
        try:
            if 'responses' in response_data and len(response_data['responses']) > 0:
                response = response_data['responses'][0]
                
                # Check for document text detection
                if 'textAnnotations' in response and len(response['textAnnotations']) > 0:
                    # First annotation contains the entire document bounds
                    doc_bounds = response['textAnnotations'][0].get('boundingPoly', {})
                    vertices = doc_bounds.get('vertices', [])
                    
                    if len(vertices) >= 4:
                        for vertex in vertices:
                            if 'x' in vertex and 'y' in vertex:
                                corners.append([vertex['x'], vertex['y']])
                        
                        if self.debug_mode:
                            print(f"Extracted {len(corners)} corners from document text detection")
            
            # Convert to numpy array
            if len(corners) >= 4:
                corners = np.array(corners[:4], dtype=np.int32)
                return corners
            else:
                return np.array([])
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting corners from document text: {e}")
            return np.array([])
    
    def _extract_corners_from_object_localization(self, response_data: dict) -> np.ndarray:
        """
        Extract corner coordinates from object localization response.
        
        Args:
            response_data: Response from Google Vision API
            
        Returns:
            Array of corner coordinates
        """
        corners = []
        
        try:
            if 'responses' in response_data and len(response_data['responses']) > 0:
                response = response_data['responses'][0]
                
                if 'localizedObjectAnnotations' in response:
                    # Look for document-like objects
                    document_objects = []
                    for obj in response['localizedObjectAnnotations']:
                        name = obj.get('name', '').lower()
                        if any(keyword in name for keyword in ['document', 'paper', 'note', 'page']):
                            document_objects.append(obj)
                    
                    # Use the first document-like object found
                    if document_objects:
                        obj = document_objects[0]
                        bounding_poly = obj.get('boundingPoly', {})
                        vertices = bounding_poly.get('vertices', [])
                        
                        if len(vertices) >= 4:
                            for vertex in vertices:
                                if 'x' in vertex and 'y' in vertex:
                                    corners.append([vertex['x'], vertex['y']])
                        
                        if self.debug_mode:
                            print(f"Extracted {len(corners)} corners from object localization")
            
            # Convert to numpy array
            if len(corners) >= 4:
                corners = np.array(corners[:4], dtype=np.int32)
                return corners
            else:
                return np.array([])
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting corners from object localization: {e}")
            return np.array([])
    
    def _extract_corners_from_web_detection(self, response_data: dict) -> np.ndarray:
        """
        Extract corner coordinates from web detection response.
        
        Args:
            response_data: Response from Google Vision API
            
        Returns:
            Array of corner coordinates
        """
        # Web detection doesn't provide bounding boxes, so return empty
        # This method is included for completeness but won't extract corners
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
    
    def detect_grid_corners(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Detect corners using multiple Google AI approaches.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (corners array, method description)
        """
        if self.debug_mode:
            print("Starting Google AI-based corner detection...")
        
        # Try multiple detection methods
        methods_results = []
        
        # Method 1: Document text detection (most reliable for documents)
        doc_corners, doc_method = self.detect_corners_document_text(image_path)
        methods_results.append((doc_method, doc_corners))
        
        # Method 2: Object localization
        obj_corners, obj_method = self.detect_corners_object_localization(image_path)
        methods_results.append((obj_method, obj_corners))
        
        # Method 3: Web detection (for completeness)
        web_corners, web_method = self.detect_corners_web_detection(image_path)
        methods_results.append((web_method, web_corners))
        
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original image with corners
        axes[0].imshow(image)
        axes[0].set_title(f'Original Image with Google AI Detected Corners\nMethod: {method}')
        axes[0].axis('off')
        
        if len(corners) > 0:
            # Draw corners
            for i, corner in enumerate(corners):
                axes[0].plot(corner[0], corner[1], 'go', markersize=12)  # Green for Google AI
                axes[0].text(corner[0] + 10, corner[1] + 10, f'C{i+1}', 
                            color='green', fontsize=14, fontweight='bold')
            
            # Draw lines connecting corners if we have 4
            if len(corners) == 4:
                for i in range(4):
                    pt1 = corners[i]
                    pt2 = corners[(i + 1) % 4]
                    axes[0].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=3)
        
        # API response visualization placeholder
        axes[1].text(0.5, 0.5, f'Google AI Detection\nMethod: {method}\nCorners: {len(corners)}', 
                    ha='center', va='center', fontsize=16, transform=axes[1].transAxes)
        axes[1].set_title('Google AI Analysis Results')
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
        print(f"Using Google Cloud AI Vision API for corner detection")
        
        # Load image for visualization
        image = self.load_image(image_path)
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Detect corners using Google AI
        corners, method = self.detect_grid_corners(image_path)
        
        print(f"\nGoogle AI Corner Detection Results:")
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
            output_path = image_path.rsplit('.', 1)[0] + '_google_ai_corner_detection.png'
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
            'detector_type': 'google_ai'
        }


def main():
    """Main function to run the Google AI corner detector."""
    parser = argparse.ArgumentParser(
        description='Detect corners in note paper images using Google Cloud AI Vision API'
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--api-key', required=True, help='Google Cloud AI API key')
    parser.add_argument('--api-endpoint', help='Google Cloud AI API endpoint URL (optional)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization and save')
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = NoteCornerDetectorGoogleAI(
            api_key=args.api_key,
            api_endpoint=args.api_endpoint,
            debug_mode=args.debug
        )
        
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
