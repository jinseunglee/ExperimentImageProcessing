#!/usr/bin/env python3
"""
Note Corner Detector using Google Cloud Vision API

This script detects the four corners of note paper images using Google Cloud Vision API.
It includes caching, coordinate scaling, debug directory management, and perspective transformation.
"""

import os
import sys
import time
import pickle
import hashlib
import argparse
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import vision
from google.oauth2 import service_account

class NoteCornerDetectorGoogleAI:
    """
    Corner detector using Google Cloud Vision API with caching and coordinate scaling.
    """
    
    def __init__(self, credentials_path: str = None, debug_mode: bool = False):
        """
        Initialize the detector.
        
        Args:
            credentials_path: Path to OAuth credentials JSON file
            debug_mode: Enable debug output
        """
        self.debug_mode = debug_mode
        self.cache_dir = ".cache"
        self.cache_expiry_hours = 24
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            if self.debug_mode:
                print(f"Created cache directory: {self.cache_dir}")
        
        # Setup Google Cloud Vision client
        self.client = self._setup_credentials(credentials_path)
        
        if self.debug_mode:
            print("Google Cloud Vision API client initialized")
    
    def _setup_credentials(self, credentials_path: str = None) -> vision.ImageAnnotatorClient:
        """
        Setup Google Cloud Vision API credentials.
        
        Args:
            credentials_path: Path to credentials JSON file
            
        Returns:
            Vision API client
        """
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                client = vision.ImageAnnotatorClient(credentials=credentials)
                if self.debug_mode:
                    print(f"Using credentials from: {credentials_path}")
            else:
                # Try to use default credentials
                client = vision.ImageAnnotatorClient()
                if self.debug_mode:
                    print("Using default credentials")
            
            return client
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error setting up credentials: {e}")
            # Return None to allow cache operations without API access
            return None
    
    def _get_cache_key(self, image_path: str) -> str:
        """
        Generate cache key for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Cache key string
        """
        # Read image file and create hash
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create SHA-256 hash of image data
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Include file modification time in cache key
        mtime = os.path.getmtime(image_path)
        cache_key = f"{image_hash}_{int(mtime)}"
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get cache file path for a cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """
        Check if cache is still valid (not expired).
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
        
        # Check if cache has expired
        cache_age = time.time() - os.path.getmtime(cache_path)
        max_age = self.cache_expiry_hours * 3600  # Convert hours to seconds
        
        return cache_age < max_age
    
    def _save_to_cache(self, cache_path: str, data: dict):
        """
        Save data to cache file.
        
        Args:
            cache_path: Path to cache file
            data: Data to cache
        """
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            if self.debug_mode:
                print(f"Saved to cache: {cache_path}")
        except Exception as e:
            if self.debug_mode:
                print(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_path: str) -> Optional[dict]:
        """
        Load data from cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Cached data or None if failed
        """
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if self.debug_mode:
                print(f"Loaded from cache: {cache_path}")
            return data
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading from cache: {e}")
            return None
    
    def _get_original_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get original image dimensions (before any resizing).
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (width, height)
        """
        # Read image directly to get original dimensions
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            return width, height
        return 0, 0
    
    def _scale_coordinates(self, corners: np.ndarray, original_dims: Tuple[int, int], 
                          current_dims: Tuple[int, int]) -> np.ndarray:
        """
        Scale coordinates from original image dimensions to current dimensions.
        
        Args:
            corners: Corner coordinates from Vision API
            original_dims: Original image dimensions (width, height)
            current_dims: Current image dimensions (width, height)
            
        Returns:
            Scaled corner coordinates
        """
        if original_dims[0] == 0 or original_dims[1] == 0:
            return corners
        
        # Calculate scaling factors
        scale_x = current_dims[0] / original_dims[0]
        scale_y = current_dims[1] / original_dims[1]
        
        # Scale coordinates
        scaled_corners = corners.copy()
        scaled_corners[:, 0] *= scale_x
        scaled_corners[:, 1] *= scale_y
        
        if self.debug_mode:
            print(f"Coordinate scaling: {original_dims} -> {current_dims}")
            print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        return scaled_corners
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (image, image_rgb)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, image_rgb
    
    def detect_corners_google_ai(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Detect corners using Google Cloud Vision API.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (corners, method_name)
        """
        # Check cache first
        cache_key = self._get_cache_key(image_path)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data and 'corners' in cached_data:
                if self.debug_mode:
                    print("Using cached results")
                return cached_data['corners'], cached_data['method']
        
        # If no valid cache, call Vision API
        if self.client is None:
            raise RuntimeError("Google Cloud Vision API client not available")
        
        try:
            if self.debug_mode:
                print("Starting Google Cloud Vision API corner detection...")
                start_time = time.time()
            
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            # Create image object
            image = vision.Image(content=content)
            
            # Perform object localization to find document boundaries
            response = self.client.object_localization(image=image)
            
            if self.debug_mode:
                elapsed_time = time.time() - start_time
                print(f"Vision API call completed in {elapsed_time:.3f} seconds")
            
            # Extract corners from response
            corners = self._extract_corners_from_vision_api(response, image_path)
            
            if self.debug_mode:
                print(f"Vision API extracted {len(corners)} corners")
            
            # Cache the results
            cache_data = {
                'corners': corners,
                'method': 'google_vision_api',
                'timestamp': time.time()
            }
            self._save_to_cache(cache_path, cache_data)
            
            return corners, 'google_vision_api'
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in Vision API detection: {e}")
            return np.array([]), 'google_vision_api_error'
    
    def _extract_corners_from_vision_api(self, response, image_path: str = None) -> np.ndarray:
        """
        Extract corner coordinates from Vision API response.
        
        Args:
            response: Vision API object localization response
            image_path: Path to the image for coordinate scaling
            
        Returns:
            Array of corner coordinates
        """
        try:
            corners = []
            
            # Look for document-like objects
            for obj in response.localized_object_annotations:
                if self.debug_mode:
                    print(f"Found object: {obj.name} (confidence: {obj.score:.3f})")
                
                if obj.name.lower() in ['document', 'paper', 'text', 'book', 'magazine']:
                    # Get bounding polygon
                    if hasattr(obj, 'bounding_poly') and obj.bounding_poly.normalized_vertices:
                        vertices = obj.bounding_poly.normalized_vertices
                        
                        # Convert normalized coordinates to pixel coordinates
                        if image_path:
                            original_dims = self._get_original_image_dimensions(image_path)
                            for vertex in vertices:
                                x = vertex.x * original_dims[0]
                                y = vertex.y * original_dims[1]
                                corners.append([x, y])
                        else:
                            # Use normalized coordinates directly if no image path
                            for vertex in vertices:
                                corners.append([vertex.x, vertex.y])
                        
                        if self.debug_mode:
                            print(f"Extracted {len(vertices)} vertices from {obj.name}")
                        
                        if len(corners) >= 4:
                            break
            
            # If no document objects found, try to find any rectangular objects
            if len(corners) < 4:
                if self.debug_mode:
                    print("No document objects found, looking for rectangular objects...")
                
                for obj in response.localized_object_annotations:
                    if hasattr(obj, 'bounding_poly') and obj.bounding_poly.normalized_vertices:
                        vertices = obj.bounding_poly.normalized_vertices
                        if len(vertices) >= 4:  # At least 4 vertices for a rectangle
                            if image_path:
                                original_dims = self._get_original_image_dimensions(image_path)
                                for vertex in vertices:
                                    x = vertex.x * original_dims[0]
                                    y = vertex.y * original_dims[1]
                                    corners.append([x, y])
                            else:
                                for vertex in vertices:
                                    corners.append([vertex.x, vertex.y])
                            
                            if self.debug_mode:
                                print(f"Using {obj.name} as fallback object")
                            break
            
            # Ensure we have exactly 4 corners
            if len(corners) > 4:
                # If more than 4 corners, try to simplify
                corners = self._simplify_corners(corners)
            
            # Scale coordinates to current image dimensions if we have image path
            if len(corners) == 4 and image_path:
                current_dims = self._get_original_image_dimensions(image_path)
                corners = self._scale_coordinates(np.array(corners), current_dims, current_dims)
            
            return np.array(corners).astype(np.float32)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting corners from Vision API response: {e}")
            return np.array([])
    
    def _extract_corners_from_web_detection(self, web_response) -> List[List[float]]:
        """
        Extract corners from web detection response.
        
        Args:
            web_response: Vision API web detection response
            
        Returns:
            List of corner coordinates
        """
        # This is a simplified implementation
        # In practice, you might need more sophisticated logic
        corners = []
        
        # Try to extract corners from web detection
        # This is a placeholder - actual implementation would depend on the response structure
        
        return corners
    
    def _simplify_corners(self, corners: List[List[float]]) -> List[List[float]]:
        """
        Simplify corners to get exactly 4 points.
        
        Args:
            corners: List of corner coordinates
            
        Returns:
            Simplified list with 4 corners
        """
        if len(corners) <= 4:
            return corners
        
        # Simple approach: take the first 4 corners
        # In practice, you might want more sophisticated corner selection
        return corners[:4]
    
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
    
    def calculate_perspective_transform(self, corners: np.ndarray, target_size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """
        Calculate perspective transform matrix.
        
        Args:
            corners: Ordered corner coordinates (4 corners)
            target_size: Target image size (width, height)
            
        Returns:
            Perspective transform matrix or None if invalid
        """
        if len(corners) != 4:
            return None
        
        # Ensure corners are in the right format
        if corners.ndim == 1:
            corners = corners.reshape(-1, 2)
        
        # Define target corners for top-down view
        target_corners = np.array([
            [0, 0],                    # Top-left
            [target_size[0], 0],       # Top-right
            [target_size[0], target_size[1]],  # Bottom-right
            [0, target_size[1]]        # Bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        try:
            matrix = cv2.getPerspectiveTransform(
                corners.astype(np.float32), target_corners
            )
            return matrix
        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating perspective transform: {e}")
            return None
    
    def apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray, 
                                 target_size: Tuple[int, int] = (800, 600)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply perspective transform to correct the image.
        
        Args:
            image: Input image
            corners: Corner coordinates
            target_size: Target image size
            
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
    
    def get_cache_info(self, image_path: str = None) -> Dict:
        """
        Get information about cache status and files.
        
        Args:
            image_path: Optional image path to check specific cache
            
        Returns:
            Dictionary with cache information
        """
        cache_info = {
            'cache_dir': self.cache_dir,
            'total_files': 0,
            'total_size_mb': 0,
            'files': []
        }
        
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            cache_info['total_files'] = len(cache_files)
            
            total_size = 0
            for filename in cache_files:
                filepath = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(filepath)
                total_size += file_size
                
                # Check if cache is valid
                is_valid = self._is_cache_valid(filepath)
                
                cache_info['files'].append({
                    'filename': filename,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'is_valid': is_valid,
                    'age_hours': round((time.time() - os.path.getmtime(filepath)) / 3600, 1)
                })
            
            cache_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return cache_info
    
    def clear_cache(self, image_path: str = None):
        """
        Clear cache files.
        
        Args:
            image_path: Optional image path to clear specific cache
        """
        if image_path:
            # Clear specific cache
            cache_key = self._get_cache_key(image_path)
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                if self.debug_mode:
                    print(f"Cleared cache for: {image_path}")
        else:
            # Clear all cache
            if os.path.exists(self.cache_dir):
                cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
                for filename in cache_files:
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
                
                if self.debug_mode:
                    print(f"Cleared {len(cache_files)} cache files")
    
    def get_debug_info(self, image_path: str = None) -> Dict:
        """
        Get information about debug files.
        
        Args:
            image_path: Optional image path to check specific debug directory
            
        Returns:
            Dictionary with debug information
        """
        if image_path:
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), '.debug')
        else:
            debug_dir = '.debug'
        
        debug_info = {
            'debug_dir': debug_dir,
            'total_files': 0,
            'total_size_mb': 0,
            'files': []
        }
        
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            debug_info['total_files'] = len(debug_files)
            
            total_size = 0
            for filename in debug_files:
                filepath = os.path.join(debug_dir, filename)
                file_size = os.path.getsize(filepath)
                total_size += file_size
                
                debug_info['files'].append({
                    'filename': filename,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
            
            debug_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return debug_info
    
    def clear_debug(self, image_path: str = None):
        """
        Clear debug visualization files.
        
        Args:
            image_path: Optional image path to clear specific debug directory
        """
        if image_path:
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), '.debug')
        else:
            debug_dir = '.debug'
        
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for filename in debug_files:
                filepath = os.path.join(debug_dir, filename)
                os.remove(filepath)
            
            if self.debug_mode:
                print(f"Cleared {len(debug_files)} debug files from {debug_dir}")
    
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
            
            # Detect corners using Google Vision API
            corners, method = self.detect_corners_google_ai(image_path)
            
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
                opencv_output_path = os.path.join(debug_dir, f"{base_filename}_google_ai_corners_opencv.jpg")
                self._save_opencv_visualization(image, corners, method, opencv_output_path)
                print(f"\nOpenCV visualization saved to: {opencv_output_path}")
                
                # Create and save matplotlib visualization
                fig = self.visualize_corners(image_rgb, corners, method)
                matplotlib_output_path = os.path.join(debug_dir, f"{base_filename}_google_ai_corner_detection.png")
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
            
            # Return results
            return {
                'success': True,
                'image_path': image_path,
                'image_dimensions': (image_width, image_height),
                'method': method,
                'corners': corners.tolist() if len(corners) > 0 else [],
                'perspective_matrix': perspective_matrix.tolist() if perspective_matrix is not None else None,
                'corrected_image': corrected_image is not None
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }

def main():
    """Main function to run the corner detector."""
    parser = argparse.ArgumentParser(
        description="Detect corners in note paper images using Google Cloud Vision API"
    )
    parser.add_argument("image_path", nargs='?', help="Path to the input image")
    parser.add_argument("--credentials", help="Path to OAuth credentials JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization and save")
    parser.add_argument("--cache-info", action="store_true", help="Show cache information")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache files")
    parser.add_argument("--debug-info", action="store_true", help="Show debug file information")
    parser.add_argument("--clear-debug", action="store_true", help="Clear debug files")
    
    args = parser.parse_args()
    
    # Handle cache and debug management commands
    if args.cache_info or args.clear_cache or args.debug_info or args.clear_debug:
        # Create detector instance without requiring credentials for cache operations
        detector = NoteCornerDetectorGoogleAI(credentials_path=None, debug_mode=args.debug)
        
        if args.cache_info:
            cache_info = detector.get_cache_info()
            print(f"\nCache Information:")
            print(f"  Cache directory: {cache_info['cache_dir']}")
            print(f"  Total files: {cache_info['total_files']}")
            print(f"  Total size: {cache_info['total_size_mb']} MB")
            if cache_info['files']:
                print(f"  Files:")
                for file_info in cache_info['files']:
                    status = "✓" if file_info['is_valid'] else "✗"
                    print(f"    {status} {file_info['filename']} ({file_info['size_mb']} MB, {file_info['age_hours']}h old)")
        
        if args.clear_cache:
            detector.clear_cache()
            print("Cache cleared successfully")
        
        if args.debug_info:
            debug_info = detector.get_debug_info()
            print(f"\nDebug Information:")
            print(f"  Debug directory: {debug_info['debug_dir']}")
            print(f"  Total files: {debug_info['total_files']}")
            print(f"  Total size: {debug_info['total_size_mb']} MB")
            if debug_info['files']:
                print(f"  Files:")
                for file_info in debug_info['files']:
                    print(f"    {file_info['filename']} ({file_info['size_mb']} MB)")
        
        if args.clear_debug:
            detector.clear_debug()
            print("Debug files cleared successfully")
        
        return
    
    # Require image path for actual processing
    if not args.image_path:
        print("Error: image_path is required for image processing")
        print("Use --cache-info, --clear-cache, --debug-info, or --clear-debug for cache management")
        return 1
    
    try:
        # Create detector
        detector = NoteCornerDetectorGoogleAI(
            credentials_path=args.credentials,
            debug_mode=args.debug
        )
        
        # Process image
        results = detector.process_image(
            args.image_path,
            save_visualization=not args.no_viz
        )
        
        if results['success']:
            print(f"\nProcessing completed successfully!")
            print(f"Detector type: {results['method']}")
            return 0
        else:
            print(f"Processing failed: {results['error']}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
