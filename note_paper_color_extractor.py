#!/usr/bin/env python3
"""
Note Paper Color Extractor

This script processes images of squared note paper to extract two representative colors:
1. The background color of the note paper
2. The color of the grid squares/lines

The script uses color clustering and analysis to identify the dominant colors.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import os


class NotePaperColorExtractor:
    def __init__(self, n_colors=8, grid_threshold=0.1):
        """
        Initialize the color extractor.
        
        Args:
            n_colors (int): Number of colors to extract initially
            grid_threshold (float): Threshold for identifying grid lines vs background
        """
        self.n_colors = n_colors
        self.grid_threshold = grid_threshold
        
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
        if width > 1000 or height > 1000:
            scale = min(1000/width, 1000/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
        return image_rgb
    
    def extract_colors(self, image):
        """Extract dominant colors using K-means clustering."""
        # Reshape image to 2D array of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and labels
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Count occurrences of each color
        color_counts = Counter(labels)
        
        return colors, color_counts
    
    def analyze_grid_pattern(self, image):
        """Analyze the image to identify grid patterns."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to connect grid lines
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Calculate grid density
        grid_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return grid_density, edges
    
    def classify_colors(self, image, colors, color_counts, grid_density):
        """Classify colors as background or grid based on various criteria."""
        # Convert colors to HSV for better analysis
        colors_hsv = []
        for color in colors:
            color_rgb = np.uint8([[color]])
            color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
            colors_hsv.append(color_hsv[0][0])
        
        colors_hsv = np.array(colors_hsv)
        
        # Calculate color scores based on multiple criteria
        background_scores = []
        grid_scores = []
        
        print(f"\nAnalyzing color distribution and grid patterns...")
        
        for i in range(len(colors)):
            color = colors[i]
            count = color_counts[i]
            
            try:
                # Calculate distribution score
                distribution_score = self.calculate_color_distribution(image, color, tolerance=60)
                
                # Calculate grid pattern score
                grid_pattern_score = self.detect_grid_pattern(image, color, tolerance=60)
                
                print(f"  Color {i+1}: RGB{tuple(color)}, Count={count}, Dist={distribution_score:.4f}, Grid={grid_pattern_score:.4f}")
                
            except Exception as e:
                print(f"  Color {i+1}: Error in analysis: {e}")
                distribution_score = 0.0
                grid_pattern_score = 0.0
            
            # Background scoring (prioritizes count and distribution)
            bg_score = 0
            normalized_count = count / max(color_counts.values())
            bg_score += normalized_count * 0.5  # Higher count weight for background
            bg_score += distribution_score * 0.3  # Distribution important for background
            
            # Lower saturation = more likely to be background (paper)
            saturation = colors_hsv[i][1]
            bg_score += (255 - saturation) / 255.0 * 0.15
            
            # Higher value = more likely to be background (lighter colors)
            value = colors_hsv[i][2]
            bg_score += value / 255.0 * 0.05
            
            background_scores.append((bg_score, i, color, count, distribution_score))
            
            # Grid scoring (prioritizes distribution and grid pattern formation)
            grid_score = 0
            grid_score += distribution_score * 0.4  # Must be distributed across image
            grid_score += grid_pattern_score * 0.5  # Must form grid patterns
            grid_score += (1.0 - normalized_count) * 0.1  # Grid typically has fewer pixels than background
            
            grid_scores.append((grid_score, i, color, count, distribution_score, grid_pattern_score))
        
        # Sort by scores
        background_scores.sort(reverse=True)
        grid_scores.sort(reverse=True)
        
        # Select best background color
        background_color = colors[background_scores[0][1]]
        background_count = background_scores[0][3]
        background_distribution = background_scores[0][4]
        
        # Select best grid color (iterate through candidates to find best match)
        print(f"\n  Grid color candidates (top 5):")
        grid_color = None
        grid_count = 0
        grid_distribution = 0.0
        grid_pattern_score = 0.0
        
        for j, (score, idx, color, count, dist, pattern) in enumerate(grid_scores[:5]):
            print(f"    Candidate {j+1}: RGB{tuple(color)}, Score={score:.4f} (Dist={dist:.4f}, Pattern={pattern:.4f})")
            
            # Select the first candidate with decent grid pattern score
            if grid_color is None and pattern > 0.05:  # Minimum grid pattern threshold
                grid_color = color
                grid_count = count
                grid_distribution = dist
                grid_pattern_score = pattern
                print(f"    → Selected as grid color")
        
        # Fallback: if no good grid pattern found, use second best background candidate
        if grid_color is None:
            print(f"    → No strong grid pattern found, using second best background candidate")
            grid_color = colors[background_scores[1][1]]
            grid_count = background_scores[1][3]
            grid_distribution = background_scores[1][4]
            grid_pattern_score = 0.0
        
        return background_color, grid_color, background_count, grid_count, background_distribution, grid_distribution
    
    def calculate_color_distribution(self, image, target_color, tolerance=30):
        """
        Calculate how evenly distributed a color is across the image.
        
        Args:
            image: Input image
            target_color: RGB color to analyze
            tolerance: Color distance tolerance
            
        Returns:
            Distribution score (higher = more evenly distributed)
        """
        # Convert target_color to numpy array and ensure proper type
        target_color = np.array(target_color, dtype=np.uint8)
        
        # Create color filter mask
        mask = self.create_color_filter(image, target_color, tolerance)
        
        if not np.any(mask):
            return 0.0
        
        # Divide image into grid sections
        height, width = image.shape[:2]
        grid_size = 8  # 8x8 grid
        
        section_height = height // grid_size
        section_width = width // grid_size
        
        distribution_scores = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Define section boundaries
                y_start = i * section_height
                y_end = min((i + 1) * section_height, height)
                x_start = j * section_width
                x_end = min((j + 1) * section_width, width)
                
                # Calculate color density in this section
                section_mask = mask[y_start:y_end, x_start:x_end]
                section_density = np.sum(section_mask) / section_mask.size
                
                distribution_scores.append(section_density)
        
        # Calculate distribution uniformity (lower variance = more uniform)
        mean_density = np.mean(distribution_scores)
        variance = np.var(distribution_scores)
        
        # Normalize variance to 0-1 scale (0 = perfectly uniform, 1 = very uneven)
        max_variance = mean_density * (1 - mean_density)  # Theoretical maximum variance
        if max_variance > 0:
            normalized_variance = variance / max_variance
        else:
            normalized_variance = 0
        
        # Distribution score (higher = more evenly distributed)
        distribution_score = 1.0 - normalized_variance
        
        return distribution_score
    
    def detect_grid_pattern(self, image, target_color, tolerance=30):
        """
        Detect how well a color forms grid/square patterns.
        
        Args:
            image: Input image
            target_color: RGB color to analyze
            tolerance: Color distance tolerance
            
        Returns:
            Grid pattern score (higher = more grid-like)
        """
        # Convert target_color to numpy array
        target_color = np.array(target_color, dtype=np.uint8)
        
        # Create color filter mask
        mask = self.create_color_filter(image, target_color, tolerance)
        
        if not np.any(mask):
            return 0.0
        
        # Convert mask to uint8 for OpenCV operations
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((2, 2), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using Hough Line Transform
        edges = cv2.Canny(mask_cleaned, 50, 150)
        
        # Detect horizontal and vertical lines with adaptive threshold
        edge_pixels = np.sum(edges > 0)
        threshold = max(10, min(30, int(edge_pixels * 0.05)))  # More sensitive threshold
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)
        
        # Debug info (remove after testing)
        if False:  # Set to True for debugging
            print(f"      Debug: Edge pixels={edge_pixels}, Threshold={threshold}, Lines found={len(lines) if lines is not None else 0}")
        
        if lines is None:
            return 0.0
        
        # Classify lines as horizontal or vertical
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
        
        # Calculate grid score based on presence of both horizontal and vertical lines
        h_count = len(horizontal_lines)
        v_count = len(vertical_lines)
        
        if h_count == 0 or v_count == 0:
            return 0.0
        
        # Calculate regularity of line spacing
        def calculate_line_regularity(lines_list):
            if len(lines_list) < 2:
                return 0.0
            
            rhos = [line[0] for line in lines_list]
            rhos.sort()
            
            # Calculate spacing between consecutive lines
            spacings = [rhos[i+1] - rhos[i] for i in range(len(rhos)-1)]
            
            if len(spacings) == 0:
                return 0.0
            
            # Calculate coefficient of variation (lower = more regular)
            mean_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            
            if mean_spacing == 0:
                return 0.0
            
            cv = std_spacing / mean_spacing
            
            # Convert to regularity score (higher = more regular)
            regularity = max(0, 1.0 - cv)
            return regularity
        
        h_regularity = calculate_line_regularity(horizontal_lines)
        v_regularity = calculate_line_regularity(vertical_lines)
        
        # Combine factors into grid score
        line_balance = min(h_count, v_count) / max(h_count, v_count)  # Balance between h and v lines
        avg_regularity = (h_regularity + v_regularity) / 2
        line_density = min(1.0, (h_count + v_count) / 20)  # Normalize line count
        
        grid_score = line_balance * 0.4 + avg_regularity * 0.4 + line_density * 0.2
        
        # Alternative approach: Frequency domain analysis for grid patterns
        if grid_score < 0.1:  # If Hough line detection failed, try frequency analysis
            grid_score = self.detect_grid_by_frequency(mask_uint8)
        
        return grid_score
    
    def detect_grid_by_frequency(self, mask):
        """
        Detect grid patterns using frequency domain analysis.
        
        Args:
            mask: Binary mask of the color
            
        Returns:
            Grid score based on frequency analysis
        """
        if mask.size == 0:
            return 0.0
        
        # Apply FFT to detect periodic patterns
        f_transform = np.fft.fft2(mask)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Look for peaks in the frequency domain that indicate regular patterns
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # Sample frequency components that would indicate grid patterns
        # Check for horizontal frequency components
        h_region = magnitude_spectrum[center_y-2:center_y+3, :]
        v_region = magnitude_spectrum[:, center_x-2:center_x+3]
        
        h_max = np.max(h_region)
        v_max = np.max(v_region)
        overall_max = np.max(magnitude_spectrum)
        
        if overall_max == 0:
            return 0.0
        
        # Calculate grid score based on frequency peaks
        h_score = h_max / overall_max
        v_score = v_max / overall_max
        
        # Look for regularity in the frequency domain
        h_variance = np.var(h_region) / (np.mean(h_region) + 1e-6)
        v_variance = np.var(v_region) / (np.mean(v_region) + 1e-6)
        
        regularity_score = 1.0 / (1.0 + h_variance + v_variance)
        
        frequency_grid_score = (h_score + v_score) * regularity_score * 0.5
        
        return min(1.0, frequency_grid_score)
    
    def create_color_filter(self, image, target_color, tolerance=30):
        """
        Create a binary mask for pixels similar to the target color.
        
        Args:
            image: Input image
            target_color: RGB color to filter for
            tolerance: Color distance tolerance
            
        Returns:
            Binary mask where True indicates pixels similar to target color
        """
        # Ensure target_color is a numpy array with proper shape
        target_color = np.array(target_color, dtype=np.uint8)
        
        # Calculate color distance for each pixel
        color_diff = np.sqrt(np.sum((image - target_color) ** 2, axis=2))
        
        # Create binary mask
        mask = color_diff <= tolerance
        
        return mask
    
    def apply_color_filter_bw(self, image, target_color, tolerance=30, invert=False):
        """
        Apply color filter and convert to black and white image.
        
        Args:
            image: Input image
            target_color: RGB color to filter for
            tolerance: Color distance tolerance
            invert: If True, invert the black/white mapping
            
        Returns:
            Black and white image where white shows filtered pixels
        """
        # Create color filter mask
        mask = self.create_color_filter(image, target_color, tolerance)
        
        # Create black and white image
        bw_image = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if invert:
            bw_image[~mask] = 255  # Non-matching pixels become white
        else:
            bw_image[mask] = 255   # Matching pixels become white
            
        return bw_image
    
    def create_colored_overlay(self, image, target_color, overlay_color, tolerance=30, opacity=0.7):
        """
        Create a colored overlay highlighting the target color.
        
        Args:
            image: Input image
            target_color: RGB color to filter for
            overlay_color: RGB color for the overlay
            tolerance: Color distance tolerance
            opacity: Opacity of the overlay (0.0 to 1.0)
            
        Returns:
            Image with colored overlay
        """
        # Create color filter mask
        mask = self.create_color_filter(image, target_color, tolerance)
        
        # Create overlay image
        overlay = image.copy().astype(np.float32)
        
        # Apply colored overlay where mask is True
        overlay[mask] = overlay[mask] * (1 - opacity) + np.array(overlay_color) * opacity
        
        return overlay.astype(np.uint8)
    
    def create_isolation_view(self, image, target_color, tolerance=30):
        """
        Create an isolation view showing only the target color in its original color
        and everything else in a neutral gray.
        
        Args:
            image: Input image
            target_color: RGB color to filter for
            tolerance: Color distance tolerance
            
        Returns:
            Image with target color isolated
        """
        # Create color filter mask
        mask = self.create_color_filter(image, target_color, tolerance)
        
        # Create grayscale version of the image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Create result image starting with grayscale
        result = gray_rgb.copy()
        
        # Keep original colors where mask is True
        result[mask] = image[mask]
        
        return result
    
    def visualize_results(self, image, background_color, grid_color, colors, color_counts):
        """Create a comprehensive visualization of the extracted colors and filters."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Row 1: Original analysis
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # All extracted colors
        color_palette = np.array(colors).reshape(1, -1, 3)
        axes[0, 1].imshow(color_palette)
        axes[0, 1].set_title(f'All Extracted Colors ({len(colors)})')
        axes[0, 1].axis('off')
        
        # Color distribution pie chart
        color_labels = [f'Color {i+1}' for i in range(len(colors))]
        color_values = [color_counts[i] for i in range(len(colors))]
        color_rgb_norm = colors / 255.0  # Normalize for matplotlib
        
        axes[0, 2].pie(color_values, labels=color_labels, colors=color_rgb_norm, autopct='%1.1f%%')
        axes[0, 2].set_title('Color Distribution')
        
        # Row 2: Background analysis
        # Background color
        bg_palette = np.array([background_color]).reshape(1, 1, 3)
        axes[1, 0].imshow(bg_palette)
        axes[1, 0].set_title(f'Background Color\nRGB: {tuple(background_color)}\nHex: #{background_color[0]:02x}{background_color[1]:02x}{background_color[2]:02x}')
        axes[1, 0].axis('off')
        
        # Background highlighted with blue overlay
        bg_overlay = self.create_colored_overlay(image, background_color, [0, 150, 255], tolerance=40, opacity=0.5)
        axes[1, 1].imshow(bg_overlay)
        axes[1, 1].set_title('Background Highlighted\n(Blue Overlay)')
        axes[1, 1].axis('off')
        
        # Background isolated view (background in original color, rest in gray)
        bg_isolated = self.create_isolation_view(image, background_color, tolerance=40)
        axes[1, 2].imshow(bg_isolated)
        axes[1, 2].set_title('Background Isolated\n(Original Color + Gray)')
        axes[1, 2].axis('off')
        
        # Row 3: Grid analysis
        # Grid color
        grid_palette = np.array([grid_color]).reshape(1, 1, 3)
        axes[2, 0].imshow(grid_palette)
        axes[2, 0].set_title(f'Grid Color\nRGB: {tuple(grid_color)}\nHex: #{grid_color[0]:02x}{grid_color[1]:02x}{grid_color[2]:02x}')
        axes[2, 0].axis('off')
        
        # Grid highlighted with red overlay
        grid_overlay = self.create_colored_overlay(image, grid_color, [255, 50, 50], tolerance=40, opacity=0.6)
        axes[2, 1].imshow(grid_overlay)
        axes[2, 1].set_title('Grid Highlighted\n(Red Overlay)')
        axes[2, 1].axis('off')
        
        # Grid isolated view (grid in original color, rest in gray)
        grid_isolated = self.create_isolation_view(image, grid_color, tolerance=40)
        axes[2, 2].imshow(grid_isolated)
        axes[2, 2].set_title('Grid Isolated\n(Original Color + Gray)')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def process_image(self, image_path, save_visualization=True):
        """Main method to process an image and extract colors."""
        print(f"Processing image: {image_path}")
        
        # Load and preprocess image
        image = self.load_image(image_path)
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Extract colors
        colors, color_counts = self.extract_colors(image)
        print(f"Extracted {len(colors)} dominant colors")
        
        # Analyze grid pattern
        grid_density, edges = self.analyze_grid_pattern(image)
        print(f"Grid density: {grid_density:.4f}")
        
        # Classify colors
        background_color, grid_color, bg_count, grid_count, bg_distribution, grid_distribution = self.classify_colors(
            image, colors, color_counts, grid_density
        )
        
        # Calculate grid pattern score for the selected grid color (for display)
        grid_pattern_score = self.detect_grid_pattern(image, grid_color, tolerance=60)
        
        # Print results
        print("\n" + "="*50)
        print("EXTRACTION RESULTS")
        print("="*50)
        print(f"Background Color (Paper): RGB{tuple(background_color)}")
        print(f"  - Count: {bg_count}")
        print(f"  - Distribution Score: {bg_distribution:.4f}")
        print(f"  - Hex: #{background_color[0]:02x}{background_color[1]:02x}{background_color[2]:02x}")
        print(f"\nGrid Color (Lines): RGB{tuple(grid_color)}")
        print(f"  - Count: {grid_count}")
        print(f"  - Distribution Score: {grid_distribution:.4f}")
        print(f"  - Grid Pattern Score: {grid_pattern_score:.4f}")
        print(f"  - Hex: #{grid_color[0]:02x}{grid_color[1]:02x}{grid_color[2]:02x}")
        print("="*50)
        
        # Create visualization
        if save_visualization:
            fig = self.visualize_results(image, background_color, grid_color, colors, color_counts)
            
            # Save visualization
            output_path = image_path.rsplit('.', 1)[0] + '_color_analysis.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
            
            # Show plot
            plt.show()
        
        return {
            'background_color': background_color,
            'grid_color': grid_color,
            'background_count': bg_count,
            'grid_count': grid_count,
            'background_distribution': bg_distribution,
            'grid_distribution': grid_distribution,
            'all_colors': colors,
            'color_counts': color_counts,
            'grid_density': grid_density
        }


def main():
    """Main function to run the color extractor."""
    parser = argparse.ArgumentParser(
        description='Extract representative colors from squared note paper images'
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--n-colors', type=int, default=8, 
                       help='Number of colors to extract initially (default: 8)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization and save')
    
    args = parser.parse_args()
    
    try:
        # Create extractor
        extractor = NotePaperColorExtractor(n_colors=args.n_colors)
        
        # Process image
        results = extractor.process_image(
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
