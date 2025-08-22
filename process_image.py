#!/usr/bin/env python3
"""
Example usage of the NotePaperColorExtractor class

This script demonstrates how to use the color extractor programmatically
instead of from the command line.
"""

from note_paper_color_extractor import NotePaperColorExtractor
import os


def main():
    """Example of programmatic usage."""
    
    # Example image path (replace with your actual image)
    image_path = "example_note_paper.jpg"
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"Example image '{image_path}' not found.")
        print("Please place a note paper image in the current directory or update the path.")
        print("You can also use the command line version:")
        print(f"python note_paper_color_extractor.py {image_path}")
        return
    
    print("="*60)
    print("NOTE PAPER COLOR EXTRACTOR - PROGRAMMATIC USAGE")
    print("="*60)
    
    # Create extractor with custom parameters
    extractor = NotePaperColorExtractor(
        n_colors=10,  # Extract 10 initial colors
        grid_threshold=0.1
    )
    
    try:
        # Process the image
        results = extractor.process_image(
            image_path, 
            save_visualization=True  # Save and show visualization
        )
        
        # Access the results programmatically
        print("\n" + "="*60)
        print("PROGRAMMATIC ACCESS TO RESULTS")
        print("="*60)
        
        background_rgb = results['background_color']
        grid_rgb = results['grid_color']
        
        print(f"Background color (RGB): {background_rgb}")
        print(f"Grid color (RGB): {grid_rgb}")
        
        # Convert to hex
        bg_hex = f"#{background_rgb[0]:02x}{background_rgb[1]:02x}{background_rgb[2]:02x}"
        grid_hex = f"#{grid_rgb[0]:02x}{grid_rgb[1]:02x}{grid_rgb[2]:02x}"
        
        print(f"Background color (Hex): {bg_hex}")
        print(f"Grid color (Hex): {grid_hex}")
        
        # Access other information
        print(f"Total colors extracted: {len(results['all_colors'])}")
        print(f"Grid density: {results['grid_density']:.4f}")
        
        # Example: Use the colors for further processing
        print("\n" + "="*60)
        print("EXAMPLE: COLOR ANALYSIS")
        print("="*60)
        
        # Calculate color contrast
        bg_luminance = 0.299 * background_rgb[0] + 0.587 * background_rgb[1] + 0.114 * background_rgb[2]
        grid_luminance = 0.299 * grid_rgb[0] + 0.587 * grid_rgb[1] + 0.114 * grid_rgb[2]
        
        contrast_ratio = max(bg_luminance, grid_luminance) / min(bg_luminance, grid_luminance)
        
        print(f"Background luminance: {bg_luminance:.1f}")
        print(f"Grid luminance: {grid_luminance:.1f}")
        print(f"Contrast ratio: {contrast_ratio:.2f}")
        
        if contrast_ratio > 3.0:
            print("✓ Good contrast for readability")
        elif contrast_ratio > 2.0:
            print("⚠ Moderate contrast")
        else:
            print("✗ Low contrast - may affect readability")
            
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
