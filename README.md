# Note Paper Color Extractor

A Python script that processes images of squared note paper to extract two representative colors:
1. **Background Color** - The color of the note paper itself
2. **Grid Color** - The color of the grid lines/squares

## Features

- **Color Clustering**: Uses K-means clustering to identify dominant colors in the image
- **Smart Classification**: Automatically classifies colors as background or grid based on multiple criteria:
  - Color frequency (pixel count)
  - Saturation levels (paper tends to have lower saturation)
  - Brightness values (background is usually lighter)
  - Hue analysis (paper colors are typically in neutral ranges)
- **Grid Pattern Analysis**: Detects grid density to improve color classification
- **Visualization**: Creates a comprehensive visualization showing:
  - Original image
  - All extracted colors
  - Background color with RGB and hex values
  - Grid color with RGB and hex values
- **Output Formats**: Provides colors in RGB, hex, and pixel count formats

## Installation

1. **Clone or download** this repository
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python note_paper_color_extractor.py path/to/your/image.jpg
```

### Advanced Options

```bash
# Extract more initial colors for better analysis
python note_paper_color_extractor.py image.jpg --n-colors 12

# Skip visualization (faster processing)
python note_paper_color_extractor.py image.jpg --no-viz
```

### Command Line Arguments

- `image_path`: Path to the input image (required)
- `--n-colors`: Number of colors to extract initially (default: 8)
- `--no-viz`: Skip visualization and save (faster processing)

## How It Works

1. **Image Loading**: Loads and preprocesses the image, resizing if necessary
2. **Color Extraction**: Uses K-means clustering to identify dominant colors
3. **Grid Analysis**: Applies edge detection to identify grid patterns
4. **Color Classification**: Scores each color based on multiple criteria to determine if it's background or grid
5. **Result Output**: Provides the two representative colors with detailed information
6. **Visualization**: Creates and saves a comprehensive analysis visualization

## Example Output

```
Processing image: note_paper.jpg
Image loaded: 800x600 pixels
Extracted 8 dominant colors
Grid density: 0.0234

==================================================
EXTRACTION RESULTS
==================================================
Background Color (Paper): RGB(248, 245, 240)
  - Count: 156432
  - Hex: #f8f5f0

Grid Color (Lines): RGB(180, 175, 170)
  - Count: 23456
  - Hex: #b4afaa
==================================================

Visualization saved to: note_paper_color_analysis.png
Processing completed successfully!
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- And other formats supported by OpenCV

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Pillow

## Tips for Best Results

1. **Image Quality**: Use clear, well-lit images of note paper
2. **Grid Visibility**: Ensure the grid lines are clearly visible
3. **Color Count**: For complex images, increase `--n-colors` to 12-16
4. **Image Size**: The script automatically resizes large images for processing

## Troubleshooting

- **"Image not found"**: Check the file path and ensure the image exists
- **Poor color extraction**: Try increasing the `--n-colors` parameter
- **Memory issues**: The script automatically resizes large images
- **Installation errors**: Ensure you have Python 3.7+ and pip installed

## Note Corner Detector

The project includes two specialized corner detection scripts, each optimized for different approaches:

### 1. OpenCV-Based Corner Detector (`note_corner_detector_opencv.py`)

Uses analytical computer vision techniques with OpenCV for robust corner detection without external dependencies.

**Features:**
- **Harris Corner Detection**: Uses OpenCV's `goodFeaturesToTrack` for robust corner detection
- **Contour Analysis**: Finds document boundaries and approximates to polygons
- **Edge-Based Detection**: Uses Canny edge detection and Hough Line Transform
- **Hybrid Methods**: Combines multiple techniques for optimal results
- **Local Processing**: No external API dependencies, works offline

**Usage:**
```bash
# Basic usage
python note_corner_detector_opencv.py image.jpg

# With debug output
python note_corner_detector_opencv.py image.jpg --debug

# Skip visualization
python note_corner_detector_opencv.py image.jpg --no-viz
```

### 2. Google Cloud AI Corner Detector (`note_corner_detector_google_ai.py`)

Leverages Google Cloud AI Vision API for advanced document analysis and corner detection through cloud-based AI services.

**Features:**
- **Document Text Detection**: Identifies document boundaries from text annotations
- **Object Localization**: Detects document-like objects in images
- **Web Detection**: Additional context from web-based image analysis
- **Cloud Processing**: Offloads computation to Google's AI infrastructure
- **Advanced AI**: Uses Google's state-of-the-art computer vision models

**Usage:**
```bash
# Basic usage (requires API key)
python note_corner_detector_google_ai.py image.jpg --api-key YOUR_API_KEY

# With custom endpoint
python note_corner_detector_google_ai.py image.jpg --api-key YOUR_API_KEY --api-endpoint CUSTOM_URL

# With debug output
python note_corner_detector_google_ai.py image.jpg --api-key YOUR_API_KEY --debug

# Skip visualization
python note_corner_detector_google_ai.py image.jpg --api-key YOUR_API_KEY --no-viz
```



### Common Features Across Both Corner Detector Scripts

- Automatic corner detection and ordering
- Perspective transform calculation
- Visualization of detected corners
- Support for various image formats
- Debug mode for detailed output

## License

This project is open source and available under the MIT License.
