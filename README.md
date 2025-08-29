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

### 3. DocAligner Corner Detector (`note_corner_detector_docaligner.py`)

Uses the DocAligner library for advanced document corner detection and perspective transformation to a top-down view.

**Features:**
- **OAuth 2.0 Authentication**: Secure authentication using service account credentials
- **Document Text Detection**: Identifies document boundaries from text annotations
- **Object Localization**: Detects document-like objects in images
- **Web Detection**: Additional context from web-based image analysis
- **Cloud Processing**: Offloads computation to Google's AI infrastructure
- **Advanced AI**: Uses Google's state-of-the-art computer vision models
- **Response Caching**: Persistent storage of API responses to avoid redundant calls
- **Smart Cache Management**: Automatic cache expiration and management

**DocAligner Features:**
- **Advanced Document Detection**: Uses DocAligner's specialized document boundary detection
- **Precise Corner Detection**: Accurate corner identification for document alignment
- **Perspective Transformation**: Automatic transformation to top-down view
- **Corner Ordering**: Intelligent corner ordering for consistent transforms
- **Local Processing**: No external API dependencies, works offline
- **High-Quality Output**: Professional-grade document correction

**Usage:**
```bash
# Basic usage with OAuth 2.0 credentials file
python note_corner_detector_google_ai.py image.jpg --credentials oauth_credentials.json

# With custom endpoint
python note_corner_detector_google_ai.py image.jpg --credentials oauth_credentials.json --api-endpoint CUSTOM_URL

# With debug output
python note_corner_detector_google_ai.py image.jpg --credentials oauth_credentials.json --debug

# Skip visualization
python note_corner_detector_google_ai.py image.jpg --credentials oauth_credentials.json --no-viz

# Use default application credentials (if configured)
python note_corner_detector_google_ai.py image.jpg

# Cache management options
python note_corner_detector_google_ai.py --cache-info                    # Show cache information
python note_corner_detector_google_ai.py --clear-cache                   # Clear all cache
python note_corner_detector_google_ai.py --clear-cache-method document_text  # Clear specific method cache

# Debug file management options
python note_corner_detector_google_ai.py --debug-info                    # Show debug file information
python note_corner_detector_google_ai.py --clear-debug                   # Clear all debug files

# Custom cache directory
python note_corner_detector_google_ai.py image.jpg --cache-dir /path/to/cache

# DocAligner Corner Detector Usage
python note_corner_detector_docaligner.py image.jpg                    # Basic usage
python note_corner_detector_docaligner.py image.jpg --debug            # With debug output
python note_corner_detector_docaligner.py image.jpg --no-viz           # Skip visualization
```

#### Caching System

The Google AI corner detector includes a sophisticated caching system that stores API responses to avoid redundant calls:

**How It Works:**
1. **Image Hashing**: Each image is hashed using SHA-256 to create a unique identifier
2. **Response Storage**: API responses are stored as pickle files in the cache directory
3. **Cache Lookup**: Before making an API call, the detector checks if a cached response exists
4. **Automatic Expiration**: Cache entries expire after 7 days to ensure freshness
5. **Method-Specific Caching**: Each detection method (document_text, object_localization, web_detection) has separate cache entries

**Benefits:**
- **Cost Savings**: Avoids unnecessary API calls for previously processed images
- **Speed Improvement**: Cached responses are retrieved instantly
- **Offline Processing**: Can process cached images without internet connection
- **Batch Processing**: Efficient for processing multiple images or repeated analysis

**Cache Management:**
- **Automatic Setup**: Cache directory is created automatically in `.cache/` by default
- **Size Monitoring**: Track cache size and file count with `--cache-info`
- **Selective Clearing**: Clear cache for specific methods or all methods
- **Custom Location**: Specify custom cache directory with `--cache-dir`

**Example Cache Structure:**
```
.cache/
├── a1b2c3d4_document_text.pkl      # Document text detection response
├── a1b2c3d4_object_localization.pkl # Object localization response
└── a1b2c3d4_web_detection.pkl       # Web detection response
```

#### Debug Directory System

The detector automatically creates and manages debug directories for visualization files:

**How It Works:**
1. **Automatic Creation**: Debug directories are created automatically as `.debug/` in the image's location
2. **Organized Storage**: Visualization files are saved in organized debug directories
3. **Location-Based**: Each image gets its own debug directory in its parent folder
4. **File Management**: Easy cleanup and monitoring of debug files

**Debug File Types:**
- **OpenCV Visualization**: `{image_name}_google_ai_corners_opencv.jpg` - Direct image with corners drawn
- **Matplotlib Visualization**: `{image_name}_google_ai_corner_detection.png` - High-quality visualization with plots

**Debug Management:**
- **Information**: Use `--debug-info` to see debug file statistics
- **Cleanup**: Use `--clear-debug` to remove all debug files
- **Organization**: Debug files are automatically organized by image location

**Example Debug Structure:**
```
project/
├── .debug/                                    # Main debug directory
│   ├── image1_google_ai_corners_opencv.jpg
│   └── image1_google_ai_corner_detection.png
├── images/
│   ├── .debug/                               # Image-specific debug directory
│   │   ├── image2_google_ai_corners_opencv.jpg
│   │   └── image2_google_ai_corner_detection.png
│   └── image2.jpg
└── image1.jpg
```

**Testing Caching:**
A test script `test_caching.py` is included to demonstrate the caching functionality:

```bash
# Run the caching test
python test_caching.py

# This will:
# 1. Process an image for the first time (API calls)
# 2. Process the same image again (cached responses)
# 3. Compare processing times and results
# 4. Show cache statistics and management functions
```

**Cache Performance:**
- **First Run**: Makes API calls and stores responses (slower, costs API credits)
- **Subsequent Runs**: Uses cached responses (instant, no API costs)
- **Typical Speed Improvement**: 90%+ faster for cached images
- **Storage Efficiency**: Compressed pickle format, minimal disk usage

#### DocAligner Installation and Testing

**Installation:**
```bash
# Install DocAligner from PyPI
pip install docaligner-docsaid

# Or install from source
git clone https://github.com/DocsaidLab/DocAligner.git
cd DocAligner
pip install -e .
```

**Testing DocAligner:**
A test script `test_docaligner.py` is included to verify the implementation:

```bash
# Run the DocAligner test
python test_docaligner.py

# Test with specific image
python test_docaligner.py path/to/your/image.jpg

# This will:
# 1. Test DocAligner import and initialization
# 2. Test corner detector creation
# 3. Test image processing and corner detection
# 4. Test debug directory creation
# 5. Show comprehensive results and statistics
```

**DocAligner Performance:**
- **Local Processing**: No external API calls, works offline
- **Fast Detection**: Optimized document boundary detection
- **High Accuracy**: Specialized for document corner detection
- **Professional Quality**: Industry-standard document alignment

### Common Features Across All Corner Detector Scripts

- Automatic corner detection and ordering
- Perspective transform calculation
- Visualization of detected corners
- Support for various image formats
- Debug mode for detailed output

## License

This project is open source and available under the MIT License.