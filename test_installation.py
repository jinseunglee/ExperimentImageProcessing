#!/usr/bin/env python3
"""
Test script to verify the installation of all required packages
and ensure the NotePaperColorExtractor can be imported correctly.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow version: {Image.__version__}")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    return True


def test_color_extractor():
    """Test if the NotePaperColorExtractor can be imported."""
    print("\nTesting NotePaperColorExtractor import...")
    
    try:
        from note_paper_color_extractor import NotePaperColorExtractor
        print("✓ NotePaperColorExtractor imported successfully")
        
        # Test creating an instance
        extractor = NotePaperColorExtractor()
        print("✓ NotePaperColorExtractor instance created successfully")
        
        return True
    except ImportError as e:
        print(f"✗ NotePaperColorExtractor import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error creating NotePaperColorExtractor instance: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("INSTALLATION TEST")
    print("="*60)
    
    # Test package imports
    packages_ok = test_imports()
    
    if packages_ok:
        # Test color extractor
        extractor_ok = test_color_extractor()
        
        if extractor_ok:
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED!")
            print("✓ Installation is complete and working correctly.")
            print("="*60)
            print("\nYou can now use the script:")
            print("  python note_paper_color_extractor.py your_image.jpg")
            print("\nOr run the example:")
            print("  python example_usage.py")
        else:
            print("\n" + "="*60)
            print("✗ COLOR EXTRACTOR TEST FAILED")
            print("✗ Please check the note_paper_color_extractor.py file.")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ PACKAGE IMPORT TESTS FAILED")
        print("✗ Please install missing packages:")
        print("  pip install -r requirements.txt")
        print("="*60)


if __name__ == "__main__":
    main()
