import os
import sys
import numpy as np
import cv2
import logging
import uuid
import dlib  # This is what face_recognition uses internally

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dlib_compatible_image(output_dir="temp_images"):
    """Create a very simple image that is 100% compatible with dlib face detection"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a very simple grayscale image (more compatible with dlib)
    img = np.zeros((400, 400), dtype=np.uint8)
    img.fill(200)  # Light gray background
    
    # Draw a simple face-like circle
    cv2.circle(img, (200, 200), 100, 50, -1)  # Dark circle
    cv2.circle(img, (150, 150), 20, 255, -1)  # Left eye
    cv2.circle(img, (250, 150), 20, 255, -1)  # Right eye
    
    # Draw a simple mouth
    cv2.line(img, (150, 250), (250, 250), 50, 10)
    
    # Save as grayscale (even more compatible)
    output_path = os.path.join(output_dir, f"dlib_test_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(output_path, img)
    logger.info(f"Created dlib test image at: {output_path}")
    
    # Also create RGB version
    img_rgb = np.zeros((400, 400, 3), dtype=np.uint8)
    img_rgb[:,:,0] = img  # Copy to all channels
    img_rgb[:,:,1] = img
    img_rgb[:,:,2] = img
    
    output_path_rgb = os.path.join(output_dir, f"dlib_test_rgb_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(output_path_rgb, img_rgb)
    logger.info(f"Created dlib RGB test image at: {output_path_rgb}")
    
    return output_path, output_path_rgb

def test_with_dlib(image_path):
    """Test image with dlib face detector directly"""
    try:
        # Load the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        logger.info(f"Loaded grayscale image: shape={img.shape}, dtype={img.dtype}")
        
        # Initialize dlib's face detector
        detector = dlib.get_frontal_face_detector()
        
        # Detect faces
        dets = detector(img, 1)
        logger.info(f"Detected {len(dets)} faces with dlib")
        
        return True
    except Exception as e:
        logger.error(f"Dlib test failed: {e}")
        return False

def test_with_cv2(image_path):
    """Test image with OpenCV's face detector"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        
        # Load the face cascade
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        logger.info(f"Detected {len(faces)} faces with OpenCV")
        
        return True
    except Exception as e:
        logger.error(f"OpenCV test failed: {e}")
        return False

def main():
    # Create test images
    gray_path, rgb_path = create_dlib_compatible_image()
    
    # Test with dlib directly
    logger.info("Testing with dlib directly (what face_recognition uses internally)")
    test_with_dlib(gray_path)
    test_with_dlib(rgb_path)
    
    # Test with OpenCV
    logger.info("Testing with OpenCV (alternative to face_recognition)")
    test_with_cv2(gray_path)
    test_with_cv2(rgb_path)
    
    # Copy the files to standard locations
    standard_gray = os.path.join("temp_images", "dlib_test.jpg")
    standard_rgb = os.path.join("temp_images", "dlib_test_rgb.jpg")
    
    cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(standard_gray, cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(standard_rgb, cv2.imread(rgb_path))
    
    logger.info(f"Copied to standard locations: {standard_gray} and {standard_rgb}")
    
    # Print instructions
    print("\n==== MANUAL TESTING INSTRUCTIONS ====")
    print("1. Use the test images created in temp_images/ folder")
    print("2. Try running main_enhanced.py with the standard test images")
    print("3. If dlib detects faces but face_recognition doesn't, there's a package compatibility issue")
    print("====================================\n")

if __name__ == "__main__":
    main()
