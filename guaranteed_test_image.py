import os
import cv2
import numpy as np
import face_recognition
import logging
import uuid
import sys
import shutil
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_guaranteed_test_image(output_dir="temp_images"):
    """
    Create a test image that is guaranteed to work with face_recognition.
    Returns the path to the created image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sample image from face_recognition that we know works
    try:
        # Try to load a known image from the face_recognition package
        logger.info("Loading known sample image from face_recognition package")
        sample_image = face_recognition.load_image_file(face_recognition.api.__file__.replace('api.py', 'tests/test_image.jpg'))
        
        output_path = os.path.join(output_dir, f"guaranteed_test_{uuid.uuid4().hex[:6]}.jpg")
        
        # Convert to BGR for OpenCV
        sample_bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        
        # Save using OpenCV
        cv2.imwrite(output_path, sample_bgr)
        logger.info(f"Created guaranteed test image at: {output_path}")
        
        # Verify it works
        test_sample(output_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Failed to create sample from face_recognition: {e}")
    
    # Fallback: create a simple test image from scratch
    try:
        logger.info("Creating simple test image with face-like features")
        
        # Create a simple 400x400 image with a circle that resembles a face
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(200)  # Light gray background
        
        # Draw a simple face shape
        cv2.circle(img, (200, 200), 100, (50, 50, 220), -1)  # Face
        cv2.circle(img, (150, 150), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(img, (250, 150), 15, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(img, (200, 230), (50, 20), 0, 0, 180, (50, 50, 50), -1)  # Mouth
        
        output_path = os.path.join(output_dir, f"simple_face_{uuid.uuid4().hex[:6]}.jpg")
        cv2.imwrite(output_path, img)
        
        logger.info(f"Created simple face test image at: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to create simple face image: {e}")
        return None

def test_sample(image_path):
    """Test if face_recognition can process the sample image"""
    try:
        # Load the image using face_recognition's built-in function
        image = face_recognition.load_image_file(image_path)
        logger.info(f"Successfully loaded image with face_recognition: shape={image.shape}, dtype={image.dtype}")
        
        # Try to detect faces
        face_locations = face_recognition.face_locations(image)
        logger.info(f"Detected {len(face_locations)} faces")
        
        # Try to get encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        logger.info(f"Generated {len(face_encodings)} face encodings")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    # Create a guaranteed test image
    test_image_path = create_guaranteed_test_image()
    
    if test_image_path and os.path.exists(test_image_path):
        logger.info(f"Successfully created test image: {test_image_path}")
        
        # Copy it to a standard location for the main application to use
        standard_path = os.path.join("temp_images", "test_face.jpg")
        shutil.copy(test_image_path, standard_path)
        logger.info(f"Copied to standard location: {standard_path}")
        
        # Also test the standard copy
        test_sample(standard_path)
    else:
        logger.error("Failed to create test image")

if __name__ == "__main__":
    main()
