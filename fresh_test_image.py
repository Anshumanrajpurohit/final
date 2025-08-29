import os
import sys
import uuid
import argparse
import logging
import shutil
import face_recognition
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_fresh_image(output_path="temp_images/fresh_test_image.jpg"):
    """
    Copy a non-duplicate image to a new unique filename for testing
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # List of potential source images to check
    source_paths = [
        "temp_images/h4.jpg",
        "temp_images/WhatsApp Image 2025-08-27 at 10.13.19 PM.jpeg",
        "h4.jpg"
    ]
    
    for source in source_paths:
        if os.path.exists(source):
            # Create unique name for the output
            name, ext = os.path.splitext(output_path)
            unique_path = f"{name}_{uuid.uuid4().hex[:6]}{ext}"
            
            # Copy and convert the file
            try:
                # Open with PIL to ensure format is correct
                img = Image.open(source).convert('RGB')
                # Save as high-quality JPEG
                img.save(unique_path, format='JPEG', quality=95)
                logger.info(f"Created fresh test image: {unique_path}")
                return unique_path
            except Exception as e:
                logger.error(f"Failed to process image {source}: {e}")
    
    logger.error("No suitable source images found")
    return None

def verify_image(image_path):
    """
    Verify an image is readable and in the correct format
    """
    logger.info(f"Verifying image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False
    
    # Try PIL first
    try:
        pil_img = Image.open(image_path)
        logger.info(f"PIL loaded image: {pil_img.format}, mode={pil_img.mode}, size={pil_img.size}")
        
        # Convert to RGB and numpy array
        rgb_img = pil_img.convert('RGB')
        np_img = np.array(rgb_img)
        logger.info(f"Numpy array: shape={np_img.shape}, dtype={np_img.dtype}")
        
        # Detect faces with face_recognition
        face_locations = face_recognition.face_locations(np_img)
        logger.info(f"Detected {len(face_locations)} faces with PIL+face_recognition")
        
        return True
    except Exception as e:
        logger.error(f"PIL verification failed: {e}")
    
    # Try OpenCV
    try:
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            logger.error("OpenCV failed to load image")
            return False
            
        logger.info(f"OpenCV loaded image: shape={cv_img.shape}, dtype={cv_img.dtype}")
        
        # Convert to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Detect faces with face_recognition
        face_locations = face_recognition.face_locations(rgb_img)
        logger.info(f"Detected {len(face_locations)} faces with OpenCV+face_recognition")
        
        return True
    except Exception as e:
        logger.error(f"OpenCV verification failed: {e}")
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Create a fresh test image and verify it")
    parser.add_argument("--verify", type=str, help="Verify an existing image", default=None)
    parser.add_argument("--output", type=str, help="Output path for fresh image", default="temp_images/fresh_test_image.jpg")
    args = parser.parse_args()
    
    if args.verify:
        # Verify an existing image
        verify_image(args.verify)
    else:
        # Create and verify a fresh image
        fresh_image = create_fresh_image(args.output)
        if fresh_image:
            verify_image(fresh_image)

if __name__ == "__main__":
    main()
