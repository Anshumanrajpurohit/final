from PIL import Image
import sys
import os
import cv2
import numpy as np
import face_recognition
import uuid
import logging

logger = logging.getLogger(__name__)

def convert_to_rgb(path):
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return False
        
    try:
        # Open and convert to RGB
        img = Image.open(path)
        rgb_img = img.convert('RGB')
        
        # Save back to the same location
        rgb_img.save(path, format='JPEG', quality=95)
        print(f"Successfully converted {path} to RGB JPEG")
        return True
    except Exception as e:
        print(f"Error converting image: {e}")
        return False

def detect_and_crop_faces_enhanced(image_path: str, *args, **kwargs) -> list:
    """Compatibility wrapper used by main_enhanced.py. Returns list of (crop_path, x, y, w, h)."""
    detection_model = kwargs.get("detection_model", "hog")  # accept possible kwarg
    try:
        # Prefer an existing implementation if available
        parent = super()
        if hasattr(parent, "detect_and_crop_faces"):
            return parent.detect_and_crop_faces(image_path, *args, **kwargs)
        if hasattr(self, "detect_and_crop_faces"):
            return self.detect_and_crop_faces(image_path, *args, **kwargs)
    except Exception:
        logger.exception("detect_and_crop_faces_enhanced fallback failed")

    # Fallback: robust image normalization then detect+crop and save faces
    try:
        # Ensure path is a string
        if not isinstance(image_path, str) or not image_path:
            logger.warning(f"Invalid image path: {type(image_path)}")
            return []
            
        # Handle URL case
        if image_path.startswith('http'):
            logger.info(f"URL detected: {image_path}")
            # For URLs, you could download them first, but here we just skip
            return []
            
        # Check if file exists
        if not os.path.exists(image_path):
            logger.warning(f"File does not exist: {image_path}")
            return []
            
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Force COLOR mode
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            return []

        # Convert to RGB for face_recognition
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Force correct image format (8-bit RGB)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
            
        locations = face_recognition.face_locations(rgb, model=detection_model)
        results = []
        for i, loc in enumerate(locations):
            top, right, bottom, left = loc
            # Clamp coords to image boundaries
            top = max(0, top); left = max(0, left)
            bottom = min(image.shape[0], bottom); right = min(image.shape[1], right)
            if bottom <= top or right <= left:
                continue
            crop = image[top:bottom, left:right]
            crop_path = os.path.join(self.temp_dir, f"tmp_crop_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(crop_path, crop)
            results.append((crop_path, left, top, right - left, bottom - top))
        return results
    except Exception:
        logger.exception("detect_and_crop_faces_enhanced simple fallback failed")
        return []

def test_with_image(image_path):
    print(f"Testing image: {image_path}")
    
    # Ensure the file exists
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    # Try different loading methods
    try:
        # Method 1: OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            print("OpenCV loaded the image successfully")
            print(f"Shape: {img_cv.shape}, dtype: {img_cv.dtype}")
            rgb_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Try face detection
            locations = face_recognition.face_locations(rgb_cv, model="hog")
            print(f"OpenCV faces detected: {len(locations)}")
        else:
            print("OpenCV failed to load the image")
            
        # Method 2: PIL
        img_pil = Image.open(image_path).convert('RGB')
        print("PIL loaded the image successfully")
        img_np = np.array(img_pil)
        print(f"Shape: {img_np.shape}, dtype: {img_np.dtype}")
        
        # Try face detection
        locations = face_recognition.face_locations(img_np, model="hog")
        print(f"PIL faces detected: {len(locations)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = "temp_images/WhatsApp Image 2025-08-27 at 10.13.19 PM.jpeg"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    convert_to_rgb(path)
    test_with_image(path)