import os
import sys
import cv2
import numpy as np
import face_recognition
from PIL import Image
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_face_detection(image_path, output_dir="temp_faces"):
    """
    Test face detection on an image and save detected faces.
    Uses both PIL and OpenCV approaches to compare results.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file info
    print(f"Testing image: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    print(f"File size: {os.path.getsize(image_path)} bytes")
    
    # === METHOD 1: PIL approach ===
    print("\n=== Method 1: PIL approach ===")
    try:
        # Load with PIL and convert to RGB
        pil_img = Image.open(image_path).convert('RGB')
        print(f"PIL image mode: {pil_img.mode}, size: {pil_img.size}")
        
        # Convert to numpy array for face_recognition
        rgb_img = np.array(pil_img)
        print(f"Numpy array: shape={rgb_img.shape}, dtype={rgb_img.dtype}, range=({rgb_img.min()}, {rgb_img.max()})")
        
        # Detect faces
        locations = face_recognition.face_locations(rgb_img, model="hog")
        print(f"PIL approach detected {len(locations)} faces")
        
        # Save detected faces
        for i, loc in enumerate(locations):
            top, right, bottom, left = loc
            face = rgb_img[top:bottom, left:right]
            face_path = os.path.join(output_dir, f"pil_face_{i}_{uuid.uuid4().hex}.jpg")
            # Convert RGB to BGR for OpenCV saving
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(face_path, face_bgr)
            print(f"  Face {i+1}: Saved to {face_path}")
    except Exception as e:
        print(f"PIL approach failed: {e}")
    
    # === METHOD 2: OpenCV approach ===
    print("\n=== Method 2: OpenCV approach ===")
    try:
        # Load with OpenCV
        img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            print("Failed to load image with OpenCV")
        else:
            print(f"OpenCV image: shape={img_cv.shape}, dtype={img_cv.dtype}, range=({img_cv.min()}, {img_cv.max()})")
            
            # Convert BGR to RGB
            rgb_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            print(f"RGB image: shape={rgb_cv.shape}, dtype={rgb_cv.dtype}, range=({rgb_cv.min()}, {rgb_cv.max()})")
            
            # Detect faces
            locations = face_recognition.face_locations(rgb_cv, model="hog")
            print(f"OpenCV approach detected {len(locations)} faces")
            
            # Save detected faces
            for i, loc in enumerate(locations):
                top, right, bottom, left = loc
                face = img_cv[top:bottom, left:right]  # Use BGR image for saving
                face_path = os.path.join(output_dir, f"cv_face_{i}_{uuid.uuid4().hex}.jpg")
                cv2.imwrite(face_path, face)
                print(f"  Face {i+1}: Saved to {face_path}")
    except Exception as e:
        print(f"OpenCV approach failed: {e}")
    
    # === METHOD 3: Fixed format approach ===
    print("\n=== Method 3: Fixed format approach ===")
    try:
        # Load with PIL
        pil_img = Image.open(image_path).convert('RGB')
        
        # Save as a new file to ensure correct format
        fixed_path = os.path.join(output_dir, f"fixed_{os.path.basename(image_path)}")
        pil_img.save(fixed_path, format="JPEG", quality=95)
        print(f"Saved fixed format image to: {fixed_path}")
        
        # Load the fixed image with OpenCV
        img_fixed = cv2.imread(fixed_path, cv2.IMREAD_COLOR)
        rgb_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        locations = face_recognition.face_locations(rgb_fixed, model="hog")
        print(f"Fixed format approach detected {len(locations)} faces")
        
        # Save detected faces
        for i, loc in enumerate(locations):
            top, right, bottom, left = loc
            face = img_fixed[top:bottom, left:right]
            face_path = os.path.join(output_dir, f"fixed_face_{i}_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(face_path, face)
            print(f"  Face {i+1}: Saved to {face_path}")
    except Exception as e:
        print(f"Fixed format approach failed: {e}")
    
    return

if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "temp_images/WhatsApp Image 2025-08-27 at 10.13.19 PM.jpeg"
    
    test_face_detection(image_path)
