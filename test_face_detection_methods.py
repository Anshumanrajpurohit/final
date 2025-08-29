import cv2
import numpy as np
import face_recognition
import os
import sys
import uuid
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image(image_path):
    """Test different ways of loading and processing an image to find what works with face_recognition"""
    
    if not os.path.exists(image_path):
        logger.error(f"File not found: {image_path}")
        return
    
    print(f"\n==== Testing image: {image_path} ====")
    
    # Method 1: PIL direct to numpy
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        print(f"1. PIL method: shape={img_np.shape}, dtype={img_np.dtype}")
        
        # Verify it's contiguous
        print(f"   Contiguous: {np.ascontiguousarray(img_np) is img_np}")
        
        # Try face detection
        try:
            face_locations = face_recognition.face_locations(img_np)
            print(f"   Face detection result: {len(face_locations)} faces found")
        except Exception as e:
            print(f"   Face detection failed: {e}")
    except Exception as e:
        print(f"1. PIL method failed: {e}")
    
    # Method 2: OpenCV then BGR->RGB
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print("2. OpenCV method: Failed to load image")
        else:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            print(f"2. OpenCV method: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
            
            # Try face detection
            try:
                face_locations = face_recognition.face_locations(img_rgb)
                print(f"   Face detection result: {len(face_locations)} faces found")
            except Exception as e:
                print(f"   Face detection failed: {e}")
    except Exception as e:
        print(f"2. OpenCV method failed: {e}")
    
    # Method 3: Use face_recognition's own loader
    try:
        img_fr = face_recognition.load_image_file(image_path)
        print(f"3. face_recognition loader: shape={img_fr.shape}, dtype={img_fr.dtype}")
        
        # Try face detection
        try:
            face_locations = face_recognition.face_locations(img_fr)
            print(f"   Face detection result: {len(face_locations)} faces found")
        except Exception as e:
            print(f"   Face detection failed: {e}")
    except Exception as e:
        print(f"3. face_recognition loader failed: {e}")
    
    # Method 4: Generate a simple compatible image
    try:
        # Create a blank image
        blank_img = np.zeros((400, 400, 3), dtype=np.uint8)
        blank_img.fill(200)  # Light gray
        
        # Draw a face-like shape
        cv2.circle(blank_img, (200, 200), 100, (50, 50, 200), -1)  # Face
        cv2.circle(blank_img, (150, 150), 20, (255, 255, 255), -1)  # Eye
        cv2.circle(blank_img, (250, 150), 20, (255, 255, 255), -1)  # Eye
        
        print(f"4. Generated test image: shape={blank_img.shape}, dtype={blank_img.dtype}")
        
        # Save for reference
        test_path = os.path.join(os.path.dirname(image_path), "test_face.jpg")
        cv2.imwrite(test_path, blank_img)
        print(f"   Saved test image to: {test_path}")
        
        # Try face detection on the test image
        try:
            rgb_blank = cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_blank)
            print(f"   Face detection result: {len(face_locations)} faces found")
        except Exception as e:
            print(f"   Face detection failed: {e}")
    except Exception as e:
        print(f"4. Generated test image failed: {e}")
    
    print("\n==== Testing OpenCV face detection as alternative ====")
    
    # Method 5: OpenCV cascade classifier
    try:
        img_cv = cv2.imread(image_path) if os.path.exists(image_path) else blank_img
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print(f"5. OpenCV cascade: Detected {len(faces)} faces")
        
        # Draw rectangles for visualization
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save result
        output_path = os.path.join(os.path.dirname(image_path), "detected_faces.jpg")
        cv2.imwrite(output_path, img_cv)
        print(f"   Saved result to: {output_path}")
    except Exception as e:
        print(f"5. OpenCV cascade failed: {e}")

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the image: ")
    
    test_image(image_path)

if __name__ == "__main__":
    main()
