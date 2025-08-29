import os
import sys
import cv2
import numpy as np
import uuid
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenCVFaceDetector:
    """A replacement for face_recognition that uses OpenCV directly"""
    
    def __init__(self):
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # DNN face detector (more accurate but slower)
        self.use_dnn = True
        try:
            # Load DNN model if available
            model_path = os.path.join("models", "opencv_face_detector.caffemodel")
            config_path = os.path.join("models", "opencv_face_detector.prototxt")
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.dnn_face_detector = cv2.dnn.readNetFromCaffe(config_path, model_path)
                logger.info("Loaded DNN face detector")
            else:
                self.use_dnn = False
                logger.info("DNN model files not found, using cascade classifier")
        except:
            self.use_dnn = False
            logger.warning("Failed to load DNN detector, using cascade classifier")
    
    def load_image(self, image_path):
        """Load an image from path and convert to correct format"""
        # Try PIL first
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"PIL loading failed: {e}")
        
        # Try OpenCV
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"OpenCV couldn't load image: {image_path}")
            return img
        except Exception as e:
            logger.error(f"OpenCV loading failed: {e}")
            return None
    
    def face_locations(self, image_path):
        """Detect faces in image and return list of (top, right, bottom, left) tuples"""
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Try DNN detection first (if available)
        if self.use_dnn:
            try:
                blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
                self.dnn_face_detector.setInput(blob)
                detections = self.dnn_face_detector.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Confidence threshold
                        x1 = int(detections[0, 0, i, 3] * width)
                        y1 = int(detections[0, 0, i, 4] * height)
                        x2 = int(detections[0, 0, i, 5] * width)
                        y2 = int(detections[0, 0, i, 6] * height)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        # Skip invalid faces
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Add to list in (top, right, bottom, left) format
                        faces.append((y1, x2, y2, x1))
                
                if faces:
                    logger.info(f"DNN detected {len(faces)} faces")
                    return faces
            except Exception as e:
                logger.warning(f"DNN detection failed: {e}")
        
        # Fall back to cascade classifier
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert to (top, right, bottom, left) format
        locations = []
        for (x, y, w, h) in faces:
            locations.append((y, x+w, y+h, x))
        
        logger.info(f"Cascade classifier detected {len(locations)} faces")
        return locations
    
    def crop_faces(self, image_path, output_dir="temp_images"):
        """Detect faces, crop them, and save to output directory"""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return []
        
        # Get face locations
        locations = self.face_locations(image_path)
        
        # Crop and save faces
        results = []
        for i, loc in enumerate(locations):
            top, right, bottom, left = loc
            
            # Ensure coordinates are valid
            if bottom <= top or right <= left:
                continue
                
            # Crop face
            face = img[top:bottom, left:right]
            
            # Generate unique filename
            crop_path = os.path.join(output_dir, f"face_{uuid.uuid4().hex}.jpg")
            
            # Save cropped face
            cv2.imwrite(crop_path, face)
            logger.info(f"Saved face to: {crop_path}")
            
            # Add to results (crop_path, x, y, width, height)
            width = right - left
            height = bottom - top
            results.append((crop_path, left, top, width, height))
        
        return results

def main():
    # Test the detector
    if len(sys.argv) < 2:
        print("Usage: python opencv_face_detector.py <image_path>")
        return
    
    image_path = sys.argv[1]
    detector = OpenCVFaceDetector()
    
    logger.info(f"Testing face detection on: {image_path}")
    results = detector.crop_faces(image_path)
    
    if results:
        logger.info(f"Successfully detected and saved {len(results)} faces")
        for i, (path, x, y, w, h) in enumerate(results):
            logger.info(f"Face {i+1}: {path} at ({x}, {y}) size {w}x{h}")
    else:
        logger.warning("No faces detected")

if __name__ == "__main__":
    main()
