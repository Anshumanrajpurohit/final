import os
import sys
import logging
import uuid
import shutil
import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDetectionService:
    def __init__(self, temp_dir="temp_images"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to load DNN model if available (more accurate)
        self.use_dnn = False
        try:
            model_path = os.path.join("models", "opencv_face_detector.caffemodel")
            config_path = os.path.join("models", "opencv_face_detector.prototxt")
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.dnn_detector = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self.use_dnn = True
                logger.info("Using DNN face detector")
            else:
                logger.info("Using cascade classifier for face detection")
        except Exception as e:
            logger.warning(f"Failed to load DNN detector: {e}")
    
    def load_image(self, image_path):
        """Load image using PIL first, then OpenCV if that fails"""
        try:
            # Try PIL first (better format handling)
            img = Image.open(image_path).convert('RGB')
            # Convert to numpy array
            img_array = np.array(img)
            # Convert to BGR for OpenCV
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"PIL loading failed: {e}")
        
        # Fall back to OpenCV
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return img
        except Exception as e:
            logger.error(f"OpenCV loading failed: {e}")
            return None
    
    def detect_faces(self, image):
        """Detect faces in an image and return (top, right, bottom, left) coordinates"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Try DNN detection first if available
        if self.use_dnn:
            try:
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                self.dnn_detector.setInput(blob)
                detections = self.dnn_detector.forward()
                
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
                        
                        # Store in (top, right, bottom, left) format
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
    
    def detect_and_crop_faces(self, image_path):
        """Detect faces in an image, crop them, and save to temp directory"""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        # Detect faces
        face_locations = self.detect_faces(image)
        
        # Crop and save faces
        results = []
        for i, location in enumerate(face_locations):
            top, right, bottom, left = location
            
            # Extract face region
            face = image[top:bottom, left:right]
            
            # Generate unique filename
            crop_path = os.path.join(self.temp_dir, f"face_{uuid.uuid4().hex}.jpg")
            
            # Save face image
            cv2.imwrite(crop_path, face)
            
            # Add to results: (crop_path, x, y, width, height)
            width = right - left
            height = bottom - top
            results.append((crop_path, left, top, width, height))
        
        logger.info(f"Detected and cropped {len(results)} faces from {image_path}")
        return results

def patch_enhanced_face_service():
    """
    Replace the problematic face_recognition implementation 
    with our reliable OpenCV implementation
    """
    # Path to the enhanced face service
    service_path = os.path.join("services", "enhanced_face_service.py")
    backup_path = os.path.join("services", "enhanced_face_service.py.bak")
    
    # Create backup
    if os.path.exists(service_path) and not os.path.exists(backup_path):
        shutil.copy(service_path, backup_path)
        logger.info(f"Created backup of {service_path} to {backup_path}")
    
    # Add the improved implementation to the enhanced face service
    service_code = """
    def detect_and_crop_faces_enhanced(self, image_path: str, *args, **kwargs):
        \"\"\"Compatibility wrapper used by main_enhanced.py. Returns list of (crop_path, x, y, w, h).\"\"\"
        logger.info(f"Using OpenCV-based face detection for {image_path}")
        
        # Validate image path
        if not isinstance(image_path, str) or not image_path:
            logger.warning(f"Invalid image path type: {type(image_path)}")
            return []
            
        if not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            return []
            
        try:
            # Load image with PIL (better format handling)
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array for OpenCV
            img_array = np.array(img)
            image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Crop and save faces
            results = []
            for (x, y, w, h) in faces:
                # Extract face region
                face = image[y:y+h, x:x+w]
                
                # Generate unique filename
                crop_path = os.path.join(self.temp_dir, f"tmp_crop_{uuid.uuid4().hex}.jpg")
                
                # Save face image
                cv2.imwrite(crop_path, face)
                
                # Add to results (crop_path, x, y, w, h)
                results.append((crop_path, x, y, w, h))
            
            logger.info(f"OpenCV detected {len(results)} faces in {image_path}")
            return results
        except Exception as e:
            logger.exception(f"OpenCV face detection failed: {e}")
            return []
    """
    
    # Find the right place to insert the code and add it
    # (This is a simple example; in a real scenario, you'd parse the file more carefully)
    logger.info("To use the OpenCV-based face detection, add the code to enhanced_face_service.py")
    logger.info(f"The code to add has been saved to a file for your reference")
    
    # Save the implementation to a file for reference
    with open("opencv_implementation.py", "w") as f:
        f.write(service_code)
    
    logger.info("Implementation saved to opencv_implementation.py")

def main():
    # Create the face detection service
    detector = FaceDetectionService()
    
    # Process command line arguments
    if len(sys.argv) < 2:
        print("Usage: python guaranteed_face_detection.py <image_path>")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    # Detect and crop faces
    results = detector.detect_and_crop_faces(image_path)
    
    if results:
        print(f"Successfully detected {len(results)} faces:")
        for i, (face_path, x, y, width, height) in enumerate(results):
            print(f"  Face {i+1}: saved to {face_path}, position: x={x}, y={y}, width={width}, height={height}")
    else:
        print("No faces detected.")
    
    # Offer to patch the enhanced face service
    patch_enhanced_face_service()

if __name__ == "__main__":
    main()
