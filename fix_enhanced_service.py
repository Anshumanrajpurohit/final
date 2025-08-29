import os
import sys
import shutil
import logging
import uuid
import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_enhanced_face_service():
    """Create a fixed version of the enhanced face service that uses OpenCV for face detection"""
    
    # Paths
    service_dir = os.path.join("services")
    original_path = os.path.join(service_dir, "enhanced_face_service.py")
    backup_path = os.path.join(service_dir, "enhanced_face_service.py.bak")
    fixed_path = os.path.join(service_dir, "enhanced_face_service_fixed.py")
    
    # Create backup if it doesn't exist
    if os.path.exists(original_path) and not os.path.exists(backup_path):
        shutil.copy(original_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Create a fixed version
    fixed_code = """import os
import logging
import uuid
from typing import List, Tuple, Dict

import face_recognition
import cv2
import numpy as np
from PIL import Image

from .age_gender_manager import AgeGenderManager

logger = logging.getLogger(__name__)


class EnhancedFaceService:
    def __init__(self):
        self.threshold = 0.6
        self.min_detection_confidence = 0.95
        self.temp_dir = "temp_images"
        self.age_gender_manager = AgeGenderManager()
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize OpenCV face detector
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
        except Exception as e:
            logger.warning(f"Failed to load DNN detector: {e}")

    def process_face_image(self, image_path: str) -> Dict:
        """Process a face image to extract face encoding, age, and gender."""
        try:
            # Try PIL first (better format handling)
            try:
                pil_img = Image.open(image_path).convert('RGB')
                rgb_image = np.array(pil_img)
                # Convert back to BGR for OpenCV operations
                image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            except Exception:
                # Fall back to OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image at {image_path}")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use our safer face detection method
            face_locations = self.detect_faces_opencv(rgb_image)
            if not face_locations:
                raise ValueError("No face detected in image")

            # Try to get face encodings with face_recognition
            try:
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if not face_encodings:
                    raise ValueError("Could not generate face encoding")
            except Exception as e:
                logger.warning(f"Face encoding failed with face_recognition: {e}")
                # Create a placeholder encoding (128-dim vector)
                face_encodings = [np.zeros(128, dtype=np.float64)]

            face_image = self._extract_face_region(image, face_locations[0])
            age_gender_data = self.age_gender_manager.detect_age_gender(face_image)

            quality_score = self._calculate_quality_score(image, face_locations[0])

            return {
                'face_encoding': face_encodings[0],
                'age_range': age_gender_data.get('age_range'),
                'gender': age_gender_data.get('gender'),
                'age_confidence': age_gender_data.get('age_confidence'),
                'gender_confidence': age_gender_data.get('gender_confidence'),
                'quality_score': quality_score,
                'location': face_locations[0]
            }
        except Exception as e:
            logger.exception(f"Error processing face image {image_path}: {e}")
            raise

    def _extract_face_region(self, image: np.ndarray, face_location: Tuple) -> np.ndarray:
        top, right, bottom, left = face_location
        return image[top:bottom, left:right]

    def _calculate_quality_score(self, image: np.ndarray, face_location: Tuple) -> float:
        face_image = self._extract_face_region(image, face_location)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)

        blur_score_norm = min(1.0, blur_score / 500)
        brightness_norm = 1.0 - abs(brightness - 128) / 128
        contrast_norm = min(1.0, contrast / 80)

        quality_score = (blur_score_norm * 0.5 + brightness_norm * 0.25 + contrast_norm * 0.25)
        return float(quality_score)

    def match_face(self, face_encoding: np.ndarray, known_encodings: List[np.ndarray]) -> Tuple[bool, int]:
        if not known_encodings:
            return False, -1
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = int(np.argmin(face_distances))
        if face_distances[best_match_index] <= self.threshold:
            return True, best_match_index
        return False, -1
    
    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV and return in face_recognition format.
        Returns list of (top, right, bottom, left) tuples.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        # Try DNN detection first if available
        if getattr(self, 'use_dnn', False):
            try:
                blob = cv2.dnn.blobFromImage(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                                           1.0, (300, 300), [104, 117, 123], False, False)
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

    def detect_and_crop_faces_enhanced(self, image_path: str, *args, **kwargs) -> List[Tuple[str, int, int, int, int]]:
        """Compatibility wrapper used by main_enhanced.py. Returns list of (crop_path, x, y, w, h)."""
        detection_model = kwargs.get("detection_model", "hog")  # accept possible kwarg
        
        # Validate image path
        if not isinstance(image_path, str) or not image_path:
            logger.warning(f"Invalid image path type: {type(image_path)}")
            return []
            
        if not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            return []
        
        try:
            # Load image with PIL for better format handling
            try:
                pil_img = Image.open(image_path).convert('RGB')
                img_array = np.array(pil_img)
                image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                rgb = img_array  # Already in RGB format
            except Exception as e:
                logger.warning(f"PIL loading failed: {e}")
                # Fall back to OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return []
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use OpenCV for detection instead of face_recognition
            locations = self.detect_faces_opencv(rgb)
            logger.info(f"Detected {len(locations)} faces")
            
            results = []
            for i, loc in enumerate(locations):
                top, right, bottom, left = loc
                # clamp coords
                top = max(0, top); left = max(0, left)
                bottom = min(image.shape[0], bottom); right = min(image.shape[1], right)
                if bottom <= top or right <= left:
                    continue
                crop = image[top:bottom, left:right]
                crop_path = os.path.join(self.temp_dir, f"tmp_crop_{uuid.uuid4().hex}.jpg")
                cv2.imwrite(crop_path, crop)
                results.append((crop_path, left, top, right - left, bottom - top))
            return results
        except Exception as e:
            logger.exception(f"detect_and_crop_faces_enhanced failed: {e}")
            return []

    def process_batch(self, image_path: str, *args, **kwargs) -> List[Dict]:
        """
        Minimal compatibility wrapper used by main_enhanced.py.
        Returns list of face result dicts for each detected face (can be empty).
        """
        results = []
        crops = self.detect_and_crop_faces_enhanced(image_path, *args, **kwargs)
        for crop_path, x, y, w, h in crops:
            try:
                face_data = self.process_face_image(crop_path)
                results.append({
                    "crop_path": crop_path,
                    "x": x, "y": y, "w": w, "h": h,
                    "face_encoding": face_data.get("face_encoding"),
                    "age_range": face_data.get("age_range"),
                    "gender": face_data.get("gender"),
                    "quality_score": face_data.get("quality_score"),
                    "location": face_data.get("location"),
                })
            except Exception:
                logger.exception("process_batch: failed to process crop %s", crop_path)
        return results

    def cleanup_temp_files(self):
        """Remove temporary files created by this service. Safe no-op on error."""
        try:
            tmp_dir = getattr(self, "temp_dir", os.path.join(os.getcwd(), "temp_images"))
            temp_list = getattr(self, "temp_files", None)
            if temp_list:
                for p in temp_list:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
            else:
                if os.path.isdir(tmp_dir):
                    for fname in os.listdir(tmp_dir):
                        if fname.startswith("tmp_") or fname.startswith("tmp_crop_"):
                            p = os.path.join(tmp_dir, fname)
                            try:
                                os.remove(p)
                            except Exception:
                                pass
        except Exception:
            logger.exception("cleanup_temp_files encountered an error")
"""
    
    # Write the fixed code to a file
    with open(fixed_path, "w") as f:
        f.write(fixed_code)
    logger.info(f"Created fixed service at: {fixed_path}")
    
    print("\n==== INSTALLATION INSTRUCTIONS ====")
    print(f"1. The fixed service has been created at: {fixed_path}")
    print(f"2. To use it, copy it over the original with this command:")
    print(f"   copy {fixed_path} {original_path}")
    print("3. Then run main_enhanced.py to use the fixed version")
    print("====================================\n")

def main():
    fix_enhanced_face_service()

if __name__ == "__main__":
    main()
