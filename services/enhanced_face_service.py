import os
import logging
import uuid
from typing import List, Tuple, Dict, Any
import numpy as np

import face_recognition
from PIL import Image

# Import MediaPipe properly
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe face detector initialized")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - using OpenCV fallback")

from .age_gender_manager import AgeGenderManager

logger = logging.getLogger(__name__)


class EnhancedFaceService:
    def __init__(self):
        self.threshold = 0.6
        self.min_detection_confidence = 0.5  # Lower this for better detection
        self.temp_dir = "temp_images"
        self.age_gender_manager = AgeGenderManager()
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize MediaPipe face detector if available
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        if self.use_mediapipe:
            self.mediapipe_detector = mp_face_detection.FaceDetection(
                model_selection=1,  # 1 = full-range detection
                min_detection_confidence=self.min_detection_confidence
            )

    def process_face_image(self, image_path: str) -> Dict:
        """Process a face image to extract face encoding, age, and gender."""
        try:
            # Load image using PIL (RGB)
            pil_img = Image.open(image_path).convert('RGB')
            rgb_img = np.array(pil_img)
            face_img = rgb_img[:, :, ::-1]  # RGB->BGR for age/gender model

            # Find face encodings
            face_locations = face_recognition.face_locations(rgb_img, model="hog")
            if not face_locations:
                raise ValueError(f"No faces found in cropped image {image_path}")
            
            # Use the first face location
            face_encoding = face_recognition.face_encodings(rgb_img, [face_locations[0]])[0]
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(face_img, face_locations[0])
            
            # Get age and gender
            age_range, gender = "unknown", "unknown"
            try:
                # Provide a simple wrapper: accept BGR image
                pred = self.age_gender_manager.predict(face_img)
                age_range = pred.get('age_range', 'unknown')
                gender = pred.get('gender', 'unknown')
            except Exception as e:
                logger.warning(f"Age/gender detection failed: {e}")
            
            return {
                "success": True,
                "face_encoding": face_encoding,
                "age_range": age_range,
                "gender": gender,
                "quality_score": quality_score
            }
            
        except Exception as e:
            logger.exception(f"Error processing face image {image_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_face_region(self, image: np.ndarray, face_location: Tuple) -> np.ndarray:
        """Extract face region from image."""
        top, right, bottom, left = face_location
        return image[top:bottom, left:right]

    def _calculate_quality_score(self, image: np.ndarray, face_location: Tuple) -> float:
        """Calculate quality score for a face image."""
        try:
            # Extract metrics
            face_size = image.shape[0] * image.shape[1]
            # Approximate blur detection using simple gradient variance on a single channel
            try:
                import cv2  # local import to avoid enforcing project-wide OpenCV usage
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
            except Exception:
                # Fallback without cv2: use numpy ops on RGB
                gray_np = np.mean(image, axis=2) if image.ndim == 3 else image
                # Simple gradient magnitude
                gy, gx = np.gradient(gray_np.astype(np.float32))
                laplacian_var = float(np.var(gx) + np.var(gy))
                brightness = float(np.mean(gray_np))
                contrast = float(np.std(gray_np))
            
            # Calculate face size ratio
            img_size = image.shape[0] * image.shape[1]
            face_ratio = face_size / img_size if img_size > 0 else 0
            
            # Combined score (normalized)
            blur_score = min(1.0, laplacian_var / 500)  # Higher is better
            size_score = min(1.0, face_ratio * 3)  # Higher is better
            bright_score = 1.0 - abs(brightness - 127.5) / 127.5  # Closer to middle is better
            
            # Weighted score (adjust weights as needed)
            quality_score = (blur_score * 0.5 + size_score * 0.3 + bright_score * 0.2)
            return round(quality_score, 3)
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.0

    def match_face(self, face_encoding: np.ndarray, known_encodings: List[np.ndarray]) -> Tuple[bool, int]:
        """
        Match a face encoding against known encodings.
        
        Args:
            face_encoding: Face encoding to match
            known_encodings: List of known face encodings
            
        Returns:
            Tuple of (is_match, match_index)
        """
        if not known_encodings:
            return False, -1
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]
        
        # Check if it's a match (lower distance is better)
        is_match = best_match_distance <= self.threshold
        
        return is_match, best_match_index if is_match else -1

    def detect_and_crop_faces_enhanced(self, image_path: str, *args, **kwargs) -> List[Tuple[str, int, int, int, int]]:
        """Detect and crop faces with MediaPipe if available. Returns list of (crop_path, x, y, w, h)."""
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
                img = Image.open(image_path).convert('RGB')
                image_rgb = np.array(img)
                image = image_rgb[:, :, ::-1]  # BGR for quality scoring
            except Exception as e:
                logger.warning(f"Failed to load image via PIL: {e}")
                return []
            
            faces = []
            
            # MediaPipe only
            if self.use_mediapipe:
                logger.info(f"Using MediaPipe face detection for {image_path}")
                try:
                    # Process with MediaPipe
                    results = self.mediapipe_detector.process(image_rgb)
                    
                    if results.detections:
                        height, width = image.shape[:2]
                        for detection in results.detections:
                            # Get bounding box
                            bbox = detection.location_data.relative_bounding_box
                            
                            # Convert relative coordinates to absolute
                            x = int(bbox.xmin * width)
                            y = int(bbox.ymin * height)
                            w = int(bbox.width * width)
                            h = int(bbox.height * height)
                            
                            # Add margin for better recognition (15%)
                            margin_x = int(w * 0.15)
                            margin_y = int(h * 0.15)
                            
                            # Ensure coordinates stay within image bounds
                            x = max(0, x - margin_x)
                            y = max(0, y - margin_y)
                            w = min(width - x, w + 2 * margin_x)
                            h = min(height - y, h + 2 * margin_y)
                            
                            faces.append((x, y, w, h))
                        
                        logger.info(f"Detected {len(faces)} faces with MediaPipe")
                        
                except Exception as e:
                    logger.warning(f"MediaPipe detection failed: {e}")
            
            # Crop and save faces
            results = []
            for (x, y, w, h) in faces:
                # Extract face region
                face = image[y:y+h, x:x+w]
                
                # Generate unique filename
                crop_path = os.path.join(self.temp_dir, f"tmp_crop_{uuid.uuid4().hex}.jpg")
                
                # Save face image using PIL
                try:
                    Image.fromarray(face[:, :, ::-1]).save(crop_path, format='JPEG', quality=95)
                except Exception:
                    # fallback using numpy-only conversion
                    Image.fromarray(face[:, :, ::-1].astype(np.uint8)).save(crop_path)
                
                # Add to results (crop_path, x, y, w, h)
                results.append((crop_path, x, y, w, h))
            
            return results
        except Exception as e:
            logger.exception(f"Face detection failed: {e}")
            return []

    def process_batch(self, image_paths, *args, **kwargs) -> List[Dict]:
        """
        Process multiple images or a single image to extract faces and their data.
        Returns a list of face result dictionaries.
        """
        results = []
        
        # Handle both single string and list inputs
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif not isinstance(image_paths, list):
            logger.warning(f"Invalid image path type: {type(image_paths)}")
            return []
        
        # Process each image
        for image_path in image_paths:
            logger.info(f"Processing image: {image_path}")
            
            crops = self.detect_and_crop_faces_enhanced(image_path, *args, **kwargs)
            for crop_path, x, y, w, h in crops:
                try:
                    face_data = self.process_face_image(crop_path)
                    if face_data.get("success", False):
                        results.append({
                            "success": True,
                            "data": {
                                "crop_path": crop_path,
                                "x": x, "y": y, "w": w, "h": h,
                                "face_encoding": face_data.get("face_encoding"),
                                "age_range": face_data.get("age_range", "unknown"),
                                "gender": face_data.get("gender", "unknown"),
                                "quality_score": face_data.get("quality_score", 0.0),
                                "original_image": image_path
                            }
                        })
                    else:
                        results.append({
                            "success": False,
                            "error": face_data.get("error", "Unknown error"),
                            "data": {
                                "crop_path": crop_path,
                                "x": x, "y": y, "w": w, "h": h,
                                "original_image": image_path
                            }
                        })
                except Exception as e:
                    logger.exception(f"Failed to process crop {crop_path}: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "data": {
                            "crop_path": crop_path,
                            "x": x, "y": y, "w": w, "h": h,
                            "original_image": image_path
                        }
                    })
        
        return results

    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            # Clean up temporary cropped face images
            for filename in os.listdir(self.temp_dir):
                if filename.startswith("tmp_crop_"):
                    file_path = os.path.join(self.temp_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")
