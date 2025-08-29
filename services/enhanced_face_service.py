import os
import logging
import uuid
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import face_recognition
import mediapipe as mp

# Import MediaPipe properly
try:
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

    def _ensure_init(self):
        """Ensure logger and MediaPipe detector are ready."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(__name__)
        if not hasattr(self, "_mp_detector"):
            # MediaPipe Face Detection: model_selection=1 for >2m, 0 for close-up
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
            try:
                self.logger.info("MediaPipe face detector initialized")
            except Exception:
                pass

    def _detect_faces_mediapipe(self, image_path: str):
        """Detect faces using MediaPipe and return list of dicts: x,y,w,h (pixels)."""
        self._ensure_init()
        # Load as RGB
        image_rgb = face_recognition.load_image_file(image_path)
        img_h, img_w = image_rgb.shape[:2]
        res = self._mp_detector.process(image_rgb)
        boxes = []
        if res and getattr(res, "detections", None):
            for det in res.detections:
                rbb = det.location_data.relative_bounding_box
                x = max(0, int(rbb.xmin * img_w))
                y = max(0, int(rbb.ymin * img_h))
                w = max(1, int(rbb.width * img_w))
                h = max(1, int(rbb.height * img_h))
                boxes.append({"x": x, "y": y, "w": w, "h": h})
        return boxes

    def _expand_bbox(self, x, y, w, h, img_w, img_h, ratio=0.25):
        dx = int(w * ratio); dy = int(h * ratio)
        nx = max(0, x - dx); ny = max(0, y - dy)
        nr = min(img_w, x + w + dx); nb = min(img_h, y + h + dy)
        return nx, ny, nr, nb  # left, top, right, bottom

    def _encode_from_bbox(self, image_rgb, x, y, w, h, try_ratios=(0.25, 0.5, 0.8)):
        img_h, img_w = image_rgb.shape[:2]
        for r in try_ratios:
            left, top, right, bottom = self._expand_bbox(x, y, w, h, img_w, img_h, ratio=r)
            loc = (top, right, bottom, left)  # (t, r, b, l)
            encs = face_recognition.face_encodings(image_rgb, known_face_locations=[loc], num_jitters=1)
            if encs:
                return encs[0], (left, top, right, bottom)
        # Fallback small HOG in expanded crop (without cv2)
        left, top, right, bottom = self._expand_bbox(x, y, w, h, img_w, img_h, ratio=1.0)
        crop = image_rgb[top:bottom, left:right]
        if crop.size > 0:
            locs = face_recognition.face_locations(crop, model="hog")
            if locs:
                t, r, b, l = locs[0]
                loc_abs = (top + t, left + r, top + b, left + l)
                encs = face_recognition.face_encodings(image_rgb, known_face_locations=[loc_abs], num_jitters=1)
                if encs:
                    return encs[0], (left, top, right, bottom)
        return None, None

    def process_batch(self, image_path: str):
        """
        MediaPipe-only detection; encode directly from original image using detected bboxes.
        """
        self._ensure_init()
        results = []
        try:
            # Load once (RGB)
            image_rgb = face_recognition.load_image_file(image_path)
            img_h, img_w = image_rgb.shape[:2]

            detections = self._detect_faces_mediapipe(image_path)
            detected_count = len(detections) if detections else 0

            success_count = 0
            for det in detections or []:
                x = int(det.get("x", 0)); y = int(det.get("y", 0))
                w = int(det.get("w", 0)); h = int(det.get("h", 0))

                encoding, enc_bbox = self._encode_from_bbox(image_rgb, x, y, w, h)
                success = encoding is not None
                success_count += 1 if success else 0

                # Build face chip for age/gender
                if enc_bbox:
                    l, t, r, b = enc_bbox
                else:
                    l, t, r, b = self._expand_bbox(x, y, w, h, img_w, img_h, ratio=0.25)
                face_rgb = image_rgb[t:b, l:r]
                face_bgr = face_rgb[..., ::-1] if face_rgb.size > 0 else None

                age_range, gender = "unknown", "unknown"
                if face_bgr is not None and hasattr(self, "age_gender_manager") and getattr(self.age_gender_manager, "predict", None):
                    try:
                        pred = self.age_gender_manager.predict(face_bgr) or {}
                        age_range = pred.get("age_range", "unknown")
                        gender = pred.get("gender", "unknown")
                    except Exception as e:
                        self.logger.warning(f"Age/gender detection failed: {e}")

                quality_score = 0.0
                if face_rgb.size > 0:
                    try:
                        gy = np.gradient(face_rgb.astype(np.float32), axis=(0, 1))
                        fm = (gy[0] ** 2 + gy[1] ** 2).mean()
                        quality_score = float(fm)
                    except Exception:
                        pass

                self.logger.info(f"Face bbox ({x},{y},{w},{h}) -> encoding={'OK' if success else 'FAIL'}")

                results.append({
                    "success": success,
                    "data": {
                        "x": x, "y": y, "w": w, "h": h,
                        "quality_score": quality_score,
                        "age_range": age_range,
                        "gender": gender,
                        "face_encoding": encoding.tolist() if (success and hasattr(encoding, "tolist")) else (encoding if success else None),
                    }
                })

            if detected_count:
                self.logger.info(f"Detected {detected_count} faces with MediaPipe, encodings OK: {success_count}")

        except Exception as e:
            # Ensure logger exists even if init partially failed
            try:
                self.logger.exception(f"Error processing image {image_path}: {e}")
            except Exception:
                print(f"[EnhancedFaceService] Error processing image {image_path}: {e}")
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
