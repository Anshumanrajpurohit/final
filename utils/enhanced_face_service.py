"""
Enhanced Face Service with Performance Optimization
Integrates with PerformanceOptimizer for better efficiency
"""
import face_recognition
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Import our performance optimizer
from utils.performance_optimizer import PerformanceOptimizer

class EnhancedFaceService:
    def __init__(self, threshold: float = 0.6, max_workers: int = 4):
        self.threshold = threshold
        self.max_workers = max_workers
        self.temp_dir = "temp_images"
        self.performance_optimizer = PerformanceOptimizer()
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Face detection models
        self.detection_models = {
            'hog': 'hog',  # Faster, less accurate
            'cnn': 'cnn'   # Slower, more accurate
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_face_size': 80,  # Minimum face size in pixels
            'min_confidence': 0.5,  # Minimum detection confidence
            'max_blur': 100,  # Maximum blur threshold
            'min_brightness': 30,  # Minimum brightness
            'max_brightness': 250  # Maximum brightness
        }
        
        # Load attribute extraction models (placeholders)
        self.age_model = self._load_age_model()
        self.gender_model = self._load_gender_model()
        self.skin_tone_model = self._load_skin_tone_model()
        self.hair_model = self._load_hair_model()
    
    def _load_age_model(self):
        # TODO: Replace with actual model loading code
        return None
    
    def _load_gender_model(self):
        # TODO: Replace with actual model loading code
        return None
    
    def _load_skin_tone_model(self):
        # TODO: Replace with actual model loading code
        return None
    
    def _load_hair_model(self):
        # TODO: Replace with actual model loading code
        return None
    
    def extract_attributes(self, face_image: np.ndarray) -> dict:
        """
        Extract age, gender, skin tone, and hair status from a face image.
        Replace the mock logic with actual model inference.
        """
        # TODO: Replace with actual inference using loaded models
        # Example mock output:
        attributes = {
            'age': 25,  # int(self.age_model.predict(face_image))
            'gender': 'male',  # self.gender_model.predict(face_image)
            'skin_tone': 'medium',  # self.skin_tone_model.predict(face_image)
            'hair_status': 'hair'  # self.hair_model.predict(face_image)
        }
        return attributes
    
    def detect_and_crop_faces_enhanced(self, image_path: str, 
                                     detection_model: str = 'hog',
                                     quality_check: bool = True) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Enhanced face detection with quality assessment and performance optimization
        Returns: List of (cropped_path, encoding, quality_metrics)
        """
        try:
            # Validate input file
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return []
            
            # Load image with error handling
            image = self._load_image_safely(image_path)
            if image is None:
                return []
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                image, 
                model=detection_model,
                number_of_times_to_upsample=1
            )
            
            if not face_locations:
                self.logger.info(f"No faces detected in {image_path}")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) != len(face_locations):
                self.logger.warning(f"Encoding count mismatch in {image_path}")
            
            cropped_faces = []
            
            # Process each detected face
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                try:
                    # Crop face with padding
                    cropped_face, crop_path = self._crop_face_with_padding(image, face_location, i)
                    
                    if cropped_face is None:
                        continue
                    
                    # Quality assessment
                    quality_metrics = {}
                    if quality_check:
                        quality_metrics = self._assess_face_quality(cropped_face)
                        
                        # Skip low-quality faces
                        if not self._is_face_quality_acceptable(quality_metrics):
                            self.logger.info(f"Skipping low-quality face {i} in {image_path}")
                            continue
                    
                    cropped_faces.append((crop_path, face_encoding, quality_metrics))
                    
                except Exception as crop_error:
                    self.logger.error(f"Error processing face {i} in {image_path}: {crop_error}")
                    continue
            
            self.logger.info(f"Successfully processed {len(cropped_faces)} faces from {image_path}")
            return cropped_faces
            
        except Exception as e:
            self.logger.error(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def _load_image_safely(self, image_path: str) -> Optional[np.ndarray]:
        """Safely load image with multiple fallback methods"""
        try:
            # Try face_recognition first
            image = face_recognition.load_image_file(image_path)
            return image
        except Exception as load_error:
            self.logger.warning(f"face_recognition failed: {load_error}")
            
            try:
                # Try PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                
                # Handle different color formats
                if len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    elif image.shape[2] == 1:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                return image
                
            except Exception as pil_error:
                self.logger.error(f"PIL fallback also failed: {pil_error}")
                return None
    
    def _crop_face_with_padding(self, image: np.ndarray, face_location: Tuple, 
                               face_index: int) -> Tuple[Optional[np.ndarray], str]:
        """Crop face with intelligent padding"""
        try:
            top, right, bottom, left = face_location
            
            # Calculate padding (15% of face size for better context)
            face_width = right - left
            face_height = bottom - top
            padding_x = int(face_width * 0.15)
            padding_y = int(face_height * 0.15)
            
            # Apply padding with bounds checking
            top = max(0, top - padding_y)
            right = min(image.shape[1], right + padding_x)
            bottom = min(image.shape[0], bottom + padding_y)
            left = max(0, left - padding_x)
            
            # Crop face
            face_image = image[top:bottom, left:right]
            
            if face_image.size == 0:
                self.logger.warning(f"Empty face crop for face {face_index}")
                return None, ""
            
            # Convert RGB to BGR for OpenCV
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Save cropped face
            crop_filename = f"face_{uuid.uuid4().hex[:8]}_{face_index}.jpg"
            crop_path = os.path.join(self.temp_dir, crop_filename)
            
            success = cv2.imwrite(crop_path, face_bgr)
            if not success:
                self.logger.error(f"Failed to save cropped face: {crop_path}")
                return None, ""
            
            return face_image, crop_path
            
        except Exception as e:
            self.logger.error(f"Error cropping face: {e}")
            return None, ""
    
    def _assess_face_quality(self, face_image: np.ndarray) -> Dict[str, float]:
        """Assess the quality of a detected face"""
        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Calculate quality metrics
            quality_metrics = {
                'brightness': np.mean(gray),
                'contrast': np.std(gray),
                'blur': self._calculate_blur(gray),
                'face_size': min(face_image.shape[:2]),
                'aspect_ratio': face_image.shape[1] / face_image.shape[0]
            }
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(quality_metrics)
            quality_metrics['overall_score'] = quality_score
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing face quality: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_blur(self, gray_image: np.ndarray) -> float:
        """Calculate blur using Laplacian variance"""
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            return laplacian.var()
        except:
            return 0.0
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics"""
        score = 0.0
        
        # Brightness score (0-1)
        brightness = metrics['brightness']
        if self.quality_thresholds['min_brightness'] <= brightness <= self.quality_thresholds['max_brightness']:
            score += 0.3
        elif brightness > 0:
            score += 0.1
        
        # Contrast score (0-1)
        contrast = metrics['contrast']
        if contrast > 20:
            score += 0.3
        elif contrast > 10:
            score += 0.1
        
        # Blur score (0-1)
        blur = metrics['blur']
        if blur > self.quality_thresholds['max_blur']:
            score += 0.2
        elif blur > 50:
            score += 0.1
        
        # Size score (0-1)
        face_size = metrics['face_size']
        if face_size >= self.quality_thresholds['min_face_size']:
            score += 0.2
        elif face_size > 40:
            score += 0.1
        
        return min(score, 1.0)
    
    def _is_face_quality_acceptable(self, quality_metrics: Dict[str, float]) -> bool:
        """Check if face quality meets minimum standards"""
        return quality_metrics.get('overall_score', 0) >= 0.3
    
    def compare_encodings_enhanced(self, encoding1: np.ndarray, encoding2: List[float]) -> float:
        """Enhanced encoding comparison with validation"""
        try:
            # Ensure both are numpy arrays
            if not isinstance(encoding1, np.ndarray):
                encoding1 = np.array(encoding1)
            
            if not isinstance(encoding2, np.ndarray):
                encoding2 = np.array(encoding2)
            
            # Validate encoding dimensions
            if len(encoding1) != 128 or len(encoding2) != 128:
                self.logger.warning(f"Invalid encoding dimensions: {len(encoding1)}, {len(encoding2)}")
                return 1.0
            
            # Use performance optimizer for comparison
            distance = face_recognition.face_distance([encoding2], encoding1)[0]
            return float(distance)
            
        except Exception as e:
            self.logger.error(f"Error comparing encodings: {e}")
            return 1.0
    
    def find_matching_person_enhanced(self, face_encoding: np.ndarray, 
                                    known_encodings: List[Tuple[str, List[float]]],
                                    use_faiss: bool = True) -> Optional[str]:
        """Enhanced person matching with performance optimization"""
        if not known_encodings:
            return None
        
        # Use performance optimizer for efficient comparison
        return self.performance_optimizer.efficient_embedding_comparison(
            face_encoding, known_encodings, self.threshold, use_faiss
        )
    
    def batch_process_images(self, image_paths: List[str], 
                           detection_model: str = 'hog',
                           quality_check: bool = True) -> List[Tuple[str, List]]:
        """Process multiple images in parallel"""
        return self.performance_optimizer.parallel_face_processing(image_paths, self.max_workers)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_optimizer.get_performance_stats()
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary face files with performance monitoring"""
        try:
            if not os.path.exists(self.temp_dir):
                return
                
            cleaned_count = 0
            for filename in os.listdir(self.temp_dir):
                if filename.startswith('face_') and filename.endswith('.jpg'):
                    file_path = os.path.join(self.temp_dir, filename)
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.error(f"Error removing {file_path}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} temporary face files")
            
            # Check if garbage collection is needed
            if self.performance_optimizer.should_garbage_collect():
                self.performance_optimizer.force_garbage_collection()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if image file is readable and suitable for processing"""
        try:
            if not os.path.exists(image_path):
                return False
            
            # Check file size
            if os.path.getsize(image_path) == 0:
                return False
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                img.verify()
                
                # Check image dimensions
                img.seek(0)  # Reset to beginning
                width, height = img.size
                
                # Skip very small images
                if width < 100 or height < 100:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get current quality thresholds"""
        return self.quality_thresholds.copy()
    
    def set_quality_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update quality thresholds"""
        self.quality_thresholds.update(thresholds)
        self.logger.info(f"Updated quality thresholds: {thresholds}")
    
    def log_performance_stats(self) -> None:
        """Log current performance statistics"""
        self.performance_optimizer.log_performance_stats()
