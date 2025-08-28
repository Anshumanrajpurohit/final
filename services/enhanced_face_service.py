import face_recognition
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from PIL import Image
import uuid
import logging
from .age_gender_manager import AgeGenderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFaceService:
    def __init__(self):
        self.threshold = 0.6
        self.min_detection_confidence = 0.95
        self.temp_dir = "temp_images"
        self.age_gender_manager = AgeGenderManager()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def process_face_image(self, image_path: str) -> Dict:
        """
        Process a face image to extract face encoding, age, and gender
        
        Returns:
            Dict containing:
            - face_encoding: numpy array
            - age: int or None
            - gender: str ('M' or 'F') or None
            - quality_score: float
        """
        # Read and convert image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            raise ValueError("No face detected in image")
            
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            raise ValueError("Could not generate face encoding")
            
        # Get age and gender
        face_image = self._extract_face_region(image, face_locations[0])
        age_gender_data = self.age_gender_manager.detect_age_gender(face_image)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(image, face_locations[0])
        
        return {
            'face_encoding': face_encodings[0],
            'age_range': age_gender_data['age_range'],
            'gender': age_gender_data['gender'],
            'age_confidence': age_gender_data['age_confidence'],
            'gender_confidence': age_gender_data['gender_confidence'],
            'quality_score': quality_score,
            'location': face_locations[0]  # Save location for visualization
        }
    
    def _extract_face_region(self, image: np.ndarray, face_location: Tuple) -> np.ndarray:
        """Extract face region from the image using face location"""
        top, right, bottom, left = face_location
        return image[top:bottom, left:right]
    
    def _calculate_quality_score(self, image: np.ndarray, face_location: Tuple) -> float:
        """Calculate face image quality score"""
        # Extract face region
        face_image = self._extract_face_region(image, face_location)
        
        # Convert to grayscale for calculations
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Normalize scores
        blur_score_norm = min(1.0, blur_score / 500)  # Higher is better
        brightness_norm = 1.0 - abs(brightness - 128) / 128  # Closer to middle (128) is better
        contrast_norm = min(1.0, contrast / 80)  # Higher is better (up to a point)
        
        # Weighted combination
        quality_score = (blur_score_norm * 0.5 + 
                        brightness_norm * 0.25 + 
                        contrast_norm * 0.25)
        
        return quality_score
    
    def match_face(self, face_encoding: np.ndarray, known_encodings: List[np.ndarray]) -> Tuple[bool, int]:
        """Match a face encoding against known encodings using 0.6 threshold"""
        if not known_encodings:
            return False, -1
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Check if the best match is below threshold
        if face_distances[best_match_index] <= self.threshold:
            return True, best_match_index
        return False, -1
