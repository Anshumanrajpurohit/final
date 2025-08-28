import face_recognition
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from PIL import Image
import uuid
import logging

logging.basicConfig(level=logging.INFO)

class FaceService:
    def __init__(self, threshold: float = 0.6, min_detection_confidence: float = 0.95, temp_dir: str = "temp_images"):
        """FaceService initialization.

        Args:
            threshold: similarity threshold for matching (default 0.6)
            min_detection_confidence: reserved for future detectors
            temp_dir: directory to store cropped faces
        """
        self.threshold = float(threshold)
        self.min_detection_confidence = float(min_detection_confidence)
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def _classify_skin_tone(self, rgb_values):
        """Simple skin tone classification based on RGB values"""
        r, g, b = rgb_values
        
        # Calculate luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Simple classification based on luminance
        if luminance > 200:
            return "Very Light"
        elif luminance > 170:
            return "Light"
        elif luminance > 140:
            return "Medium"
        elif luminance > 100:
            return "Dark"
        else:
            return "Very Dark"
        
    def match_face(self, face_encoding: np.ndarray, known_encodings: List[np.ndarray]) -> Tuple[bool, int]:
        """Match a face encoding against known encodings using 0.6 threshold"""
        if not known_encodings:
            return False, -1
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Check if the best match is below threshold (0.6)
        if face_distances[best_match_index] <= self.threshold:
            return True, best_match_index
        return False, -1

    def analyze_face_attributes(self, face_image: np.ndarray) -> Dict:
        """Analyze face attributes using face_recognition landmarks"""
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                if face_image.dtype != np.uint8:
                    face_image = (face_image * 255).astype(np.uint8)
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image

            # Get face landmarks for analysis
            face_landmarks = face_recognition.face_landmarks(rgb_image)
            
            if not face_landmarks:
                return {
                    'age': None,
                    'gender': None,
                    'skin_tone': None
                }
            
            # Analyze skin tone from face region
            face_roi = rgb_image[face_landmarks[0]['chin'][0][1]:face_landmarks[0]['chin'][6][1],
                               face_landmarks[0]['chin'][0][0]:face_landmarks[0]['chin'][16][0]]
            
            if face_roi.size > 0:
                # Simple skin tone analysis based on average RGB values
                avg_color = np.mean(face_roi, axis=(0, 1))
                skin_tone = self._classify_skin_tone(avg_color)
            else:
                skin_tone = "Unknown"
            
            return {
                'age': None,  # Age detection would require additional ML model
                'gender': None,  # Gender detection would require additional ML model
                'skin_tone': skin_tone
            }
        except Exception as e:
            print(f"Error analyzing face attributes: {e}")
            return {
                'age': None,
                'gender': None,
                'skin_tone': None
            }
    
    def detect_and_crop_faces(self, image_path: str) -> List[Tuple[str, np.ndarray]]:
        """Detect faces and crop them, return list of (cropped_path, encoding)"""
        try:
            # Validate input file
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return []
            
            # Load image with error handling
            try:
                image = face_recognition.load_image_file(image_path)
            except Exception as load_error:
                print(f"Error loading image {image_path}: {load_error}")
                # Try with PIL as fallback
                try:
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    if image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                except Exception as pil_error:
                    print(f"PIL fallback also failed: {pil_error}")
                    return []
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image, model="hog")  # Use HOG for speed
            
            if not face_locations:
                print(f"No faces detected in {image_path}")
                return []
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) != len(face_locations):
                print(f"Warning: Encoding count mismatch in {image_path}")
            
            cropped_faces = []
            
            # Process each detected face
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                
                # Get face attributes
                attributes = self.analyze_face_attributes(face_image)
                
                # Save cropped face
                cropped_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}.jpg")
                cv2.imwrite(cropped_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                
                # Store face encoding and attributes together
                cropped_faces.append((cropped_path, face_encoding, attributes))
                
                # Check if face already exists in database
                if hasattr(self, 'mysql_service'):
                    # Get all known face encodings
                    known_encodings = self.mysql_service.get_all_face_encodings()
                    is_match, match_index = self.match_face(face_encoding, known_encodings)
                    
                    if is_match:
                        # Update existing person's visit count and attributes
                        person_id = self.mysql_service.get_person_id_by_index(match_index)
                        self.mysql_service.increment_visit_count(person_id)
                        self.mysql_service.update_customer_data(
                            person_id=person_id,
                            age=attributes['age'] if attributes['age'] is not None else 0,
                            gender=attributes['gender'] if attributes['gender'] is not None else 'Unknown',
                            skin_tone=attributes['skin_tone'] if attributes['skin_tone'] is not None else 'Unknown',
                            hair_status='Not detected'
                        )
                    else:
                        # Create new person entry
                        person_id = str(uuid.uuid4())
                        self.mysql_service.insert_customer_data(
                            person_id=person_id,
                            age=attributes['age'] if attributes['age'] is not None else 0,
                            gender=attributes['gender'] if attributes['gender'] is not None else 'Unknown',
                            skin_tone=attributes['skin_tone'] if attributes['skin_tone'] is not None else 'Unknown',
                            hair_status='Not detected'
                        )
                        # Store the face encoding for future matching
                        self.mysql_service.store_face_encoding(person_id, face_encoding)
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                try:
                    # Get face coordinates
                    top, right, bottom, left = face_location
                    
                    # Calculate padding (10% of face size)
                    face_width = right - left
                    face_height = bottom - top
                    padding_x = int(face_width * 0.1)
                    padding_y = int(face_height * 0.1)
                    
                    # Apply padding with bounds checking
                    top = max(0, top - padding_y)
                    right = min(image.shape[1], right + padding_x)
                    bottom = min(image.shape[0], bottom + padding_y)
                    left = max(0, left - padding_x)
                    
                    # Crop face
                    face_image = image[top:bottom, left:right]
                    
                    if face_image.size == 0:
                        print(f"Empty face crop for face {i} in {image_path}")
                        continue
                    
                    # Convert RGB to BGR for OpenCV
                    face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    
                    # Save cropped face
                    crop_filename = f"face_{uuid.uuid4().hex[:8]}_{i}.jpg"
                    crop_path = os.path.join(self.temp_dir, crop_filename)
                    
                    success = cv2.imwrite(crop_path, face_bgr)
                    if not success:
                        print(f"Failed to save cropped face: {crop_path}")
                        continue
                    
                    cropped_faces.append((crop_path, face_encoding))
                    
                except Exception as crop_error:
                    print(f"Error processing face {i} in {image_path}: {crop_error}")
                    continue
            
            print(f"Successfully processed {len(cropped_faces)} faces from {image_path}")
            return cropped_faces
            
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def compare_encodings(self, encoding1: np.ndarray, encoding2: List[float]) -> float:
        """Compare two face encodings and return distance"""
        try:
            # Ensure both are numpy arrays
            if not isinstance(encoding1, np.ndarray):
                encoding1 = np.array(encoding1)
            
            if not isinstance(encoding2, np.ndarray):
                encoding2 = np.array(encoding2)
            
            # Validate encoding dimensions
            if len(encoding1) != 128 or len(encoding2) != 128:
                print(f"Warning: Invalid encoding dimensions: {len(encoding1)}, {len(encoding2)}")
                return 1.0
            
            distance = face_recognition.face_distance([encoding2], encoding1)[0]
            return float(distance)
            
        except Exception as e:
            print(f"Error comparing encodings: {e}")
            return 1.0
    
    def find_matching_person(self, face_encoding: np.ndarray, known_encodings: List[Tuple[str, List[float]]]) -> Optional[str]:
        """Find matching person in known encodings"""
        if not known_encodings:
            return None
            
        best_match_id = None
        best_distance = float('inf')
        
        for person_id, known_encoding in known_encodings:
            try:
                distance = self.compare_encodings(face_encoding, known_encoding)
                
                if distance < self.threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = person_id
                    
            except Exception as e:
                print(f"Error comparing with person {person_id}: {e}")
                continue
        
        if best_match_id:
            print(f"Found match: {best_match_id} (distance: {best_distance:.3f})")
        
        return best_match_id
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary face files"""
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
                        print(f"Error removing {file_path}: {e}")
            
            print(f"Cleaned up {cleaned_count} temporary face files")
            
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if image file is readable"""
        try:
            if not os.path.exists(image_path):
                return False
            
            # Check file size
            if os.path.getsize(image_path) == 0:
                return False
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception:
            return False

    def compare_faces(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return False
            
        # Normalize embeddings first
        embedding1_normalized = embedding1 / np.linalg.norm(embedding1)
        embedding2_normalized = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity with normalized vectors
        similarity = np.dot(embedding1_normalized, embedding2_normalized)
        
        # Add confidence score
        confidence = (similarity - self.threshold) / (1 - self.threshold)
        return similarity >= self.threshold, confidence

    def find_matching_face(self, new_embedding, stored_embeddings):
        best_match = None
        best_similarity = -1
        
        for stored_id, stored_embedding in stored_embeddings.items():
            is_match, confidence = self.compare_faces(new_embedding, stored_embedding)
            if is_match and confidence > best_similarity:
                best_match = stored_id
                best_similarity = confidence
                
        return best_match, best_similarity if best_match else None