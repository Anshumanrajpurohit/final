import cv2
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional, List

logger = logging.getLogger(__name__)

class AgeGenderManager:
    """Manages age and gender detection using pre-trained OpenCV DNN models"""
    
    # Age ranges for classification
    AGE_RANGES = [
        (0, 2), (4, 6), (8, 12), (15, 20), (25, 32),
        (38, 43), (48, 53), (60, 100)
    ]
    
    def __init__(self, models_dir: str = "models"):
        """Initialize age and gender detection models
        
        Args:
            models_dir: Directory containing the model files
        """
        self.models_dir = models_dir
        
        # Load models
        logger.info("Loading age and gender detection models...")
        self.age_net = self._load_model("age_deploy.prototxt", "age_net.caffemodel")
        self.gender_net = self._load_model("gender_deploy.prototxt", "gender_net.caffemodel")
        
        # Gender labels
        self.gender_labels = ['F', 'M']
        
        logger.info("Age and gender detection models loaded successfully")
    
    def _load_model(self, proto_file: str, model_file: str) -> cv2.dnn.Net:
        """Load a caffe model using OpenCV DNN
        
        Args:
            proto_file: Path to .prototxt file
            model_file: Path to .caffemodel file
            
        Returns:
            Loaded OpenCV DNN model
        """
        proto_path = os.path.join(self.models_dir, proto_file)
        model_path = os.path.join(self.models_dir, model_file)
        
        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model files not found. Looking for {proto_path} and {model_path}"
            )
        
        return cv2.dnn.readNet(proto_path, model_path)
    
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input
        
        Args:
            face_img: BGR face image
            
        Returns:
            Preprocessed blob ready for network inference
        """
        # Create a blob from the face image
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )
        return blob
    
    def detect_age_gender(self, face_img: np.ndarray) -> Dict[str, str]:
        """Predict age and gender for a face image
        
        Args:
            face_img: BGR face image
            
        Returns:
            Dictionary containing age_range and gender predictions
        """
        try:
            # Preprocess face
            blob = self._preprocess_face(face_img)
            
            # Gender prediction
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_labels[gender_preds[0].argmax()]
            
            # Age prediction
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = f"{self.AGE_RANGES[age_idx][0]}-{self.AGE_RANGES[age_idx][1]}"
            
            # Calculate confidence scores
            gender_confidence = float(gender_preds[0][gender_preds[0].argmax()])
            age_confidence = float(age_preds[0][age_idx])
            
            return {
                'age_range': age_range,
                'gender': gender,
                'age_confidence': age_confidence,
                'gender_confidence': gender_confidence
            }
            
        except Exception as e:
            logger.error(f"Error in age/gender prediction: {str(e)}")
            return {
                'age_range': 'unknown',
                'gender': 'unknown',
                'age_confidence': 0.0,
                'gender_confidence': 0.0
            }

    def predict(self, face_bgr):
        """
        Wrapper for age/gender inference. Returns dict(age_range, gender).
        Tries common internal method names and normalizes output.
        """
        try:
            if hasattr(self, "predict_age_gender"):
                age, gender = self.predict_age_gender(face_bgr)
                return {"age_range": age, "gender": gender}
            if hasattr(self, "infer"):
                out = self.infer(face_bgr)
                if isinstance(out, dict):
                    return {"age_range": out.get("age_range", "unknown"), "gender": out.get("gender", "unknown")}
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    return {"age_range": out[0], "gender": out[1]}
            # Fallback to existing public API if present
            if hasattr(self, "get_age_and_gender"):
                age, gender = self.get_age_and_gender(face_bgr)
                return {"age_range": age, "gender": gender}
        except Exception as e:
            logging.getLogger(__name__).warning(f"AgeGenderManager.predict fallback failed: {e}")
        return {"age_range": "unknown", "gender": "unknown"}
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, str]]:
        """Predict age and gender for a batch of face images
        
        Args:
            face_images: List of BGR face images
            
        Returns:
            List of dictionaries containing predictions for each face
        """
        results = []
        for face in face_images:
            results.append(self.predict_age_gender(face))
        return results
    
    def annotate_image(self, image: np.ndarray, face_loc: Tuple[int, int, int, int],
                      age_range: str, gender: str) -> np.ndarray:
        """Draw bounding box and labels on image
        
        Args:
            image: Original BGR image
            face_loc: Tuple of (top, right, bottom, left) coordinates
            age_range: Predicted age range
            gender: Predicted gender
            
        Returns:
            Annotated image
        """
        top, right, bottom, left = face_loc
        
        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Prepare label
        label = f"Age: {age_range}, Gender: {gender}"
        
        # Draw label background
        cv2.rectangle(image, (left, top - 20), (right, top), (0, 255, 0), cv2.FILLED)
        
        # Add label text
        cv2.putText(image, label, (left + 6, top - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        
        return image
