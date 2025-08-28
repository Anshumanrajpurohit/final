import cv2
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional

class AgeGenderDetector:
    def __init__(self, model_dir="models"):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        
        # Age ranges
        self.age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        
        # Load models
        self.age_net = self._load_network("age_deploy.prototxt", "age_net.caffemodel")
        self.gender_net = self._load_network("gender_deploy.prototxt", "gender_net.caffemodel")
        
    def _load_network(self, proto_file: str, model_file: str) -> cv2.dnn.Net:
        """Load a Caffe model"""
        proto_path = os.path.join(self.model_dir, proto_file)
        model_path = os.path.join(self.model_dir, model_file)
        
        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found: {proto_path} or {model_path}")
            
        return cv2.dnn.readNet(proto_path, model_path)
        
    def detect_age_gender(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Detect age and gender from a face image using OpenCV DNN
        
        Args:
            face_image: numpy array of the face image (BGR format)
            
        Returns:
            Dictionary with age_range and gender
        """
        try:
            # Preprocess image
            blob = cv2.dnn.blobFromImage(
                face_image, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            # Predict gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = 'M' if gender_preds[0].argmax() == 1 else 'F'
            gender_confidence = float(gender_preds[0].max())
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = self.age_ranges[age_idx]
            age_confidence = float(age_preds[0].max())
            
            return {
                'age_range': age_range,
                'gender': gender,
                'age_confidence': age_confidence,
                'gender_confidence': gender_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error in age/gender detection: {str(e)}")
            return {
                'age_range': 'unknown',
                'gender': 'unknown',
                'age_confidence': 0.0,
                'gender_confidence': 0.0
            }
    
    def is_valid_prediction(self, prediction: Dict[str, any]) -> bool:
        """Validate if the prediction is valid"""
        return (prediction['age_range'] != 'unknown' and
                prediction['gender'] != 'unknown' and
                prediction['age_confidence'] > 0.5 and
                prediction['gender_confidence'] > 0.5)
