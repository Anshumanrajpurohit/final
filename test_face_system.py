import os
import face_recognition
from services.face_service import FaceService
from services.mysql_service import MySQLService
import logging
import cv2
from PIL import Image
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_face_detection():
    """Test face detection and attribute extraction"""
    face_service = FaceService()
    mysql_service = MySQLService()
    
    # Test image path - using the sample image in temp_images
    test_image_path = os.path.join(os.path.dirname(__file__), "temp_images", "h4.jpg")
    
    if not os.path.exists(test_image_path):
        logger.error(f"Test image not found at {test_image_path}")
        return
    
    logger.info("Starting face detection test...")
    
    try:
        # Detect faces and get attributes
        results = face_service.detect_and_crop_faces(test_image_path)
        
        if not results:
            logger.error("No faces detected in the test image")
            return
        
        for idx, (cropped_path, face_encoding, attributes) in enumerate(results):
            logger.info(f"\nFace {idx + 1} Results:")
            logger.info(f"Age: {attributes['age']}")
            logger.info(f"Gender: {attributes['gender']}")
            logger.info(f"Skin Tone: {attributes['skin_tone']}")
            
            # Display confidence scores
            if hasattr(face_service, 'match_face'):
                known_encodings = mysql_service.get_all_face_encodings()
                if known_encodings:
                    is_match, match_idx = face_service.match_face(face_encoding, known_encodings)
                    if is_match:
                        logger.info(f"Matched with existing face! Visit count will be updated.")
                    else:
                        logger.info("New face detected! Will be added to database.")
            
            # Show the detected face
            img = cv2.imread(cropped_path)
            if img is not None:
                cv2.imshow(f'Detected Face {idx + 1}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_face_detection()
