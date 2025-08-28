import os
import logging
import cv2
from services.enhanced_face_service import EnhancedFaceService
from services.enhanced_mysql_service import EnhancedMySQLService
from config.database import MySQLConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_age_gender_detection():
    """Test age and gender detection functionality"""
    logger.info("Starting age and gender detection test...")
    
    # Initialize services
    try:
        face_service = EnhancedFaceService()
        mysql_config = MySQLConfig()
        mysql_service = EnhancedMySQLService(mysql_config)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        return
    
    # Test with a sample image
    test_image = "temp_images/h4.jpg"  # Using existing test image
    
    if not os.path.exists(test_image):
        logger.error(f"Test image not found: {test_image}")
        return
    
    try:
        # Process face with age and gender detection
        face_data = face_service.process_face_image(test_image)
        
        logger.info("Face processing results:")
        logger.info(f"Age Range: {face_data['age_range']}")
        logger.info(f"Gender: {face_data['gender']}")
        logger.info(f"Age Confidence: {face_data['age_confidence']:.2f}")
        logger.info(f"Gender Confidence: {face_data['gender_confidence']:.2f}")
        logger.info(f"Quality Score: {face_data['quality_score']:.2f}")
        
        # Extract age number from range (e.g., "25-32" -> 25)
        age = int(face_data['age_range'].split('-')[0]) if face_data['age_range'] != 'unknown' else None
        
        # Prepare person data
        person_data = {
            'face_encoding': face_data['face_encoding'],
            'age': age,
            'gender': face_data['gender'],
            'quality_score': face_data['quality_score']
        }
        
        # Insert new person
        logger.info("\nTesting database operations...")
        person_id = mysql_service.insert_new_person(person_data)
        
        if person_id:
            logger.info(f"Successfully created new person with ID: {person_id}")
            
            # Update attributes
            update_success = mysql_service.update_person_attributes(
                person_id,
                age=age,
                gender=face_data['gender']
            )
            if update_success:
                logger.info("Successfully updated person attributes")
            else:
                logger.error("Failed to update person attributes")
        else:
            logger.error("Failed to create new person")
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_age_gender_detection()
