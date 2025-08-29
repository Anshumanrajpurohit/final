import cv2
import numpy as np
import face_recognition
import os
import uuid
import sys
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_face_recognition_compatible_image(image_path):
    """
    Load an image and ensure it's compatible with face_recognition library.
    Returns: RGB numpy array of dtype uint8
    """
    logger.info(f"Processing image: {image_path}")
    
    # First try loading with PIL which handles more formats correctly
    try:
        # Load with PIL and convert to RGB
        pil_img = Image.open(image_path).convert('RGB')
        # Convert PIL image to numpy array
        image = np.array(pil_img)
        logger.info(f"PIL loaded image: shape={image.shape}, dtype={image.dtype}")
        return image
    except Exception as e:
        logger.warning(f"PIL failed to load image: {e}")
    
    # Fallback to OpenCV
    try:
        # Read image with OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"OpenCV failed to load image: {image_path}")
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"OpenCV loaded image: shape={image.shape}, dtype={image.dtype}")
        return image
    except Exception as e:
        logger.error(f"OpenCV failed to load image: {e}")
        return None

def detect_and_save_faces(image_path, output_dir="temp_images", detection_model="hog"):
    """
    Detect faces in an image and save them to output directory.
    Returns: List of (face_file_path, x, y, width, height) tuples
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and ensure image is compatible
    image = ensure_face_recognition_compatible_image(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return []
    
    # Detect face locations
    try:
        logger.info(f"Detecting faces using {detection_model} model")
        face_locations = face_recognition.face_locations(image, model=detection_model)
        logger.info(f"Found {len(face_locations)} faces")
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return []
    
    # Process each face
    results = []
    for i, location in enumerate(face_locations):
        top, right, bottom, left = location
        
        # Ensure coordinates are within image bounds
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.shape[0], bottom)
        right = min(image.shape[1], right)
        
        # Skip invalid faces
        if bottom <= top or right <= left:
            logger.warning(f"Invalid face coordinates: top={top}, right={right}, bottom={bottom}, left={left}")
            continue
        
        # Extract face
        face_image = image[top:bottom, left:right]
        
        # Convert back to BGR for saving with OpenCV
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        # Generate unique filename
        face_filename = f"face_{uuid.uuid4()}.jpg"
        face_path = os.path.join(output_dir, face_filename)
        
        # Save face image
        cv2.imwrite(face_path, face_bgr)
        logger.info(f"Saved face to: {face_path}")
        
        # Add to results
        width = right - left
        height = bottom - top
        results.append((face_path, left, top, width, height))
    
    return results

def main():
    # Get image path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python detect_faces.py <image_path>")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    # Detect and save faces
    results = detect_and_save_faces(image_path)
    
    if not results:
        print("No faces detected or an error occurred.")
    else:
        print(f"Detected {len(results)} faces:")
        for i, (face_path, x, y, width, height) in enumerate(results):
            print(f"  Face {i+1}: saved to {face_path}, position: x={x}, y={y}, width={width}, height={height}")

if __name__ == "__main__":
    main()
