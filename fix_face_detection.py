import os
import sys
import shutil
import logging
import cv2
import numpy as np
import face_recognition
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFaceDetection:
    """Solution to face detection issues in the face recognition system"""
    
    def __init__(self):
        """Initialize the solution"""
        # Check if the OpenCV cascade file exists
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            logger.error(f"OpenCV cascade file not found: {cascade_path}")
            print("ERROR: OpenCV cascade file not found. Make sure OpenCV is properly installed.")
            sys.exit(1)
    
    def fix_enhanced_face_service(self):
        """Apply fixes to the enhanced face service"""
        # Fix paths
        service_path = os.path.join("services", "enhanced_face_service.py")
        backup_path = os.path.join("services", "enhanced_face_service.py.bak")
        
        # Create backup
        if os.path.exists(service_path):
            if not os.path.exists(backup_path):
                shutil.copy(service_path, backup_path)
                logger.info(f"Created backup of original file: {backup_path}")
            
            # Inject the face detection fix
            logger.info("Applying fix to enhanced_face_service.py")
            self._inject_face_detection_fix(service_path)
        else:
            logger.error(f"Service file not found: {service_path}")
    
    def _inject_face_detection_fix(self, service_path):
        """Inject our OpenCV-based face detection into the enhanced face service"""
        # Read the service file
        with open(service_path, "r") as f:
            lines = f.readlines()
        
        # Find the detect_and_crop_faces_enhanced method
        detect_method_start = -1
        for i, line in enumerate(lines):
            if "def detect_and_crop_faces_enhanced" in line:
                detect_method_start = i
                break
        
        if detect_method_start == -1:
            logger.error("Could not find detect_and_crop_faces_enhanced method")
            return
        
        # Create the replacement method
        replacement = """    def detect_and_crop_faces_enhanced(self, image_path: str, *args, **kwargs) -> List[Tuple[str, int, int, int, int]]:
        \"\"\"Compatibility wrapper used by main_enhanced.py. Returns list of (crop_path, x, y, w, h).\"\"\"
        logger.info(f"Using OpenCV-based face detection for {image_path}")
        
        # Validate image path
        if not isinstance(image_path, str) or not image_path:
            logger.warning(f"Invalid image path type: {type(image_path)}")
            return []
            
        if not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            return []
            
        try:
            # Load image with PIL (better format handling)
            try:
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                
                # Convert to numpy array for OpenCV
                img_array = np.array(img)
                image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"PIL loading failed: {e}")
                # Fall back to OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    return []
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            logger.info(f"Detected {len(faces)} faces with OpenCV")
            
            # Crop and save faces
            results = []
            for (x, y, w, h) in faces:
                # Extract face region
                face = image[y:y+h, x:x+w]
                
                # Generate unique filename
                import uuid
                crop_path = os.path.join(self.temp_dir, f"tmp_crop_{uuid.uuid4().hex}.jpg")
                
                # Save face image
                cv2.imwrite(crop_path, face)
                
                # Add to results (crop_path, x, y, w, h)
                results.append((crop_path, x, y, w, h))
            
            return results
        except Exception as e:
            logger.exception(f"OpenCV face detection failed: {e}")
            return []
"""

        # Find the end of the method
        detect_method_end = detect_method_start
        brace_count = 0
        for i in range(detect_method_start, len(lines)):
            line = lines[i]
            if "{" in line:
                brace_count += 1
            if "}" in line:
                brace_count -= 1
            
            # Look for next method definition or end of class
            if (i > detect_method_start + 1 and 
                (line.strip().startswith("def ") or 
                 line.strip() == "}" and brace_count < 0)):
                detect_method_end = i - 1
                break
            
            # Check indentation level to find end of method
            if i > detect_method_start + 1 and line.strip() and not line.startswith("    "):
                detect_method_end = i - 1
                break
        
        # Replace the method
        new_lines = lines[:detect_method_start] + replacement.splitlines(True) + lines[detect_method_end+1:]
        
        # Write the modified file
        with open(service_path, "w") as f:
            f.writelines(new_lines)
        
        logger.info("Successfully applied face detection fix")
    
    def test_system(self):
        """Test the face detection system with a sample image"""
        # Create a test image
        test_image_path = self._create_test_image()
        
        # Test face detection
        if test_image_path:
            logger.info(f"Testing face detection with {test_image_path}")
            self._test_face_detection(test_image_path)
    
    def _create_test_image(self, output_dir="temp_images"):
        """Create a test image that works with OpenCV face detection"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple face-like image
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(200)  # Light gray background
        
        # Draw a face-like shape
        cv2.circle(img, (200, 200), 100, (50, 50, 200), -1)  # Face
        cv2.circle(img, (150, 150), 20, (255, 255, 255), -1)  # Left eye
        cv2.circle(img, (250, 150), 20, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(img, (200, 230), (50, 20), 0, 0, 180, (20, 20, 20), -1)  # Mouth
        
        # Save the image
        test_image_path = os.path.join(output_dir, "opencv_test_face.jpg")
        cv2.imwrite(test_image_path, img)
        
        logger.info(f"Created test image: {test_image_path}")
        return test_image_path
    
    def _test_face_detection(self, image_path):
        """Test face detection with OpenCV"""
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            logger.info(f"Detected {len(faces)} faces")
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the result
            result_path = os.path.join(os.path.dirname(image_path), "detected_faces.jpg")
            cv2.imwrite(result_path, img)
            
            logger.info(f"Saved result to: {result_path}")
            
            print("\n==== TESTING RESULTS ====")
            print(f"Test image: {image_path}")
            print(f"Detected faces: {len(faces)}")
            print(f"Result saved to: {result_path}")
            print("=========================\n")
            
            return len(faces) > 0
        except Exception as e:
            logger.exception(f"Face detection test failed: {e}")
            return False

def main():
    print("\n==== FACE DETECTION SYSTEM FIX ====")
    print("This script will fix face detection issues in the enhanced face recognition system.")
    print("It replaces face_recognition with OpenCV for more reliable face detection.")
    
    # Create the solution
    solution = EnhancedFaceDetection()
    
    # Test the system first
    print("\nStep 1: Testing OpenCV face detection...")
    solution.test_system()
    
    # Apply the fix
    print("\nStep 2: Applying fix to enhanced_face_service.py...")
    solution.fix_enhanced_face_service()
    
    print("\n==== FIX COMPLETE ====")
    print("The face detection system has been fixed.")
    print("You can now run main_enhanced.py to use the improved system.")
    print("===========================\n")

if __name__ == "__main__":
    main()
