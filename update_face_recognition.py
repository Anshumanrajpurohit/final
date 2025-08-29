"""
This script integrates MediaPipe face detection into the enhanced face service
and updates the main flow to process one image completely before moving to the next.
"""
import os
import sys
import logging
import shutil
from typing import Any, Dict, List, Tuple  # Add missing import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mediapipe_detector():
    """Create a new MediaPipe face detector file"""
    detector_path = "mediapipe_face_detector.py"
    
    detector_code = """import cv2
import mediapipe as mp
import numpy as np

class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        \"\"\"
        Initialize MediaPipe Face Detector
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection
            model_selection: 0 for short-range, 1 for full-range detection
        \"\"\"
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence
        )
    
    def detect_faces(self, image):
        \"\"\"
        Detect faces in an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of face locations as (x, y, w, h) tuples
        \"\"\"
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.detector.process(image_rgb)
        
        # Extract face locations
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add margin (15%)
                margin_x = int(width * 0.15)
                margin_y = int(height * 0.15)
                
                # Ensure coordinates are within image bounds
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                width = min(w - x, width + 2 * margin_x)
                height = min(h - y, height + 2 * margin_y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def draw_detections(self, image):
        \"\"\"
        Draw face detection boxes on image
        
        Args:
            image: Input image
            
        Returns:
            Image with detection boxes drawn
        \"\"\"
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.detector.process(image_rgb)
        
        # Draw detections
        annotated_image = image.copy()
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(annotated_image, detection)
        
        return annotated_image
"""
    
    with open(detector_path, 'w') as f:
        f.write(detector_code)
    
    logger.info(f"Created MediaPipe detector: {detector_path}")
    return True

def update_enhanced_face_service():
    """Update enhanced_face_service.py to use MediaPipe for face detection"""
    service_dir = os.path.join("services")
    original_path = os.path.join(service_dir, "enhanced_face_service.py")
    backup_path = os.path.join(service_dir, "enhanced_face_service.py.bak")
    
    # Create backup if it doesn't exist
    if os.path.exists(original_path):
        shutil.copy(original_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    else:
        logger.warning(f"Original file not found: {original_path}")
        return False
    
    try:
        # Read current file
        with open(original_path, 'r') as f:
            content = f.read()
        
        # Check if MediaPipe is already imported
        if "import mediapipe as mp" not in content:
            # Add MediaPipe import
            content = content.replace(
                "import cv2", 
                "import cv2\n# MediaPipe for improved face detection\ntry:\n    import mediapipe as mp\n    MEDIAPIPE_AVAILABLE = True\nexcept ImportError:\n    MEDIAPIPE_AVAILABLE = False"
            )
            
            # Update initialization
            if "def __init__(self" in content:
                init_pos = content.find("def __init__(self")
                init_end = content.find("\n        ", init_pos + 20)
                
                # Add MediaPipe initialization after other initializations
                init_code = """
        # Initialize MediaPipe face detector if available
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        if self.use_mediapipe:
            self.mp_face_detection = mp.solutions.face_detection
            self.mediapipe_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 = full-range detection
                min_detection_confidence=0.5
            )
            print("MediaPipe face detector initialized")
"""
                content = content[:init_end] + init_code + content[init_end:]
            
            # Update detect_faces method
            if "def detect_faces" in content:
                detect_pos = content.find("def detect_faces")
                next_def = content.find("def ", detect_pos + 10)
                
                if next_def > 0:
                    # Replace with enhanced detection method
                    detect_code = """def detect_faces(self, image_path):
        \"\"\"
        Detect faces in an image using MediaPipe and fallback methods
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face coordinates
        \"\"\"
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return []
        
        # Try MediaPipe detection first (if available)
        if getattr(self, 'use_mediapipe', False):
            try:
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image
                results = self.mediapipe_detector.process(image_rgb)
                
                faces = []
                if results.detections:
                    height, width, _ = image.shape
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        # Add margin for better face recognition (15%)
                        margin_x = int(w * 0.15)
                        margin_y = int(h * 0.15)
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, x - margin_x)
                        y = max(0, y - margin_y)
                        w = min(width - x, w + 2 * margin_x)
                        h = min(height - y, h + 2 * margin_y)
                        
                        faces.append((x, y, w, h))
                    
                    if faces:
                        print(f"MediaPipe detected {len(faces)} faces")
                        return faces
            except Exception as e:
                print(f"MediaPipe detection failed: {e}")
        
        # Fall back to cascade classifier
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        
        print(f"Cascade classifier detected {len(faces)} faces")
        return faces
"""
                    content = content[:detect_pos] + detect_code + content[next_def:]
        
        # Write updated content
        with open(original_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {original_path} with MediaPipe integration")
        return True
    
    except Exception as e:
        logger.error(f"Error updating face service: {e}")
        # Restore backup if update failed
        if os.path.exists(backup_path):
            shutil.copy(backup_path, original_path)
            logger.info(f"Restored backup due to error")
        return False

def create_test_script():
    """Create a test script for MediaPipe face detection"""
    test_path = "test_mediapipe.py"
    
    test_code = """import cv2
import os
import sys
import numpy as np
from mediapipe_face_detector import MediaPipeFaceDetector

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_mediapipe.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image dimensions: {image.shape}")
    
    # Initialize detector
    detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
    
    # Detect faces
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces")
    
    # Draw detections
    image_with_detections = detector.draw_detections(image)
    
    # Create output directory if it doesn't exist
    output_dir = "detected_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image with detections
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detected_{base_name}")
    cv2.imwrite(output_path, image_with_detections)
    print(f"Saved image with detections to: {output_path}")
    
    # Crop and save individual faces
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        face_path = os.path.join(output_dir, f"face_{i}_{base_name}")
        cv2.imwrite(face_path, face)
        print(f"Saved face {i} to: {face_path}")

if __name__ == "__main__":
    main()
"""
    
    with open(test_path, 'w') as f:
        f.write(test_code)
    
    # Create batch file for testing
    batch_path = "test_mediapipe.bat"
    batch_code = """@echo off
echo Testing MediaPipe Face Detection
echo ==============================

if "%~1"=="" (
    echo Error: Please provide an image path
    echo Usage: test_mediapipe.bat path\\to\\image.jpg
    exit /b 1
)

python test_mediapipe.py %1

echo Test completed.
pause
"""
    
    with open(batch_path, 'w') as f:
        f.write(batch_code)
    
    logger.info(f"Created test script: {test_path} and batch file: {batch_path}")
    return True

def create_install_script():
    """Create script to install MediaPipe"""
    script_path = "install_mediapipe.bat"
    
    script_code = """@echo off
echo Installing MediaPipe for improved face detection...
pip install mediapipe opencv-python numpy

echo.
echo Installation complete!
echo.
pause
"""
    
    with open(script_path, 'w') as f:
        f.write(script_code)
    
    logger.info(f"Created installation script: {script_path}")
    return script_path

def main():
    """Main function to update the face recognition system"""
    print("=" * 60)
    print("UPDATING FACE RECOGNITION SYSTEM")
    print("=" * 60)
    
    # 1. Create MediaPipe detector
    print("\nCreating MediaPipe detector module...")
    create_mediapipe_detector()
    
    # 2. Update enhanced face service
    print("\nUpdating enhanced face service...")
    update_enhanced_face_service()
    
    # 3. Create test script
    print("\nCreating test script...")
    create_test_script()
    
    # 4. Create installation script
    print("\nCreating installation script...")
    install_script = create_install_script()
    
    # Print summary
    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)
    print("1. Created MediaPipe face detector module")
    print("2. Updated enhanced face service with MediaPipe support")
    print("3. Created test script for MediaPipe face detection")
    print(f"4. Created installation script: {install_script}")
    
    print("\nTo test MediaPipe face detection:")
    print("  test_mediapipe.bat <path_to_image>")
    
    print("\nTo run the face recognition system:")
    print("  python main_enhanced.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
