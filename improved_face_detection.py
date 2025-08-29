"""
Enhanced Face Detection Accuracy Improvement

This script improves face detection accuracy by:
1. Loading multiple face detection models
2. Implementing a confidence-based approach
3. Ensuring higher-quality face detections
4. Testing various detection parameters for optimal results
"""
import os
import sys
import cv2
import logging
import numpy as np
from PIL import Image
import uuid
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    def __init__(self, confidence_threshold=0.6, use_all_models=True):
        """
        Initialize face detector with multiple detection models
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            use_all_models: Use all available models for maximum accuracy
        """
        self.confidence_threshold = confidence_threshold
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 1. Haar Cascade Classifier (Fast but less accurate)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 2. DNN Face Detector (More accurate)
        self.use_dnn = False
        self.dnn_detector = None
        model_path = os.path.join("models", "opencv_face_detector.caffemodel")
        config_path = os.path.join("models", "opencv_face_detector.prototxt")
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            try:
                self.dnn_detector = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self.use_dnn = True
                logger.info("DNN face detector loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load DNN detector: {e}")
        
        # 3. Check for DLib if available (Most accurate for some cases)
        self.use_dlib = False
        try:
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.use_dlib = True
            logger.info("DLib face detector loaded successfully")
        except ImportError:
            logger.warning("DLib not available. Install with: pip install dlib")
        
        # Use all models for maximum accuracy if requested
        self.use_all_models = use_all_models
        
        logger.info(f"Enhanced Face Detector initialized with:")
        logger.info(f"- Confidence threshold: {self.confidence_threshold}")
        logger.info(f"- Haar Cascade: Available")
        logger.info(f"- DNN: {'Available' if self.use_dnn else 'Not available'}")
        logger.info(f"- DLib: {'Available' if self.use_dlib else 'Not available'}")
        logger.info(f"- Using all models: {self.use_all_models}")

    def detect_faces(self, image_path):
        """
        Detect faces in an image using multiple detection methods
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected face coordinates (x, y, w, h)
        """
        # Load image
        try:
            # Try PIL first (better format handling)
            try:
                pil_img = Image.open(image_path).convert('RGB')
                rgb_image = np.array(pil_img)
                # Convert back to BGR for OpenCV operations
                image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"PIL loading failed: {e}")
                # Fall back to OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image at {image_path}")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return []
        
        height, width = image.shape[:2]
        all_faces = []
        
        # 1. Try DNN detection (most accurate)
        if self.use_dnn:
            logger.info("Detecting faces with DNN model...")
            faces_dnn = self._detect_faces_dnn(image)
            all_faces.extend([(x, y, w, h, 0.9) for x, y, w, h in faces_dnn])  # High confidence
            logger.info(f"DNN found {len(faces_dnn)} faces")
        
        # 2. Try DLib detection
        if self.use_dlib and (not all_faces or self.use_all_models):
            logger.info("Detecting faces with DLib...")
            faces_dlib = self._detect_faces_dlib(rgb_image)
            all_faces.extend([(x, y, w, h, 0.85) for x, y, w, h in faces_dlib])  # Good confidence
            logger.info(f"DLib found {len(faces_dlib)} faces")
        
        # 3. Try Haar Cascade (fallback)
        if not all_faces or self.use_all_models:
            logger.info("Detecting faces with Haar Cascade...")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_haar = self._detect_faces_haar(gray)
            all_faces.extend([(x, y, w, h, 0.7) for x, y, w, h in faces_haar])  # Lower confidence
            logger.info(f"Haar found {len(faces_haar)} faces")
        
        # Merge and filter detections
        merged_faces = self._merge_and_filter_detections(all_faces, width, height)
        logger.info(f"After merging and filtering: {len(merged_faces)} unique faces")
        
        # Save detected faces for verification
        if merged_faces:
            self._save_detected_faces(image, merged_faces, os.path.basename(image_path))
        
        return merged_faces

    def _detect_faces_dnn(self, image):
        """Detect faces using OpenCV DNN model"""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Skip invalid faces
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Store in (x, y, w, h) format
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces

    def _detect_faces_haar(self, gray_image):
        """Detect faces using Haar Cascade"""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for x, y, w, h in faces]
    
    def _detect_faces_dlib(self, rgb_image):
        """Detect faces using DLib"""
        import dlib
        # Convert faces to (x, y, w, h) format
        faces = []
        dlib_faces = self.dlib_detector(rgb_image, 1)  # 1 = upsample once for better detection
        
        for face in dlib_faces:
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            faces.append((x, y, w, h))
        
        return faces

    def _merge_and_filter_detections(self, faces, img_width, img_height):
        """Merge overlapping detections and filter out low confidence ones"""
        if not faces:
            return []
        
        # Sort by confidence (highest first)
        faces = sorted(faces, key=lambda x: x[4], reverse=True)
        
        # Initialize result list with the highest confidence face
        result = [faces[0]]
        
        # Calculate IoU (Intersection over Union) and filter
        for face in faces[1:]:
            x1, y1, w1, h1, conf = face
            
            # Check if this face significantly overlaps with any existing face
            overlapping = False
            for idx, (x2, y2, w2, h2, _) in enumerate(result):
                # Calculate IoU
                intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection_area = intersection_x * intersection_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - intersection_area
                
                iou = intersection_area / union_area if union_area > 0 else 0
                
                if iou > 0.3:  # Overlapping faces
                    overlapping = True
                    # If current face has higher confidence, replace the existing one
                    if conf > result[idx][4]:
                        result[idx] = face
                    break
            
            if not overlapping:
                # Add new unique face
                result.append(face)
        
        # Final filtering: ensure faces are within image bounds and reasonable size
        filtered_result = []
        min_face_size = 20  # Minimum face size (pixels)
        
        for x, y, w, h, conf in result:
            # Clamp coordinates to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Filter out tiny faces (likely false positives)
            if w >= min_face_size and h >= min_face_size and conf >= self.confidence_threshold:
                filtered_result.append((x, y, w, h, conf))
        
        return filtered_result

    def _save_detected_faces(self, image, faces, original_filename):
        """Save detected faces for verification"""
        # Create a copy of the image
        img_with_faces = image.copy()
        
        # Draw rectangles around faces
        for x, y, w, h, conf in faces:
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(img_with_faces, f"{conf:.2f}", (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the annotated image
        output_path = os.path.join(self.temp_dir, f"detected_faces_{os.path.splitext(original_filename)[0]}.jpg")
        cv2.imwrite(output_path, img_with_faces)
        logger.info(f"Saved detected faces to {output_path}")
        
        # Also save individual face crops
        for i, (x, y, w, h, _) in enumerate(faces):
            face_crop = image[y:y+h, x:x+w]
            face_path = os.path.join(self.temp_dir, f"face_{os.path.splitext(original_filename)[0]}_{i}.jpg")
            cv2.imwrite(face_path, face_crop)
            logger.info(f"Saved face crop to {face_path}")

def test_face_detection(image_path, confidence=0.6, use_all_models=True):
    """Test face detection on a single image"""
    detector = EnhancedFaceDetector(confidence_threshold=confidence, use_all_models=use_all_models)
    
    logger.info(f"Testing face detection on {image_path}")
    faces = detector.detect_faces(image_path)
    
    logger.info(f"Detected {len(faces)} faces in {image_path}")
    for i, (x, y, w, h, conf) in enumerate(faces):
        logger.info(f"Face {i+1}: Position=({x}, {y}), Size={w}x{h}, Confidence={conf:.2f}")
    
    return faces

def main():
    parser = argparse.ArgumentParser(description="Enhanced Face Detection Tool")
    parser.add_argument("--image", "-i", type=str, help="Path to the image file")
    parser.add_argument("--confidence", "-c", type=float, default=0.6, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--all-models", "-a", action="store_true", help="Use all available detection models")
    
    args = parser.parse_args()
    
    if not args.image:
        logger.error("Please provide an image path with --image")
        return
    
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return
    
    faces = test_face_detection(args.image, args.confidence, args.all_models)
    
    if not faces:
        logger.warning("No faces detected. Try adjusting parameters:")
        logger.warning("1. Lower the confidence threshold (--confidence)")
        logger.warning("2. Use all models (--all-models)")
        logger.warning("3. Try a clearer image with better lighting")

if __name__ == "__main__":
    main()
