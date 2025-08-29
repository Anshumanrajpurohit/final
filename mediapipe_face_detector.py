import cv2
import mediapipe as mp
import numpy as np

class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        """
        Initialize MediaPipe Face Detector
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection
            model_selection: 0 for short-range, 1 for full-range detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence
        )
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of face locations as (x, y, w, h) tuples
        """
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
        """
        Draw face detection boxes on image
        
        Args:
            image: Input image
            
        Returns:
            Image with detection boxes drawn
        """
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
