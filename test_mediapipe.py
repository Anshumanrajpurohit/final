import cv2
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
