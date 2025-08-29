import cv2
import mediapipe as mp
import os
import sys
import argparse

def test_mediapipe_detection(image_path, save_dir="detected_faces"):
    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
        
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(image_rgb)
        
        # Draw face detections
        annotated_image = image.copy()
        if not results.detections:
            print("No faces detected")
            return False
            
        print(f"Detected {len(results.detections)} faces")
        
        # Draw and crop each detected face
        faces = []
        for i, detection in enumerate(results.detections):
            # Draw on image
            mp_drawing.draw_detection(annotated_image, detection)
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add margin (15%)
            margin_x = int(width * 0.15)
            margin_y = int(height * 0.15)
            
            # Ensure coordinates stay within image bounds
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            width = min(w - x, width + 2 * margin_x)
            height = min(h - y, height + 2 * margin_y)
            
            # Crop face
            face = image[y:y+height, x:x+width]
            
            # Save cropped face
            base_name = os.path.basename(image_path)
            face_path = os.path.join(save_dir, f"face_{i}_{base_name}")
            cv2.imwrite(face_path, face)
            print(f"Saved face {i} to {face_path}")
            
            faces.append((x, y, width, height))
        
        # Save annotated image
        annotated_path = os.path.join(save_dir, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Saved annotated image to {annotated_path}")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MediaPipe face detection")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--save-dir", default="detected_faces", help="Directory to save results")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image path does not exist: {args.image_path}")
        sys.exit(1)
        
    test_mediapipe_detection(args.image_path, args.save_dir)