import cv2
import sys
import os

def inspect_image(path):
    # Check if file exists
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return
    
    # Try to read the image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(f"Path: {path}")
    print(f"File exists: {os.path.exists(path)}")
    print(f"Image loaded: {img is not None}")
    
    if img is None:
        print("ERROR: Could not load image (null)")
        return
    
    # Print image properties
    print(f"Shape: {img.shape}")
    print(f"Dimensions: {img.ndim}")
    print(f"Data type: {img.dtype}")
    print(f"Min/Max values: {img.min()}/{img.max()}")
    
    # Try saving a copy as JPG (converts to 8-bit RGB)
    fixed_path = path + ".fixed.jpg"
    try:
        # If 4-channel, convert to BGR
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # If grayscale, convert to BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        # Ensure 8-bit
        if img.dtype != 'uint8':
            img = cv2.convertScaleAbs(img)
            
        # Save the fixed image
        cv2.imwrite(fixed_path, img)
        print(f"Fixed image saved to: {fixed_path}")
    except Exception as e:
        print(f"Error fixing image: {e}")

if __name__ == "__main__":
    # Use command line argument or default path
    path = "temp_images/WhatsApp Image 2025-08-27 at 10.13.19 PM.jpeg"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    inspect_image(path)