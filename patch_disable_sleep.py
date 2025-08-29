"""
Patch for main_enhanced.py to disable sleep state functionality

This script applies a patch to the EnhancedFaceRecognitionSystem class to 
disable the sleep state functionality and ensure real-time detection without delays.

Apply this patch using:
python patch_disable_sleep.py
"""
import os
import re
import sys
import shutil
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def patch_main_enhanced():
    """Apply patches to main_enhanced.py to disable sleep state"""
    file_path = "main_enhanced.py"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    # Create backup
    if not backup_file(file_path):
        return False
    
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Patch 1: Disable sleep mode check in process_batch_enhanced
        # Replace the sleep check code with a pass-through
        pattern1 = r"# Check if system is in sleep mode\s+if self\.duplicate_detector\.is_sleeping\(\):[\s\S]+?sleep_status\[\'remaining_seconds\'\]}s remaining\"\)\s+return"
        replacement1 = "# Sleep mode is disabled for real-time detection\n        # No delay will be introduced here"
        
        content = re.sub(pattern1, replacement1, content)
        
        # Patch 2: Disable sleep mode in initialization
        pattern2 = r"(sleep_config = SleepModeConfig\([^)]+)enable_sleep_mode=os\.getenv\('ENABLE_SLEEP_MODE', 'true'\)\.lower\(\) == 'true'"
        replacement2 = r"\1enable_sleep_mode=False"
        
        content = re.sub(pattern2, replacement2, content)
        
        # Patch 3: Remove duplicate detection sleep logic
        pattern3 = r"# Enter sleep mode if threshold exceeded\s+if duplicate_count >= self\.duplicate_detector\.config\.max_duplicate_threshold:[\s\S]+?batch_stats\['status'\] = 'sleep_mode'\s+return"
        replacement3 = "# Sleep mode is disabled - continuing with processing\n                pass"
        
        content = re.sub(pattern3, replacement3, content)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {file_path} to disable sleep state functionality")
        return True
    
    except Exception as e:
        print(f"Error patching file: {e}")
        return False

def patch_duplicate_detector():
    """Apply patch to disable sleep logic in duplicate_detector.py"""
    # Try to find the duplicate_detector.py file
    possible_paths = [
        os.path.join("utils", "duplicate_detector.py"),
        "duplicate_detector.py"
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        print("Warning: duplicate_detector.py not found, skipping this patch")
        return False
    
    # Create backup
    if not backup_file(file_path):
        return False
    
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Patch 1: Disable is_sleeping method
        pattern1 = r"def is_sleeping\(self\):[\s\S]+?return False"
        replacement1 = "def is_sleeping(self):\n        \"\"\"Always return False to disable sleep functionality\"\"\"\n        return False"
        
        content = re.sub(pattern1, replacement1, content)
        
        # Patch 2: Disable enter_sleep_mode method
        pattern2 = r"def enter_sleep_mode\(self, reason, duration=None\):[\s\S]+?return True"
        replacement2 = "def enter_sleep_mode(self, reason, duration=None):\n        \"\"\"Do nothing - sleep mode is disabled\"\"\"\n        return False"
        
        content = re.sub(pattern2, replacement2, content)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {file_path} to disable sleep state functionality")
        return True
    
    except Exception as e:
        print(f"Error patching file: {e}")
        return False

def main():
    print("=" * 60)
    print("Applying Patches to Disable Sleep State")
    print("=" * 60)
    
    success = True
    
    # Patch main_enhanced.py
    print("\nPatching main_enhanced.py...")
    if not patch_main_enhanced():
        success = False
    
    # Patch duplicate_detector.py if available
    print("\nPatching duplicate detector...")
    if not patch_duplicate_detector():
        print("Warning: Could not patch duplicate detector, but main file was patched")
    
    if success:
        print("\n" + "=" * 60)
        print("Patches Applied Successfully")
        print("=" * 60)
        print("\nThe system will now perform real-time detection without sleep state delays.")
        print("To apply these changes, restart the face recognition system:")
        print("python main_enhanced.py")
    else:
        print("\n" + "=" * 60)
        print("Patching Failed")
        print("=" * 60)
        print("\nPlease check the error messages above and try again.")

if __name__ == "__main__":
    main()
