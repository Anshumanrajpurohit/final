"""
Quick fixes for immediate issues
"""
import os
import subprocess
import sys

def install_mysql_if_missing():
    """Guide: Use phpMyAdmin/XAMPP for database management instead of installing MySQL server locally."""
    print("🗄️ Database setup guidance")
    print("=" * 30)

    print("This project assumes you will use phpMyAdmin (typically via XAMPP, WAMP, or a hosted LAMP stack) to host your MySQL databases.")
    print("Recommended options:")
    print("- Use XAMPP: https://www.apachefriends.org/ (includes phpMyAdmin)")
    print("- Use WAMP: https://www.wampserver.com/ (includes phpMyAdmin)")
    print("- Use a hosted phpMyAdmin / MySQL provider if you prefer not to run locally")

    print("Quick steps with XAMPP:")
    print("1. Install XAMPP and start the Apache & MySQL modules from the XAMPP Control Panel")
    print("2. Open phpMyAdmin at: http://localhost/phpmyadmin/")
    print("3. Create a database named 'face_recognition_db' (or update .env to match your database name)")
    print("4. Create the required tables by running the SQL scripts in 'utils/' using phpMyAdmin's SQL import")

    print("If your phpMyAdmin/MySQL is remote, update the .env variables: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_PORT")

def start_mysql_service():
    """Try to start MySQL service"""
    print("🚀 Attempting to Start MySQL...")
    print("=" * 30)
    
    try:
        # Try Windows service
        result = subprocess.run(['net', 'start', 'mysql'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ MySQL service started successfully!")
            return True
        else:
            print(f"❌ Failed to start MySQL service: {result.stderr}")
    except Exception as e:
        print(f"❌ Error starting MySQL: {e}")
    
    # Try XAMPP
    xampp_paths = [
        r"C:\xampp\mysql\bin\mysqld.exe",
        r"C:\Program Files\XAMPP\mysql\bin\mysqld.exe"
    ]
    
    for path in xampp_paths:
        if os.path.exists(path):
            print(f"💡 Found XAMPP MySQL at: {path}")
            print("   Start XAMPP Control Panel and start MySQL service")
            return False
    
    return False

def create_basic_env():
    """Create basic .env file"""
    print("📝 Creating Basic .env File...")
    print("=" * 30)
    
    if os.path.exists('.env'):
        print("⚠️ .env file already exists")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            return
    
    env_content = """# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=face_recognition_db
MYSQL_PORT=3306

# Supabase Configuration (REQUIRED - Update with your values)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_BUCKET_NAME=images

# Face Recognition Settings
FACE_THRESHOLD=0.6
MAX_BATCH_SIZE=10
PROCESSING_INTERVAL=5
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with default values")
    print("⚠️  IMPORTANT: Update Supabase credentials in .env file!")

def check_python_packages():
    """Check if required packages are installed"""
    print("📦 Checking Python Packages...")
    print("=" * 30)
    
    # Map of package names to their import names
    package_imports = {
        'face-recognition': 'face_recognition',
        'opencv-python': 'cv2', 
        'mysql-connector-python': 'mysql.connector',
        'supabase': 'supabase',
        'python-dotenv': 'dotenv',
        'pillow': 'PIL',
        'schedule': 'schedule',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def main():
    print("🔧 Quick Fix Tool")
    print("=" * 50)
    
    # Check packages first
    if not check_python_packages():
        print("\n❌ Install missing packages first, then run this script again")
        return
    
    # Create .env if missing
    if not os.path.exists('.env'):
        create_basic_env()
    
    # Try to start MySQL
    start_mysql_service()
    
    # Show MySQL installation guide if needed
    try:
        subprocess.run(['mysql', '--version'], capture_output=True, text=True, check=True)
        print("✅ MySQL client found")
    except:
        install_mysql_if_missing()
    
    # Create directories
    dirs = ['temp_images', 'logs']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Update .env file with your Supabase credentials")
    print("2. Ensure MySQL is running (check XAMPP or Windows services)")
    print("3. Run: python diagnosis_and_fix.py for detailed diagnosis")
    print("4. If everything is OK, run: python main.py")

if __name__ == "__main__":
    main()