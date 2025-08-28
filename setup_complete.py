"""
Complete Database Setup Script
Handles MySQL installation check, database creation, user setup, and schema initialization
"""
import subprocess
import sys
import os
try:
    from mysql.connector import connect, Error
except ImportError:
    connect = None
    Error = Exception
    # leave a flag; functions will check and provide instructions
import logging
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        # noop if python-dotenv not installed; will rely on environment variables
        return
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_db_connection(host, user, password, port):
    """Check if we can connect to the target MySQL server."""
    try:
        connect(host=host, user=user, password=password, port=port)
        logger.info("✅ Connected to MySQL server")
        return True
    except Error as e:
        logger.error(f"❌ Cannot connect to MySQL server: {e}")
        return False

def setup_database():
    """Set up database, user, and schema"""
    try:
        # Ensure mysql connector is available
        if connect is None:
            logger.error("mysql-connector-python is not installed in this environment.")
            logger.error("You can either install it (pip install mysql-connector-python) or import the SQL files via phpMyAdmin:")
            logger.error(" - Open phpMyAdmin -> Select/Create database -> Import -> choose 'utils/database_schema.sql' and 'utils/update_schema_age_gender.sql'")
            return False

        # Load environment variables
        load_dotenv()
        
        # Database configuration
        DB_NAME = os.getenv('MYSQL_DATABASE', 'face_recognition_db')
        DB_USER = os.getenv('MYSQL_USER', 'face_user')
        DB_PASS = os.getenv('MYSQL_PASSWORD', 'face_password')
        
        # Connect to the target MySQL server (use credentials from .env)
        DB_HOST = os.getenv('MYSQL_HOST', 'localhost')
        DB_PORT = int(os.getenv('MYSQL_PORT', 3306))

        if not check_db_connection(DB_HOST, os.getenv('MYSQL_USER', 'root'), os.getenv('MYSQL_PASSWORD', ''), DB_PORT):
            logger.error("Cannot proceed: update your .env with working MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD and ensure phpMyAdmin/XAMPP or your MySQL server is reachable.")
            logger.error("Alternatively, import the SQL files in 'utils/' using phpMyAdmin if you prefer a GUI.")
            return False

        # Connect as admin/root or provided user to create database/user if permitted
        connection = connect(
            host=DB_HOST,
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
            port=DB_PORT
        )
        cursor = connection.cursor()
        
        # Create database
        logger.info(f"Creating database {DB_NAME}...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        
        # Create user and grant privileges
        logger.info(f"Setting up user {DB_USER}...")
        cursor.execute(f"CREATE USER IF NOT EXISTS '{DB_USER}'@'localhost' IDENTIFIED BY '{DB_PASS}'")
        cursor.execute(f"GRANT ALL PRIVILEGES ON {DB_NAME}.* TO '{DB_USER}'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        # Close root connection
        cursor.close()
        connection.close()
        
        # Connect as new user (or same user) to run schema
        connection = connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()
        
        # Initialize schema
        logger.info("Initializing database schema...")
        schema_files = [
            "utils/database_schema.sql",
            "utils/update_schema_age_gender.sql"
        ]
        
        for schema_file in schema_files:
            if os.path.exists(schema_file):
                with open(schema_file, 'r', encoding='utf-8') as file:
                    schema_sql = file.read()
                
                # Execute each statement
                for statement in schema_sql.split(';'):
                    if statement.strip() and not statement.strip().startswith('--'):
                        try:
                            cursor.execute(statement)
                            logger.info(f"✅ Executed: {statement[:50]}...")
                        except Error as e:
                            if "already exists" not in str(e).lower():
                                logger.error(f"❌ Error executing: {statement[:50]}...")
                                logger.error(f"Error: {str(e)}")
        
        connection.commit()
        logger.info("✅ Database setup completed successfully!")
        
        # Update .env file with new credentials
        update_env_file(DB_USER, DB_PASS)
        
        return True
        
    except Error as e:
        logger.error(f"❌ Database setup failed: {str(e)}")
        return False

def update_env_file(db_user, db_pass):
    """Update .env file with new database credentials"""
    env_file = ".env"
    new_content = []
    updated = False
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as file:
            lines = file.readlines()
            
        for line in lines:
            if line.startswith('MYSQL_USER='):
                new_content.append(f'MYSQL_USER={db_user}\n')
                updated = True
            elif line.startswith('MYSQL_PASSWORD='):
                new_content.append(f'MYSQL_PASSWORD={db_pass}\n')
                updated = True
            else:
                new_content.append(line)
        
        if updated:
            with open(env_file, 'w') as file:
                file.writelines(new_content)
            logger.info("✅ Updated .env file with new credentials")

if __name__ == "__main__":
    logger.info("Starting database setup...")
    load_dotenv()

    # Attempt to set up the database using credentials from .env or instruct user to use phpMyAdmin
    if setup_database():
        logger.info("\n✨ Setup completed successfully!")
        logger.info("You can now run the main application.")
    else:
        logger.error("\n❌ Setup failed. Please check the errors above.")
