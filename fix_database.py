#!/usr/bin/env python3
"""
Fix Database Tables Script
Creates missing tables for the face recognition system
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

def create_missing_tables():
    """Create missing database tables"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'face_recognition_db'),
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_unicode_ci'
    }
    
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        print("✅ Connected to database")
        
        # Create person_embeddings table
        create_person_embeddings = """
        CREATE TABLE IF NOT EXISTS person_embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            person_id VARCHAR(36) NOT NULL,
            embedding LONGTEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES unique_persons(person_id) ON DELETE CASCADE,
            INDEX idx_person_id (person_id)
        )
        """
        
        # Create image_metadata table
        create_image_metadata = """
        CREATE TABLE IF NOT EXISTS image_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_url TEXT NOT NULL,
            filename VARCHAR(255) NOT NULL,
            file_size INT NULL,
            image_width INT NULL,
            image_height INT NULL,
            faces_detected INT DEFAULT 0,
            processed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP NULL,
            INDEX idx_filename (filename),
            INDEX idx_processed (processed),
            INDEX idx_created_at (created_at)
        )
        """
        
        # Create system_config table
        create_system_config = """
        CREATE TABLE IF NOT EXISTS system_config (
            config_key VARCHAR(50) PRIMARY KEY,
            config_value TEXT NOT NULL,
            description TEXT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        
        # Execute table creation
        tables_to_create = [
            ("person_embeddings", create_person_embeddings),
            ("image_metadata", create_image_metadata),
            ("system_config", create_system_config)
        ]
        
        for table_name, create_sql in tables_to_create:
            try:
                cursor.execute(create_sql)
                print(f"✅ Created table: {table_name}")
            except Error as e:
                print(f"⚠️ Warning creating {table_name}: {e}")
        
        # Insert default configuration
        insert_config = """
        INSERT IGNORE INTO system_config (config_key, config_value, description) VALUES
        ('face_threshold', '0.6', 'Face recognition similarity threshold'),
        ('max_batch_size', '10', 'Maximum images to process per batch'),
        ('processing_interval', '5', 'Processing interval in seconds'),
        ('temp_file_retention_hours', '24', 'How long to keep temp files'),
        ('enable_debug_logging', 'true', 'Enable detailed debug logging')
        """
        
        try:
            cursor.execute(insert_config)
            print("✅ Inserted default configuration")
        except Error as e:
            print(f"⚠️ Warning inserting config: {e}")
        
        connection.commit()
        
        # Verify all tables exist
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = [
            'unique_persons',
            'person_embeddings', 
            'temp_faces',
            'processing_logs',
            'image_metadata',
            'system_config'
        ]
        
        print("\n📊 Final Database Tables Status:")
        all_tables_exist = True
        for table in required_tables:
            if table in tables:
                print(f"✅ {table}")
            else:
                print(f"❌ {table} - MISSING")
                all_tables_exist = False
        
        if all_tables_exist:
            print("\n🎉 All database tables are now ready!")
        else:
            print("\n⚠️ Some tables are still missing")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    print("🔧 Fixing missing database tables...")
    create_missing_tables()
    print("✅ Database fix complete!")
