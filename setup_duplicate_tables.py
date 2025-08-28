#!/usr/bin/env python3
"""
Setup Duplicate Detection Tables
Creates new tables for duplicate detection and sleep mode functionality
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

def setup_duplicate_tables():
    """Set up duplicate detection and sleep mode tables"""
    
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
        
        # Create processed_images table
        create_processed_images = """
        CREATE TABLE IF NOT EXISTS processed_images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_hash VARCHAR(64) NOT NULL,
            image_name VARCHAR(255) NOT NULL,
            image_url TEXT NOT NULL,
            face_count INT DEFAULT 0,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_image_hash (image_hash),
            INDEX idx_image_hash (image_hash),
            INDEX idx_processed_at (processed_at),
            INDEX idx_image_name (image_name)
        )
        """
        
        # Create duplicate_events table
        create_duplicate_events = """
        CREATE TABLE IF NOT EXISTS duplicate_events (
            id INT AUTO_INCREMENT PRIMARY KEY,
            event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duplicate_type ENUM('image', 'face', 'batch') NOT NULL,
            duplicate_count INT DEFAULT 1,
            sleep_duration INT DEFAULT 300,
            description TEXT NULL,
            INDEX idx_event_time (event_time),
            INDEX idx_duplicate_type (duplicate_type)
        )
        """
        
        # Create sleep_mode_state table
        create_sleep_mode_state = """
        CREATE TABLE IF NOT EXISTS sleep_mode_state (
            id INT AUTO_INCREMENT PRIMARY KEY,
            is_sleeping BOOLEAN DEFAULT FALSE,
            sleep_start_time TIMESTAMP NULL,
            sleep_end_time TIMESTAMP NULL,
            sleep_duration INT DEFAULT 0,
            sleep_reason VARCHAR(100) NULL,
            wake_up_trigger VARCHAR(50) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_is_sleeping (is_sleeping),
            INDEX idx_sleep_start_time (sleep_start_time)
        )
        """
        
        # Execute table creation
        tables_to_create = [
            ("processed_images", create_processed_images),
            ("duplicate_events", create_duplicate_events),
            ("sleep_mode_state", create_sleep_mode_state)
        ]
        
        for table_name, create_sql in tables_to_create:
            try:
                cursor.execute(create_sql)
                print(f"✅ Created table: {table_name}")
            except Error as e:
                print(f"⚠️ Warning creating {table_name}: {e}")
        
        # Insert default sleep mode state
        insert_sleep_state = """
        INSERT IGNORE INTO sleep_mode_state (is_sleeping) VALUES (FALSE)
        """
        
        try:
            cursor.execute(insert_sleep_state)
            print("✅ Inserted default sleep mode state")
        except Error as e:
            print(f"⚠️ Warning inserting sleep state: {e}")
        
        # Update system_config with duplicate detection settings
        insert_config = """
        INSERT IGNORE INTO system_config (config_key, config_value, description) VALUES
        ('duplicate_sleep_duration', '300', 'Sleep duration in seconds when duplicates detected'),
        ('max_duplicate_threshold', '3', 'Maximum duplicates before sleep mode'),
        ('image_hash_cache_duration', '86400', 'Image hash cache duration in seconds (24 hours)'),
        ('sleep_mode_backoff_multiplier', '1.5', 'Sleep duration multiplier for repeated duplicates'),
        ('enable_sleep_mode', 'true', 'Enable sleep mode for duplicate detection'),
        ('min_sleep_duration', '60', 'Minimum sleep duration in seconds'),
        ('max_sleep_duration', '3600', 'Maximum sleep duration in seconds')
        """
        
        try:
            cursor.execute(insert_config)
            print("✅ Inserted duplicate detection configuration")
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
            'system_config',
            'processed_images',
            'duplicate_events',
            'sleep_mode_state'
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
            print("\n🎉 All duplicate detection tables are now ready!")
        else:
            print("\n⚠️ Some tables are still missing")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    print("🔧 Setting up duplicate detection tables...")
    setup_duplicate_tables()
    print("✅ Duplicate detection setup complete!")
