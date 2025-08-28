#!/usr/bin/env python3
"""
Database Setup Script for Face Recognition System
Creates all necessary tables and initializes the database
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Set up the database schema"""
    
    # Database connection parameters
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'face_recognition_db'),
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_unicode_ci'
    }
    
    try:
        # Connect to database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        print("✅ Connected to database successfully")
        
        # Read and execute schema file
        schema_file = "utils/database_schema.sql"
        if os.path.exists(schema_file):
            with open(schema_file, 'r', encoding='utf-8') as file:
                schema_sql = file.read()
            
            # Split by semicolon and execute each statement
            statements = schema_sql.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement)
                        print(f"✅ Executed: {statement[:50]}...")
                    except Error as e:
                        if "already exists" not in str(e).lower():
                            print(f"⚠️ Warning: {e}")
            
            connection.commit()
            print("✅ Database schema setup completed")
            
        else:
            print("❌ Schema file not found")
            
        # Verify tables exist
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
        
        print("\n📊 Database Tables Status:")
        for table in required_tables:
            if table in tables:
                print(f"✅ {table}")
            else:
                print(f"❌ {table} - MISSING")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"❌ Database connection error: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Face Recognition Database...")
    setup_database()
    print("✅ Database setup complete!")
