#!/usr/bin/env python3
"""
Script to add age and gender columns to the database schema
"""
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_schema():
    # Load environment variables
    load_dotenv()
    
    # Database connection parameters (using the same as in setup_database.py)
    db_config = {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'face_recognition_db')
    }
    
    connection = None
    try:
        # Connect to database
        logger.info("Connecting to database...")
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Read the SQL file
        logger.info("Reading SQL update script...")
        with open('utils/update_schema_age_gender.sql', 'r') as file:
            sql_script = file.read()
        
        # Execute each command separately (split by semicolon)
        for command in sql_script.split(';'):
            if command.strip():
                logger.info(f"Executing: {command.strip()}")
                cursor.execute(command.strip())
        
        # Commit the changes
        connection.commit()
        logger.info("Successfully updated database schema with age and gender columns!")
        
    except Error as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    try:
        update_schema()
    except Exception as e:
        logger.error(f"Schema update failed: {str(e)}")
        exit(1)
