"""
Remove Sleep State Functionality Script

This script disables sleep state functionality in the face recognition system
by modifying the system_config table and truncating the sleep_mode_state table.
"""
import os
import sys
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        return

try:
    from mysql.connector import connect, Error
except ImportError:
    print("Error: mysql-connector-python is not installed. Run: pip install mysql-connector-python")
    sys.exit(1)

# Load environment variables
load_dotenv()
DB_HOST = os.getenv('MYSQL_HOST', 'localhost')
DB_USER = os.getenv('MYSQL_USER', 'root')
DB_PASS = os.getenv('MYSQL_PASSWORD', '')
DB_NAME = os.getenv('MYSQL_DATABASE', 'face_recognition_db')
DB_PORT = int(os.getenv('MYSQL_PORT', 3306))

def connect_to_db():
    """Establish connection to the database"""
    try:
        conn = connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            port=DB_PORT
        )
        print(f"Connected to database: {DB_NAME}")
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def disable_sleep_mode(conn):
    """Disable sleep mode in system_config"""
    cursor = conn.cursor()
    
    try:
        # Update system_config to disable sleep mode
        cursor.execute("""
            UPDATE system_config 
            SET config_value = 'false' 
            WHERE config_key = 'enable_sleep_mode'
        """)
        
        # Set sleep duration to 0
        cursor.execute("""
            UPDATE system_config 
            SET config_value = '0' 
            WHERE config_key = 'duplicate_sleep_duration'
        """)
        
        # Set max duplicate threshold to a high value
        cursor.execute("""
            UPDATE system_config 
            SET config_value = '100' 
            WHERE config_key = 'max_duplicate_threshold'
        """)
        
        # Set sleep mode backoff multiplier to 0
        cursor.execute("""
            UPDATE system_config 
            SET config_value = '0' 
            WHERE config_key = 'sleep_mode_backoff_multiplier'
        """)
        
        # Set min sleep duration to 0
        cursor.execute("""
            UPDATE system_config 
            SET config_value = '0' 
            WHERE config_key = 'min_sleep_duration'
        """)
        
        # Set max sleep duration to 0
        cursor.execute("""
            UPDATE system_config 
            SET config_value = '0' 
            WHERE config_key = 'max_sleep_duration'
        """)
        
        # Commit changes
        conn.commit()
        print("Successfully disabled sleep mode in system_config")
        
    except Error as e:
        conn.rollback()
        print(f"Error updating system_config: {e}")
        return False
    
    finally:
        cursor.close()
    
    return True

def clear_sleep_mode_state(conn):
    """Clear the sleep_mode_state table"""
    cursor = conn.cursor()
    
    try:
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'sleep_mode_state'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Truncate the table
            cursor.execute("TRUNCATE TABLE sleep_mode_state")
            conn.commit()
            print("Successfully cleared sleep_mode_state table")
        else:
            print("sleep_mode_state table does not exist, no action needed")
        
    except Error as e:
        conn.rollback()
        print(f"Error clearing sleep_mode_state table: {e}")
        return False
    
    finally:
        cursor.close()
    
    return True

def main():
    print("=" * 60)
    print("Remove Sleep State Functionality")
    print("=" * 60)
    
    conn = connect_to_db()
    
    print("\nDisabling sleep mode...")
    disable_sleep_mode(conn)
    
    print("\nClearing sleep mode state...")
    clear_sleep_mode_state(conn)
    
    print("\nTruncating duplicate_events table...")
    cursor = conn.cursor()
    try:
        cursor.execute("TRUNCATE TABLE duplicate_events")
        conn.commit()
        print("Successfully truncated duplicate_events table")
    except Error as e:
        conn.rollback()
        print(f"Error truncating duplicate_events table: {e}")
    finally:
        cursor.close()
    
    # Print summary of settings
    print("\n" + "=" * 60)
    print("Current Sleep Mode Settings")
    print("=" * 60)
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT config_key, config_value FROM system_config WHERE config_key LIKE '%sleep%' OR config_key LIKE '%duplicate%'")
        settings = cursor.fetchall()
        
        for key, value in settings:
            print(f"{key}: {value}")
    except Error as e:
        print(f"Error retrieving settings: {e}")
    finally:
        cursor.close()
    
    # Close connection
    conn.close()
    print("\nDatabase connection closed.")
    print("\nSleep mode has been disabled. The system will now perform real-time detection without delays.")

if __name__ == "__main__":
    main()
