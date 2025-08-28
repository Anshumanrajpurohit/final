import os
from datetime import datetime
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return

try:
    from mysql.connector import connect, Error
except ImportError:
    print("mysql-connector-python is required. Install with: pip install mysql-connector-python")
    raise

load_dotenv()
DB_HOST = os.getenv('MYSQL_HOST', 'localhost')
DB_USER = os.getenv('MYSQL_USER', 'root')
DB_PASS = os.getenv('MYSQL_PASSWORD', '')
DB_NAME = os.getenv('MYSQL_DATABASE', 'face_recognition_db')
DB_PORT = int(os.getenv('MYSQL_PORT', 3306))

TABLE_TO_RENAME = 'user_attributes'

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"backup_{TABLE_TO_RENAME}_{ts}"
    try:
        with connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT) as conn:
            cur = conn.cursor()
            # Check if table exists
            cur.execute("SHOW TABLES LIKE %s", (TABLE_TO_RENAME,))
            if not cur.fetchone():
                print(f"Table '{TABLE_TO_RENAME}' not found in database '{DB_NAME}'. Nothing to do.")
                return
            # Rename table
            print(f"Renaming '{TABLE_TO_RENAME}' -> '{backup_name}'")
            cur.execute(f"RENAME TABLE `{TABLE_TO_RENAME}` TO `{backup_name}`")
            conn.commit()
            print("Rename successful.")
            # Show current tables
            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
            print("Current tables:")
            for t in tables:
                print(f" - {t}")
    except Error as e:
        print(f"Database error: {e}")

if __name__ == '__main__':
    main()
