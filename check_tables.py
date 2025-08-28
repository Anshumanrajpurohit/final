import os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return

try:
    from mysql.connector import connect, Error
except ImportError:
    print("mysql-connector-python is not installed. Run: pip install mysql-connector-python")
    raise

load_dotenv()
DB_HOST = os.getenv('MYSQL_HOST', 'localhost')
DB_USER = os.getenv('MYSQL_USER', 'root')
DB_PASS = os.getenv('MYSQL_PASSWORD', '')
DB_NAME = os.getenv('MYSQL_DATABASE', 'face_recognition_db')
DB_PORT = int(os.getenv('MYSQL_PORT', 3306))

try:
    with connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT) as conn:
        cur = conn.cursor()
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]
        print(f"Database: {DB_NAME}")
        if tables:
            print("Tables:")
            for t in tables:
                print(f" - {t}")
        else:
            print("No tables found in the database.")
except Error as e:
    print(f"Error connecting to database: {e}")
    raise
