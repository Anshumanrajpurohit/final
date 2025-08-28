"""Simple helper to import SQL schema files using env credentials.

This script is optional: it requires `mysql-connector-python` to be installed.
It will read DB connection values from .env (if python-dotenv is installed) or from environment variables.
"""
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return

try:
    from mysql.connector import connect, Error
except ImportError:
    connect = None
    Error = Exception

load_dotenv()

DB_HOST = os.getenv('MYSQL_HOST', 'localhost')
DB_USER = os.getenv('MYSQL_USER', 'root')
DB_PASS = os.getenv('MYSQL_PASSWORD', '')
DB_NAME = os.getenv('MYSQL_DATABASE', 'face_recognition_db')
DB_PORT = int(os.getenv('MYSQL_PORT', 3306))

SQL_FILES = [
    os.path.join('utils', 'database_schema.sql'),
    os.path.join('utils', 'update_schema_age_gender.sql')
]


def main():
    if connect is None:
        print("mysql-connector-python is not installed. Install it with: pip install mysql-connector-python")
        sys.exit(1)

    try:
        conn = connect(host=DB_HOST, user=DB_USER, password=DB_PASS, port=DB_PORT)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.execute(f"USE {DB_NAME}")

        for sql_file in SQL_FILES:
            if not os.path.exists(sql_file):
                print(f"SQL file not found: {sql_file}")
                continue
            print(f"Importing {sql_file}...")
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql = f.read()
            statements = [s.strip() for s in sql.split(';') if s.strip()]
            for stmt in statements:
                try:
                    cursor.execute(stmt)
                except Error as e:
                    if 'already exists' in str(e).lower():
                        print(f"Skipped (exists): {stmt[:60]}")
                    else:
                        print(f"Error executing statement: {e}")
        conn.commit()
        print("✅ Schema import completed.")
    except Error as e:
        print(f"Database error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
