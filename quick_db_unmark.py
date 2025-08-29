import os, sys
from dotenv import load_dotenv
load_dotenv()

def connect():
    import pymysql
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST","localhost"),
        user=os.getenv("MYSQL_USER","root"),
        password=os.getenv("MYSQL_PASSWORD",""),
        database=os.getenv("MYSQL_DATABASE","face_recognition_db"),
        port=int(os.getenv("MYSQL_PORT","3306")),
        autocommit=True
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python quick_db_unmark.py <image_name>")
        sys.exit(1)
    name = sys.argv[1]
    conn = connect()
    cur = conn.cursor()
    n = cur.execute("DELETE FROM processed_images WHERE image_name=%s", (name,))
    print(f"deleted processed_images rows: {n}")
    cur.close()
    conn.close()