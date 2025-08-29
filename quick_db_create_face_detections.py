from dotenv import load_dotenv
import os
import pymysql

load_dotenv()
DB=os.getenv("MYSQL_DATABASE","face_recognition_db")

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST","localhost"),
    user=os.getenv("MYSQL_USER","root"),
    password=os.getenv("MYSQL_PASSWORD",""),
    database=DB,
    port=int(os.getenv("MYSQL_PORT","3306")),
    autocommit=True,
)
cur = conn.cursor()

# Create table if missing
cur.execute("""
CREATE TABLE IF NOT EXISTS face_detections (
  id INT AUTO_INCREMENT PRIMARY KEY,
  image_name VARCHAR(255) NOT NULL,
  person_id VARCHAR(64) NULL,
  age_range VARCHAR(32) NULL,
  gender VARCHAR(16) NULL,
  quality_score FLOAT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_image_name (image_name),
  KEY idx_person_id (person_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")

# Ensure required columns exist (in case table existed with a different schema)
def ensure_col(name, ddl):
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA=%s AND TABLE_NAME='face_detections' AND COLUMN_NAME=%s
    """, (DB, name))
    if cur.fetchone()[0] == 0:
        cur.execute(f"ALTER TABLE face_detections ADD COLUMN {ddl}")
        print(f"[OK] Added column {name}")

ensure_col("image_name", "image_name VARCHAR(255) NOT NULL")
ensure_col("person_id", "person_id VARCHAR(64) NULL")
ensure_col("age_range", "age_range VARCHAR(32) NULL")
ensure_col("gender", "gender VARCHAR(16) NULL")
ensure_col("quality_score", "quality_score FLOAT NULL")
ensure_col("created_at", "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")

print("[OK] face_detections ready")
cur.close()
conn.close()