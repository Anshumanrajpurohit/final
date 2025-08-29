import os, argparse, sys
from typing import Set

def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

def connect():
    host = os.getenv("MYSQL_HOST", "localhost")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DATABASE", "face_recognition_db")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    try:
        import pymysql
        conn = pymysql.connect(host=host, user=user, password=password, database=db, port=port, autocommit=True)
        conn._db_name = db  # attach for info_schema lookup
        return conn, "pymysql"
    except Exception:
        import mysql.connector as mc
        conn = mc.connect(host=host, user=user, password=password, database=db, port=port)
        conn.autocommit = True
        conn._db_name = db
        return conn, "mysql-connector"

def get_table_columns(conn, table) -> Set[str]:
    db = getattr(conn, "_db_name", os.getenv("MYSQL_DATABASE", "face_recognition_db"))
    cur = conn.cursor()
    cur.execute("""
        SELECT COLUMN_NAME FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
    """, (db, table))
    cols = {r[0] for r in cur.fetchall()}
    try: cur.close()
    except: pass
    return cols

def select_existing(cols: Set[str], wanted):
    return [c for c in wanted if c in cols]

def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Exact image filename (e.g. 'download (2).jpg')")
    args = ap.parse_args()

    conn, drv = connect()
    print(f"[OK] Connected via {drv}")

    # processed_images
    if args.image:
        p_cols = get_table_columns(conn, "processed_images")
        base = select_existing(p_cols, ["id","image_name","face_count"])
        optional = select_existing(p_cols, ["created_at","created_time","created","timestamp","processed_at","inserted_at"])
        fields = base + optional
        if not fields:
            print("[processed_images] table exists but no known columns found.")
        else:
            print(f"\n[processed_images] for {args.image}")
            sql = f"SELECT {', '.join(fields)} FROM processed_images WHERE image_name=%s ORDER BY id DESC LIMIT 5"
            cur = conn.cursor()
            cur.execute(sql, (args.image,))
            for r in cur.fetchall():
                print(r)
            try: cur.close()
            except: pass

    # face_detections
    fd_cols = get_table_columns(conn, "face_detections")
    fd_base = select_existing(fd_cols, ["id","person_id"])
    fd_opt = select_existing(fd_cols, ["image_name","age_range","gender","quality_score","created_at","timestamp"])
    fd_fields = fd_base + fd_opt
    if fd_fields:
        print("\n[face_detections] last 10")
        sql = f"SELECT {', '.join(fd_fields)} FROM face_detections ORDER BY id DESC LIMIT 10"
        cur = conn.cursor(); cur.execute(sql)
        for r in cur.fetchall(): print(r)
        try: cur.close()
        except: pass
    else:
        print("\n[face_detections] table missing or no known columns.")

    # unique_persons
    up_cols = get_table_columns(conn, "unique_persons")
    up_fields = select_existing(up_cols, ["id","created_at","timestamp"])
    if up_fields:
        print("\n[unique_persons] last 5")
        sql = f"SELECT {', '.join(up_fields)} FROM unique_persons ORDER BY id DESC LIMIT 5"
        cur = conn.cursor(); cur.execute(sql)
        for r in cur.fetchall(): print(r)
        try: cur.close()
        except: pass

    # person_embeddings
    pe_cols = get_table_columns(conn, "person_embeddings")
    pe_fields = select_existing(pe_cols, ["id","person_id","created_at","timestamp"])
    if pe_fields:
        print("\n[person_embeddings] last 5")
        sql = f"SELECT {', '.join(pe_fields)} FROM person_embeddings ORDER BY id DESC LIMIT 5"
        cur = conn.cursor(); cur.execute(sql)
        for r in cur.fetchall(): print(r)
        try: cur.close()
        except: pass

if __name__ == "__main__":
    main()