"""
Database Cleanup Script

This script performs the following operations:
1. Removes unwanted tables (if requested)
2. Cleans up duplicate entries in person_embeddings
3. Removes images with zero face count from processed_images
4. Provides a summary of changes
"""
import os
import sys
import json
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        print("Warning: python-dotenv not installed. Environment variables must be set manually.")
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

def list_tables(conn):
    """List all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tables

def count_records(conn, table_name):
    """Count records in a table"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    return count

def delete_unwanted_tables(conn, tables_to_delete):
    """Delete unwanted tables if they exist"""
    cursor = conn.cursor()
    results = []
    
    all_tables = list_tables(conn)
    for table in tables_to_delete:
        if table in all_tables:
            try:
                # First check if there are any foreign key constraints
                cursor.execute(f"""
                    SELECT TABLE_NAME, CONSTRAINT_NAME
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE REFERENCED_TABLE_NAME = '{table}'
                    AND CONSTRAINT_SCHEMA = '{DB_NAME}'
                """)
                constraints = cursor.fetchall()
                
                if constraints:
                    print(f"Cannot delete table '{table}' because it is referenced by:")
                    for referencing_table, constraint_name in constraints:
                        print(f"  - {referencing_table} (constraint: {constraint_name})")
                    results.append((table, False, f"Table has foreign key constraints"))
                    continue
                
                # Check record count before deleting
                record_count = count_records(conn, table)
                
                # Delete the table
                cursor.execute(f"DROP TABLE {table}")
                conn.commit()
                results.append((table, True, f"Deleted table with {record_count} records"))
                print(f"Deleted table: {table} (had {record_count} records)")
            except Error as e:
                conn.rollback()
                results.append((table, False, str(e)))
                print(f"Error deleting table {table}: {e}")
        else:
            results.append((table, False, "Table does not exist"))
            print(f"Table {table} does not exist")
    
    cursor.close()
    return results

def cleanup_duplicate_embeddings(conn):
    """Remove duplicate entries from person_embeddings table"""
    cursor = conn.cursor()
    
    try:
        # Find duplicate embeddings
        cursor.execute("""
            SELECT embedding, COUNT(*), GROUP_CONCAT(id ORDER BY id) as ids
            FROM person_embeddings
            GROUP BY embedding
            HAVING COUNT(*) > 1
        """)
        
        duplicates = cursor.fetchall()
        
        if not duplicates:
            print("No duplicate embeddings found.")
            cursor.close()
            return 0
        
        total_deleted = 0
        for embedding, count, ids in duplicates:
            id_list = ids.split(',')
            # Keep the first one, delete the rest
            ids_to_delete = id_list[1:]
            
            print(f"Found {count} duplicate embeddings with IDs: {ids}")
            print(f"  Keeping ID {id_list[0]}, deleting IDs: {', '.join(ids_to_delete)}")
            
            # Delete duplicate records
            cursor.execute(f"""
                DELETE FROM person_embeddings 
                WHERE id IN ({','.join(ids_to_delete)})
            """)
            
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            print(f"  Deleted {deleted_count} duplicate records")
        
        conn.commit()
        print(f"Total duplicate embeddings deleted: {total_deleted}")
        cursor.close()
        return total_deleted
    
    except Error as e:
        conn.rollback()
        print(f"Error cleaning up duplicate embeddings: {e}")
        cursor.close()
        return 0

def remove_zero_face_images(conn):
    """Remove images with zero face count from processed_images table"""
    cursor = conn.cursor()
    
    try:
        # Find images with zero faces
        cursor.execute("SELECT COUNT(*) FROM processed_images WHERE face_count = 0")
        zero_face_count = cursor.fetchone()[0]
        
        if zero_face_count == 0:
            print("No images with zero face count found.")
            cursor.close()
            return 0
        
        # Delete images with zero faces
        cursor.execute("DELETE FROM processed_images WHERE face_count = 0")
        deleted_count = cursor.rowcount
        
        conn.commit()
        print(f"Deleted {deleted_count} images with zero face count")
        cursor.close()
        return deleted_count
    
    except Error as e:
        conn.rollback()
        print(f"Error removing zero face images: {e}")
        cursor.close()
        return 0

def main():
    print("=" * 60)
    print("Database Cleanup Tool")
    print("=" * 60)
    
    conn = connect_to_db()
    
    # 1. List all tables and their record counts
    tables = list_tables(conn)
    print("\nCurrent tables in database:")
    for table in tables:
        count = count_records(conn, table)
        print(f"  - {table}: {count} records")
    
    # 2. Ask user which tables to delete
    print("\nWhich tables would you like to delete? (comma-separated, or 'none' to skip)")
    print("Suggested tables to check: customerdataextr (if empty), sleep_mode_state")
    
    tables_input = input("> ").strip().lower()
    
    tables_deletion_results = []
    if tables_input != "none" and tables_input:
        tables_to_delete = [t.strip() for t in tables_input.split(",")]
        tables_deletion_results = delete_unwanted_tables(conn, tables_to_delete)
    
    # 3. Clean up duplicate embeddings
    print("\nChecking for duplicate embeddings...")
    duplicates_deleted = cleanup_duplicate_embeddings(conn)
    
    # 4. Remove images with zero face count
    print("\nRemoving images with zero face count...")
    zero_face_images_deleted = remove_zero_face_images(conn)
    
    # 5. Generate SQL for future use
    print("\nGenerating SQL commands for future use:")
    if tables_input != "none" and tables_input:
        tables_to_delete = [t.strip() for t in tables_input.split(",")]
        for table in tables_to_delete:
            print(f"DROP TABLE IF EXISTS {table};")
    
    print("DELETE FROM person_embeddings WHERE id IN (SELECT id FROM (SELECT id, embedding, ROW_NUMBER() OVER(PARTITION BY embedding ORDER BY id) as row_num FROM person_embeddings) t WHERE row_num > 1);")
    print("DELETE FROM processed_images WHERE face_count = 0;")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("Cleanup Summary")
    print("=" * 60)
    
    if tables_deletion_results:
        print("\nTable Deletion Results:")
        for table, success, message in tables_deletion_results:
            status = "✓" if success else "✗"
            print(f"  {status} {table}: {message}")
    
    print(f"\nDuplicate Embeddings Deleted: {duplicates_deleted}")
    print(f"Zero Face Images Deleted: {zero_face_images_deleted}")
    
    # Close connection
    conn.close()
    print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()
