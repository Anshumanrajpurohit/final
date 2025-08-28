import mysql.connector
from mysql.connector import Error
from config.database import MySQLConfig
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

class MySQLService:
    def insert_customer_data(self, person_id: str, age: int, gender: str, skin_tone: str, hair_status: str) -> bool:
        """Insert extracted customer data into customerdataextr table if not already present"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
            cursor = connection.cursor()
            # Check if person_id already exists
            query = "SELECT id FROM customerdataextr WHERE person_id = %s"
            cursor.execute(query, (person_id,))
            result = cursor.fetchone()
            if result:
                cursor.close()
                connection.close()
                return False  # Already exists
            # Insert new record
            query = """
            INSERT INTO customerdataextr (person_id, age, gender, skin_tone, hair_status)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (person_id, age, gender, skin_tone, hair_status))
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"Error inserting customer data: {e}")
            return False

    def customer_data_exists(self, person_id: str) -> bool:
        """Check if customer data for a person_id already exists in customerdataextr table"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
            cursor = connection.cursor()
            query = "SELECT id FROM customerdataextr WHERE person_id = %s"
            cursor.execute(query, (person_id,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            return result is not None
        except Error as e:
            print(f"Error checking customer data existence: {e}")
            return False
    def __init__(self):
        self.config = MySQLConfig()
    
    def insert_temp_face(self, original_image_url: str, cropped_face_path: str, face_encoding: List[float]) -> bool:
        """Insert face data into temporary table"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
                
            cursor = connection.cursor()
            
            # Handle numpy array conversion
            if isinstance(face_encoding, np.ndarray):
                encoding_json = json.dumps(face_encoding.tolist())
            else:
                encoding_json = json.dumps(face_encoding)
                
            query = """
            INSERT INTO temp_faces (original_image_url, cropped_face_path, face_encoding)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (original_image_url, cropped_face_path, encoding_json))
            connection.commit()
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"Error inserting temp face: {e}")
            return False
    
    def get_all_face_encodings(self) -> List[np.ndarray]:
        """Get all face encodings from unique_persons table"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return []
            
            cursor = connection.cursor()
            query = "SELECT face_encoding FROM unique_persons WHERE face_encoding IS NOT NULL"
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Convert JSON strings to numpy arrays
            return [np.array(json.loads(encoding[0])) for encoding in results]
        except Error as e:
            print(f"Error getting face encodings: {e}")
            return []

    def get_person_id_by_index(self, index: int) -> Optional[str]:
        """Get person_id by index from unique_persons table"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return None
            
            cursor = connection.cursor()
            query = "SELECT person_id FROM unique_persons LIMIT 1 OFFSET %s"
            cursor.execute(query, (index,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            return result[0] if result else None
        except Error as e:
            print(f"Error getting person_id: {e}")
            return None

    def increment_visit_count(self, person_id: str) -> bool:
        """Increment visit count for a person"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            query = """
            UPDATE unique_persons 
            SET visit_count = visit_count + 1,
                last_seen = NOW()
            WHERE person_id = %s
            """
            cursor.execute(query, (person_id,))
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"Error incrementing visit count: {e}")
            return False

    def update_customer_data(self, person_id: str, age: int, gender: str, skin_tone: str, hair_status: str) -> bool:
        """Update customer data in the database"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            query = """
            UPDATE customerdataextr
            SET age = %s,
                gender = %s,
                skin_tone = %s,
                hair_status = %s
            WHERE person_id = %s
            """
            cursor.execute(query, (age, gender, skin_tone, hair_status, person_id))
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"Error updating customer data: {e}")
            return False

    def store_face_encoding(self, person_id: str, face_encoding: np.ndarray) -> bool:
        """Store face encoding for a new person"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            encoding_json = json.dumps(face_encoding.tolist())
            
            query = """
            INSERT INTO unique_persons (person_id, face_encoding, first_seen, last_seen, visit_count)
            VALUES (%s, %s, NOW(), NOW(), 1)
            """
            cursor.execute(query, (person_id, encoding_json))
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"Error storing face encoding: {e}")
            return False

    def get_all_unique_encodings(self) -> List[Tuple[str, List[float]]]:
        """Get all face encodings from unique_persons table"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return []
                
            cursor = connection.cursor()
            
            # Check if face_encoding column exists, if not use embedding from person_embeddings
            try:
                query = "SELECT person_id, face_encoding FROM unique_persons WHERE face_encoding IS NOT NULL"
                cursor.execute(query)
                results = cursor.fetchall()
            except Error:
                # Fallback to person_embeddings table
                query = """
                SELECT pe.person_id, pe.embedding 
                FROM person_embeddings pe
                JOIN unique_persons up ON pe.person_id = up.person_id
                """
                cursor.execute(query)
                results = cursor.fetchall()
            
            encodings = []
            for person_id, encoding_data in results:
                try:
                    # Skip null or empty data
                    if encoding_data is None or encoding_data == '':
                        continue
                        
                    if isinstance(encoding_data, str):
                        if encoding_data.strip() == '':
                            continue
                        encoding = json.loads(encoding_data)
                    else:
                        # Handle BLOB data from person_embeddings
                        if encoding_data is None:
                            continue
                        encoding = json.loads(encoding_data.decode('utf-8'))
                    encodings.append((person_id, encoding))
                except Exception as e:
                    print(f"Error processing encoding for person {person_id}: {e}")
                    continue
            
            cursor.close()
            connection.close()
            return encodings
            
        except Error as e:
            print(f"Error getting encodings: {e}")
            return []
    
    def insert_new_person(self, face_encoding: List[float], image_url: str, confidence: float) -> str:
        """Insert new unique person"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return ""
                
            cursor = connection.cursor()
            
            person_id = str(uuid.uuid4())
            
            # Handle numpy array conversion
            if isinstance(face_encoding, np.ndarray):
                encoding_json = json.dumps(face_encoding.tolist())
                encoding_list = face_encoding.tolist()
            else:
                encoding_json = json.dumps(face_encoding)
                encoding_list = face_encoding
            
            # Insert into unique_persons
            query = """
            INSERT INTO unique_persons (person_id, first_seen, last_seen, visit_count)
            VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            """
            cursor.execute(query, (person_id,))
            
            # Insert encoding into person_embeddings
            encoding_blob = json.dumps(encoding_list).encode('utf-8')
            query = """
            INSERT INTO person_embeddings (person_id, embedding)
            VALUES (%s, %s)
            """
            cursor.execute(query, (person_id, encoding_blob))
            
            connection.commit()
            cursor.close()
            connection.close()
            return person_id
            
        except Error as e:
            print(f"Error inserting new person: {e}")
            return ""
    
    def update_person_visit(self, person_id: str) -> bool:
        """Update visit count and last seen for existing person"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
                
            cursor = connection.cursor()
            
            query = """
            UPDATE unique_persons 
            SET visit_count = visit_count + 1, last_seen = CURRENT_TIMESTAMP
            WHERE person_id = %s
            """
            cursor.execute(query, (person_id,))
            connection.commit()
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"Error updating person visit: {e}")
            return False
    
    def get_unprocessed_temp_faces(self) -> List[Dict]:
        """Get unprocessed faces from temp table"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return []
                
            cursor = connection.cursor()
            
            query = """
            SELECT id, original_image_url, cropped_face_path, face_encoding
            FROM temp_faces WHERE processed = FALSE
            """
            cursor.execute(query)
            results = cursor.fetchall()
            
            faces = []
            for row in results:
                try:
                    face_encoding = None
                    if row[3] and row[3].strip() != '':
                        face_encoding = json.loads(row[3])
                    
                    faces.append({
                        'id': row[0],
                        'original_image_url': row[1],
                        'cropped_face_path': row[2],
                        'face_encoding': face_encoding
                    })
                except Exception as e:
                    print(f"Error processing temp face {row[0]}: {e}")
                    continue
            
            cursor.close()
            connection.close()
            return faces
            
        except Error as e:
            print(f"Error getting temp faces: {e}")
            return []
    
    def mark_temp_face_processed(self, temp_id: int) -> bool:
        """Mark temp face as processed"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
                
            cursor = connection.cursor()
            
            query = "UPDATE temp_faces SET processed = TRUE WHERE id = %s"
            cursor.execute(query, (temp_id,))
            connection.commit()
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"Error marking face processed: {e}")
            return False
    
    def clear_processed_temp_faces(self) -> bool:
        """Clear all processed temporary faces"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
                
            cursor = connection.cursor()
            
            query = "DELETE FROM temp_faces WHERE processed = TRUE"
            cursor.execute(query)
            connection.commit()
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"Error clearing temp faces: {e}")
            return False
    
    def log_processing_batch(self, batch_data: Dict) -> bool:
        """Log processing batch statistics"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return False
                
            cursor = connection.cursor()
            
            query = """
            INSERT INTO processing_logs 
            (batch_id, images_processed, faces_detected, new_persons, existing_persons, processing_time, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                batch_data['batch_id'],
                batch_data['images_processed'],
                batch_data['faces_detected'],
                batch_data['new_persons'],
                batch_data['existing_persons'],
                batch_data['processing_time'],
                batch_data['status']
            ))
            connection.commit()
            
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            print(f"Error logging batch: {e}")
            return False
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return None
                
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                cursor.close()
                connection.close()
                return results
            else:
                # For INSERT, UPDATE, DELETE
                connection.commit()
                cursor.close()
                connection.close()
                return cursor
                
        except Error as e:
            print(f"Error executing query: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            connection = self.config.get_connection()
            if connection is None:
                return {}
                
            cursor = connection.cursor()
            
            # Get unique persons count
            cursor.execute("SELECT COUNT(*) FROM unique_persons")
            total_persons = cursor.fetchone()[0]
            
            # Get total visits
            cursor.execute("SELECT SUM(visit_count) FROM unique_persons")
            total_visits = cursor.fetchone()[0] or 0
            
            # Get recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM unique_persons 
                WHERE last_seen >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            """)
            recent_visitors = cursor.fetchone()[0]
            
            cursor.close()
            connection.close()
            
            return {
                'total_unique_persons': total_persons,
                'total_visits': total_visits,
                'recent_visitors_24h': recent_visitors
            }
            
        except Error as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_unique_persons': 0,
                'total_visits': 0,
                'recent_visitors_24h': 0
            }
    
    def store_face_embedding(self, person_id, embedding):
        embedding_bytes = embedding.tobytes()
        self.cursor.execute("""
            INSERT INTO person_embeddings (person_id, embedding) 
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE embedding = VALUES(embedding)
        """, (person_id, embedding_bytes))
    
    def get_stored_embedding(self, person_id):
        self.cursor.execute("""
            SELECT embedding FROM person_embeddings 
            WHERE person_id = %s
        """, (person_id,))
        result = self.cursor.fetchone()
        if result:
            return np.frombuffer(result[0], dtype=np.float32)
        return None