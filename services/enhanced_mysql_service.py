import mysql.connector
from mysql.connector import Error
from typing import Optional, Dict, List, Any
import numpy as np
from datetime import datetime
import logging
import uuid
import json
from .mysql_service import MySQLService

logger = logging.getLogger(__name__)


class EnhancedMySQLService(MySQLService):  # extends MySQLService
    def __init__(self, config):
        # If config is a dict-like with DB creds, try to create MySQLConfig-compatible wrapper
        try:
            super().__init__()
        except Exception:
            pass
        # Prefer base class MySQLConfig, but allow external config object with get_connection
        if hasattr(config, 'get_connection'):
            self.config = config

    def _get_conn(self):
        try:
            conn = None
            if hasattr(self.config, 'get_connection'):
                conn = self.config.get_connection()
            elif hasattr(self, 'conn') and getattr(self, 'conn'):
                conn = self.conn
            return conn
        except Exception:
            return None

    def execute_query(self, query: str, params: tuple = None) -> Optional[List[tuple]]:
        """Execute a SQL query and return results"""
        try:
            connection = self._get_conn()
            if not connection:
                return None

            cursor = connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = cursor.fetchall() if cursor.description else None
            connection.commit()
            cursor.close()
            try:
                connection.close()
            except Exception:
                pass
            return result

        except Error as e:
            logger.error(f"Database error: {str(e)}")
            return None

    def insert_face_data(self, *, image_url: str, image_path: str, face_encoding: List[float], crop_path: str,
                          quality_score: float, age: str, gender: str, x: int, y: int, width: int, height: int) -> Dict[str, Any]:
        """Insert face data and return dict with new_person flag.
        Stores encoding in person_embeddings and creates/updates unique_persons.
        """
        res: Dict[str, Any] = {"new_person": False}
        try:
            conn = self._get_conn()
            if not conn:
                return res
            cur = conn.cursor()

            # Fetch known encodings for matching
            known = self.get_all_unique_encodings()  # List[(person_id, embedding list)]
            matched_person_id = None
            try:
                import numpy as np
                q = np.array(face_encoding, dtype=np.float32)
                best_dist = 1e9
                for pid, emb in known:
                    try:
                        dist = float(np.linalg.norm(q - np.array(emb, dtype=np.float32)))
                        if dist < 0.6 and dist < best_dist:
                            best_dist = dist
                            matched_person_id = pid
                    except Exception:
                        continue
            except Exception:
                pass

            if matched_person_id:
                # Existing person: update visit
                try:
                    self.update_person_visit(matched_person_id)
                except Exception:
                    pass
                res["new_person"] = False
            else:
                # New person
                matched_person_id = self.insert_new_person(face_encoding, image_url, quality_score) or ""
                res["new_person"] = True if matched_person_id else False

            # Optionally store meta (crop, bbox) in temp or a log table if available
            try:
                cur.execute(
                    """
                    INSERT INTO face_detections (person_id, image_url, image_path, crop_path, x, y, width, height, quality_score, age, gender, detected_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    """,
                    (matched_person_id, image_url, image_path, crop_path, x, y, width, height, quality_score, age, gender),
                )
                conn.commit()
            except Exception:
                pass

            try:
                cur.close()
                conn.close()
            except Exception:
                pass
            return res
        except Exception as e:
            logger.error(f"insert_face_data failed: {e}")
            return res

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics from database"""
        try:
            stats = {
                'total_persons': 0,
                'total_faces': 0,
                'recent_detections': 0,
                'age_stats': {},
                'gender_stats': {}
            }

            # Get total persons count
            query = "SELECT COUNT(*) FROM unique_persons WHERE is_active = TRUE"
            result = self.execute_query(query)
            if result:
                stats['total_persons'] = result[0][0]

            # Get age distribution
            query = """
                SELECT 
                    CASE 
                        WHEN age < 18 THEN 'under_18'
                        WHEN age BETWEEN 18 AND 30 THEN '18_30'
                        WHEN age BETWEEN 31 AND 50 THEN '31_50'
                        ELSE 'over_50'
                    END as age_group,
                    COUNT(*) as count
                FROM unique_persons
                WHERE age IS NOT NULL
                GROUP BY age_group
            """
            result = self.execute_query(query)
            if result:
                stats['age_stats'] = {row[0]: row[1] for row in result}

            # Get gender distribution
            query = """
                SELECT gender, COUNT(*) as count
                FROM unique_persons
                WHERE gender IS NOT NULL
                GROUP BY gender
            """
            result = self.execute_query(query)
            if result:
                stats['gender_stats'] = {row[0]: row[1] for row in result}

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                'error': str(e),
                'total_persons': 0,
                'total_faces': 0,
                'recent_detections': 0
            }

    def update_person_attributes(self, person_id: str, age: Optional[int], gender: Optional[str]) -> bool:
        """Update age and gender for an existing person"""
        try:
            connection = self._get_conn()
            if not connection:
                return False

            cursor = connection.cursor()

            # Only update if we have valid values
            if age is not None or gender is not None:
                update_fields = []
                params = []

                if age is not None:
                    update_fields.append("age = %s")
                    params.append(age)

                if gender is not None:
                    update_fields.append("gender = %s")
                    params.append(gender)

                if update_fields:
                    update_fields.append("age_updated_at = CURRENT_TIMESTAMP")
                    query = f"""
                        UPDATE unique_persons 
                        SET {', '.join(update_fields)}
                        WHERE person_id = %s
                    """
                    params.append(person_id)

                    cursor.execute(query, tuple(params))
                    connection.commit()

            cursor.close()
            try:
                connection.close()
            except Exception:
                pass
            return True

        except Error as e:
            logger.error(f"Error updating person attributes: {e}")
            return False

    def insert_new_person(self, face_data: Dict[str, Any]) -> Optional[str]:
        """Insert a new person with face encoding, age, and gender"""
        try:
            connection = self._get_conn()
            if not connection:
                return None

            cursor = connection.cursor()

            # Generate new person_id
            person_id = str(uuid.uuid4())

            # Insert into unique_persons
            query = """
                INSERT INTO unique_persons 
                (person_id, age, gender, age_updated_at, confidence_score)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s)
            """
            cursor.execute(query, (
                person_id,
                face_data.get('age'),
                face_data.get('gender'),
                face_data.get('quality_score', 0.9)
            ))

            # Insert face encoding
            if 'face_encoding' in face_data:
                embedding_query = """
                    INSERT INTO person_embeddings 
                    (person_id, embedding)
                    VALUES (%s, %s)
                """
                try:
                    embedding_json = json.dumps(face_data['face_encoding'].tolist())
                except Exception:
                    # fallback: convert numpy array to list safely
                    embedding_json = json.dumps(list(map(float, face_data['face_encoding'])))
                cursor.execute(embedding_query, (person_id, embedding_json))

            connection.commit()
            cursor.close()
            try:
                connection.close()
            except Exception:
                pass

            return person_id

        except Error as e:
            logger.error(f"Error inserting new person: {e}")
            return None

    def get_unprocessed_temp_faces(self):
        """
        Return rows from temp_faces where processed=0.
        Tries parent implementation first, then falls back to a direct query.
        """
        log = logging.getLogger(__name__)
        try:
            parent = super()
            if hasattr(parent, "get_unprocessed_temp_faces"):
                return parent.get_unprocessed_temp_faces()
        except Exception:
            pass

        try:
            cur = self.conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM temp_faces WHERE processed=0")
            rows = cur.fetchall()
            cur.close()
            return rows or []
        except Exception:
            log.exception("get_unprocessed_temp_faces failed")
            return []

    def log_processing_batch(self, batch_stats: dict):
        """
        Persist batch processing summary. Accepts a dict with keys like:
        batch_id, images_processed, faces_detected, duration
        """
        log = logging.getLogger(__name__)
        try:
            parent = super()
            if hasattr(parent, "log_processing_batch"):
                return parent.log_processing_batch(batch_stats)
        except Exception:
            pass

        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO processing_logs (batch_id, images_processed, faces_detected, duration) VALUES (%s,%s,%s,%s)",
                (
                    batch_stats.get("batch_id"),
                    batch_stats.get("images_processed", 0),
                    batch_stats.get("faces_detected", 0),
                    batch_stats.get("duration", 0.0),
                ),
            )
            self.conn.commit()
            cur.close()
        except Exception:
            log.exception("log_processing_batch failed")
