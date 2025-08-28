import mysql.connector
from mysql.connector import Error
from typing import Optional, Dict, List, Any
import numpy as np
from datetime import datetime
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class EnhancedMySQLService:
    def __init__(self, config):
        self.config = config
        
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[tuple]]:
        """Execute a SQL query and return results"""
        try:
            connection = self.config.get_connection()
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
            connection.close()
            return result
            
        except Error as e:
            logger.error(f"Database error: {str(e)}")
            return None
            
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
            connection = self.config.get_connection()
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
            connection.close()
            return True
            
        except Error as e:
            logger.error(f"Error updating person attributes: {e}")
            return False
            
    def insert_new_person(self, face_data: Dict[str, Any]) -> Optional[str]:
        """Insert a new person with face encoding, age, and gender"""
        try:
            connection = self.config.get_connection()
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
                embedding_json = json.dumps(face_data['face_encoding'].tolist())
                cursor.execute(embedding_query, (person_id, embedding_json))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            return person_id
            
        except Error as e:
            logger.error(f"Error inserting new person: {e}")
            return None
