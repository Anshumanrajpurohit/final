"""
Duplicate Detection and Sleep Mode Service
Handles image and face duplicate detection with intelligent sleep mode
"""
import hashlib
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass

@dataclass
class SleepModeConfig:
    """Configuration for sleep mode behavior"""
    duplicate_sleep_duration: int = 300  # 5 minutes
    max_duplicate_threshold: int = 3
    image_hash_cache_duration: int = 86400  # 24 hours
    sleep_mode_backoff_multiplier: float = 1.5
    enable_sleep_mode: bool = True
    min_sleep_duration: int = 60  # 1 minute
    max_sleep_duration: int = 3600  # 1 hour

@dataclass
class DuplicateEvent:
    """Duplicate detection event"""
    event_time: datetime
    duplicate_type: str  # 'image', 'face', 'batch'
    duplicate_count: int
    sleep_duration: int
    description: str

class DuplicateDetector:
    def __init__(self, mysql_service, config: SleepModeConfig = None):
        self.mysql_service = mysql_service
        self.config = config or SleepModeConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize sleep mode state
        self._init_sleep_mode_state()
    
    def _init_sleep_mode_state(self):
        """Initialize sleep mode state in database"""
        try:
            # Check if sleep mode state exists, create if not
            self.mysql_service.execute_query(
                "INSERT IGNORE INTO sleep_mode_state (is_sleeping) VALUES (FALSE)"
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize sleep mode state: {e}")
    
    def generate_image_hash(self, image_path: str) -> Optional[str]:
        """Generate MD5 hash for image file"""
        try:
            if not os.path.exists(image_path):
                return None
                
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return hashlib.md5(image_data).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating image hash for {image_path}: {e}")
            return None
    
    def is_image_processed(self, image_hash: str, image_name: str) -> bool:
        """Check if image has been processed recently"""
        try:
            # Check if image hash exists in processed_images table
            query = """
            SELECT COUNT(*) FROM processed_images 
            WHERE image_hash = %s OR image_name = %s
            """
            result = self.mysql_service.execute_query(query, (image_hash, image_name))
            
            if result and result[0][0] > 0:
                self.logger.info(f"[DUPLICATE] Image already processed: {image_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking image processing status: {e}")
            return False
    
    def mark_image_processed(self, image_hash: str, image_name: str, image_url: str, face_count: int) -> bool:
        """Mark image as processed"""
        try:
            query = """
            INSERT INTO processed_images (image_hash, image_name, image_url, face_count)
            VALUES (%s, %s, %s, %s)
            """
            self.mysql_service.execute_query(query, (image_hash, image_name, image_url, face_count))
            return True
            
        except Exception as e:
            self.logger.error(f"Error marking image as processed: {e}")
            return False
    
    def detect_face_duplicates(self, face_encodings: List, known_encodings: List[Tuple[str, List[float]]]) -> Tuple[int, List[str]]:
        """Detect duplicate faces against known encodings"""
        duplicate_count = 0
        duplicate_person_ids = []
        
        if not known_encodings:
            return 0, []
        
        for face_encoding in face_encodings:
            # Find best match
            best_match_id = None
            best_distance = float('inf')
            
            for person_id, known_encoding in known_encodings:
                try:
                    import numpy as np
                    distance = np.linalg.norm(np.array(face_encoding) - np.array(known_encoding))
                    
                    if distance < self.config.duplicate_sleep_duration and distance < best_distance:
                        best_distance = distance
                        best_match_id = person_id
                        
                except Exception as e:
                    self.logger.warning(f"Error comparing face encodings: {e}")
                    continue
            
            if best_match_id:
                duplicate_count += 1
                duplicate_person_ids.append(best_match_id)
        
        return duplicate_count, duplicate_person_ids
    
    def log_duplicate_event(self, event: DuplicateEvent) -> bool:
        """Log duplicate detection event"""
        try:
            query = """
            INSERT INTO duplicate_events (event_time, duplicate_type, duplicate_count, sleep_duration, description)
            VALUES (%s, %s, %s, %s, %s)
            """
            self.mysql_service.execute_query(query, (
                event.event_time,
                event.duplicate_type,
                event.duplicate_count,
                event.sleep_duration,
                event.description
            ))
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging duplicate event: {e}")
            return False
    
    def enter_sleep_mode(self, reason: str, duration: int = None) -> bool:
        """Enter sleep mode"""
        if not self.config.enable_sleep_mode:
            return False
        
        try:
            if duration is None:
                duration = self.config.duplicate_sleep_duration
            
            # Ensure duration is within bounds
            duration = max(self.config.min_sleep_duration, 
                          min(duration, self.config.max_sleep_duration))
            
            sleep_start = datetime.now()
            sleep_end = sleep_start + timedelta(seconds=duration)
            
            query = """
            UPDATE sleep_mode_state 
            SET is_sleeping = TRUE, sleep_start_time = %s, sleep_end_time = %s, 
                sleep_duration = %s, sleep_reason = %s, updated_at = NOW()
            """
            self.mysql_service.execute_query(query, (sleep_start, sleep_end, duration, reason))
            
            self.logger.info(f"[SLEEP] Entering sleep mode for {duration}s - Reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error entering sleep mode: {e}")
            return False
    
    def exit_sleep_mode(self, trigger: str = "automatic") -> bool:
        """Exit sleep mode"""
        try:
            query = """
            UPDATE sleep_mode_state 
            SET is_sleeping = FALSE, sleep_end_time = NOW(), wake_up_trigger = %s, updated_at = NOW()
            """
            self.mysql_service.execute_query(query, (trigger,))
            
            self.logger.info(f"[WAKE] Exiting sleep mode - Trigger: {trigger}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exiting sleep mode: {e}")
            return False
    
    def is_sleeping(self) -> bool:
        """Check if system is currently in sleep mode"""
        try:
            query = "SELECT is_sleeping, sleep_end_time FROM sleep_mode_state ORDER BY id DESC LIMIT 1"
            result = self.mysql_service.execute_query(query)
            
            if result and result[0]:
                is_sleeping = result[0][0]
                sleep_end_time = result[0][1]
                
                # If sleep mode is active, check if it's time to wake up
                if is_sleeping and sleep_end_time:
                    if datetime.now() >= sleep_end_time:
                        self.exit_sleep_mode("timeout")
                        return False
                    return True
                
                return is_sleeping
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking sleep mode status: {e}")
            return False
    
    def get_sleep_status(self) -> Dict:
        """Get current sleep mode status"""
        try:
            query = """
            SELECT is_sleeping, sleep_start_time, sleep_end_time, sleep_duration, sleep_reason
            FROM sleep_mode_state ORDER BY id DESC LIMIT 1
            """
            result = self.mysql_service.execute_query(query)
            
            if result and result[0]:
                is_sleeping, start_time, end_time, duration, reason = result[0]
                
                if is_sleeping and end_time:
                    remaining_time = (end_time - datetime.now()).total_seconds()
                    remaining_time = max(0, remaining_time)
                else:
                    remaining_time = 0
                
                return {
                    'is_sleeping': bool(is_sleeping),
                    'sleep_start_time': start_time,
                    'sleep_end_time': end_time,
                    'sleep_duration': duration,
                    'sleep_reason': reason,
                    'remaining_seconds': int(remaining_time)
                }
            
            return {
                'is_sleeping': False,
                'sleep_start_time': None,
                'sleep_end_time': None,
                'sleep_duration': 0,
                'sleep_reason': None,
                'remaining_seconds': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sleep status: {e}")
            return {'is_sleeping': False, 'remaining_seconds': 0}
    
    def calculate_sleep_duration(self, duplicate_count: int, duplicate_type: str) -> int:
        """Calculate sleep duration based on duplicate count and type"""
        base_duration = self.config.duplicate_sleep_duration
        
        if duplicate_type == 'image':
            # Image duplicates get longer sleep
            duration = base_duration * (duplicate_count ** 0.5)
        elif duplicate_type == 'face':
            # Face duplicates get moderate sleep
            duration = base_duration * (duplicate_count ** 0.3)
        else:
            # Batch duplicates get shorter sleep
            duration = base_duration * (duplicate_count ** 0.2)
        
        # Apply backoff multiplier for repeated events
        recent_events = self.get_recent_duplicate_events(hours=1)
        if recent_events:
            duration *= (self.config.sleep_mode_backoff_multiplier ** len(recent_events))
        
        # Ensure duration is within bounds
        duration = max(self.config.min_sleep_duration, 
                      min(duration, self.config.max_sleep_duration))
        
        return int(duration)
    
    def get_recent_duplicate_events(self, hours: int = 1) -> List[Dict]:
        """Get recent duplicate events"""
        try:
            query = """
            SELECT event_time, duplicate_type, duplicate_count, sleep_duration, description
            FROM duplicate_events 
            WHERE event_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY event_time DESC
            """
            result = self.mysql_service.execute_query(query, (hours,))
            
            events = []
            for row in result:
                events.append({
                    'event_time': row[0],
                    'duplicate_type': row[1],
                    'duplicate_count': row[2],
                    'sleep_duration': row[3],
                    'description': row[4]
                })
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting recent duplicate events: {e}")
            return []
    
    def cleanup_old_processed_images(self, days: int = 7) -> int:
        """Clean up old processed image records"""
        try:
            query = """
            DELETE FROM processed_images 
            WHERE processed_at < DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            result = self.mysql_service.execute_query(query, (days,))
            return result.rowcount if result else 0
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old processed images: {e}")
            return 0
    
    def get_duplicate_statistics(self) -> Dict:
        """Get duplicate detection statistics"""
        try:
            # Get recent duplicate events
            recent_events = self.get_recent_duplicate_events(hours=24)
            
            # Count by type
            image_duplicates = sum(1 for e in recent_events if e['duplicate_type'] == 'image')
            face_duplicates = sum(1 for e in recent_events if e['duplicate_type'] == 'face')
            batch_duplicates = sum(1 for e in recent_events if e['duplicate_type'] == 'batch')
            
            # Get sleep mode statistics
            sleep_status = self.get_sleep_status()
            
            return {
                'total_duplicates_24h': len(recent_events),
                'image_duplicates': image_duplicates,
                'face_duplicates': face_duplicates,
                'batch_duplicates': batch_duplicates,
                'is_sleeping': sleep_status['is_sleeping'],
                'remaining_sleep_seconds': sleep_status['remaining_seconds'],
                'sleep_reason': sleep_status['sleep_reason']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting duplicate statistics: {e}")
            return {
                'total_duplicates_24h': 0,
                'image_duplicates': 0,
                'face_duplicates': 0,
                'batch_duplicates': 0,
                'is_sleeping': False,
                'remaining_sleep_seconds': 0,
                'sleep_reason': None
            }
