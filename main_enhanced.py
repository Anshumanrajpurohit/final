"""
Enhanced Face Recognition System
Integrates performance optimization, quality assessment, and advanced monitoring
"""
import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
import schedule
import logging
import json
from typing import Dict, Any, List

from services.supabase_service import SupabaseService
from services.enhanced_mysql_service import EnhancedMySQLService
from services.enhanced_face_service import EnhancedFaceService
from utils.performance_optimizer import PerformanceOptimizer
from utils.duplicate_detector import DuplicateDetector, SleepModeConfig

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/face_recognition.log'),
        logging.StreamHandler()
    ]
)

class EnhancedFaceRecognitionSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.supabase_service = SupabaseService()
        
        # Initialize MySQL service with configuration
        from config.database import MySQLConfig
        mysql_config = MySQLConfig()
        self.mysql_service = EnhancedMySQLService(mysql_config)
        
        # Initialize enhanced face service with performance optimization
        self.face_service = EnhancedFaceService()
        
        # Performance optimizer
        self.performance_optimizer = PerformanceOptimizer()
        
        # Configuration
        self.max_batch_size = int(os.getenv('MAX_BATCH_SIZE', 10))
        self.processing_interval = int(os.getenv('PROCESSING_INTERVAL', 5))
        self.enable_quality_check = os.getenv('ENABLE_QUALITY_CHECK', 'true').lower() == 'true'
        self.enable_faiss = os.getenv('ENABLE_FAISS', 'true').lower() == 'true'
        
        # Duplicate detection and sleep mode configuration
        sleep_config = SleepModeConfig(
            duplicate_sleep_duration=int(os.getenv('DUPLICATE_SLEEP_DURATION', 300)),
            max_duplicate_threshold=int(os.getenv('MAX_DUPLICATE_THRESHOLD', 3)),
            image_hash_cache_duration=int(os.getenv('IMAGE_HASH_CACHE_DURATION', 86400)),
            sleep_mode_backoff_multiplier=float(os.getenv('SLEEP_MODE_BACKOFF_MULTIPLIER', 1.5)),
            enable_sleep_mode=os.getenv('ENABLE_SLEEP_MODE', 'true').lower() == 'true',
            min_sleep_duration=int(os.getenv('MIN_SLEEP_DURATION', 60)),
            max_sleep_duration=int(os.getenv('MAX_SLEEP_DURATION', 3600))
        )
        
        # Initialize duplicate detector
        self.duplicate_detector = DuplicateDetector(self.mysql_service, sleep_config)
        
        # Statistics tracking
        self.system_stats = {
            'total_batches': 0,
            'total_images_processed': 0,
            'total_faces_detected': 0,
            'total_new_persons': 0,
            'total_existing_persons': 0,
            'total_processing_time': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("[START] Enhanced Face Recognition System initialized")
        self.logger.info(f"Configuration: batch_size={self.max_batch_size}, "
                        f"interval={self.processing_interval}s, "
                        f"quality_check={self.enable_quality_check}, "
                        f"faiss={self.enable_faiss}")
        self.logger.info(f"Sleep Mode: enabled={sleep_config.enable_sleep_mode}, "
                        f"sleep_duration={sleep_config.duplicate_sleep_duration}s, "
                        f"max_duplicates={sleep_config.max_duplicate_threshold}")
    
    def process_batch_enhanced(self):
        """Enhanced batch processing with performance monitoring and duplicate detection"""
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"[START] Starting enhanced batch processing: {batch_id}")
        
        # Check if system is in sleep mode
        if self.duplicate_detector.is_sleeping():
            sleep_status = self.duplicate_detector.get_sleep_status()
            self.logger.info(f"[SLEEP] System in sleep mode - {sleep_status['remaining_seconds']}s remaining")
            return
        
        # Initialize batch statistics
        batch_stats = {
            'batch_id': batch_id,
            'images_processed': 0,
            'faces_detected': 0,
            'new_persons': 0,
            'existing_persons': 0,
            'processing_time': 0,
            'status': 'success',
            'quality_metrics': {},
            'performance_metrics': {},
            'duplicate_events': []
        }
        
        try:
            # Monitor memory before processing
            memory_before = self.performance_optimizer.monitor_memory_usage()
            self.logger.info(f"Memory before processing: {memory_before['rss_mb']:.1f} MB")
            
            # Step 1: Fetch recent images from Supabase
            recent_images = self.supabase_service.get_recent_images(self.max_batch_size)
            self.logger.info(f"[FOUND] Found {len(recent_images)} recent images")
            
            if not recent_images:
                self.logger.info("No new images to process")
                return

            self.logger.info("[AGE/GENDER] Age and gender detection enabled")
            
            # Step 1.5: Check for image duplicates
            duplicate_images = []
            new_images = []
            
            for image_info in recent_images:
                image_name = image_info['name']
                local_path = f"temp_images/{image_name}"
                
                # Download image temporarily to generate hash
                if self.supabase_service.download_image(image_name, local_path):
                    image_hash = self.duplicate_detector.generate_image_hash(local_path)
                    
                    if image_hash and self.duplicate_detector.is_image_processed(image_hash, image_name):
                        duplicate_images.append(image_info)
                        self.logger.info(f"[DUPLICATE] Image already processed: {image_name}")
                    else:
                        new_images.append((image_info, image_hash, local_path))
                else:
                    self.logger.warning(f"Could not download image: {image_name}")
            
            # Handle image duplicates
            if duplicate_images:
                duplicate_count = len(duplicate_images)
                sleep_duration = self.duplicate_detector.calculate_sleep_duration(duplicate_count, 'image')
                
                # Log duplicate event
                from utils.duplicate_detector import DuplicateEvent
                event = DuplicateEvent(
                    event_time=datetime.now(),
                    duplicate_type='image',
                    duplicate_count=duplicate_count,
                    sleep_duration=sleep_duration,
                    description=f"Found {duplicate_count} duplicate images"
                )
                self.duplicate_detector.log_duplicate_event(event)
                batch_stats['duplicate_events'].append(event)
                
                # Enter sleep mode if threshold exceeded
                if duplicate_count >= self.duplicate_detector.config.max_duplicate_threshold:
                    self.duplicate_detector.enter_sleep_mode(
                        f"Image duplicates: {duplicate_count}", 
                        sleep_duration
                    )
                    batch_stats['status'] = 'sleep_mode'
                    return
                
                self.logger.info(f"[DUPLICATE] Found {duplicate_count} duplicate images, continuing with new images")
            
            if not new_images:
                self.logger.info("No new images to process after duplicate check")
                return
            
            # Step 2: Process each new image with enhanced face detection
            for image_info, image_hash, local_path in new_images:
                try:
                    image_name = image_info['name']
                    
                    batch_stats['images_processed'] += 1
                    
                    # Step 3: Enhanced face detection with quality assessment
                    cropped_faces = self.face_service.detect_and_crop_faces_enhanced(
                        local_path,
                        detection_model='hog',  # Use HOG for speed
                        quality_check=self.enable_quality_check
                    )
                    
                    batch_stats['faces_detected'] += len(cropped_faces)
                    
                    # Step 4: Process each detected face with age and gender
                    for face_data in self.face_service.process_batch([local_path]):
                        if face_data['success']:
                            data = face_data['data']
                            
                            # Extract age from range
                            age = int(data['age_range'].split('-')[0]) if data['age_range'] != 'unknown' else None
                            
                            # Store quality metrics
                            batch_stats['quality_metrics'][local_path] = {
                                'quality_score': data['quality_score'],
                                'age_confidence': data['age_confidence'],
                                'gender_confidence': data['gender_confidence']
                            }
                            
                            # Prepare person data
                            person_data = {
                                'face_encoding': data['face_encoding'],
                                'age': age,
                                'gender': data['gender'],
                                'quality_score': data['quality_score']
                            }
                            
                            # Insert new person with age and gender
                            image_url = self.supabase_service.get_image_url(image_name)
                            person_id = self.mysql_service.insert_new_person(person_data)
                            
                            if person_id:
                                self.logger.info(f"[SUCCESS] New person added - Age: {data['age_range']}, Gender: {data['gender']}")
                                batch_stats['new_persons'] += 1
                    
                    # Mark image as processed
                    image_url = self.supabase_service.get_image_url(image_name)
                    self.duplicate_detector.mark_image_processed(image_hash, image_name, image_url, len(cropped_faces))
                    
                    # Clean up original downloaded image
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        
                except Exception as e:
                    self.logger.error(f"Error processing image {image_info.get('name', 'unknown')}: {e}")
                    continue
            
            # Step 5: Enhanced temp faces processing with duplicate detection
            self.process_temp_faces_enhanced(batch_stats)
            
            # Step 6: Clear processed temp faces
            self.mysql_service.clear_processed_temp_faces()
            
            # Step 7: Clean up temp files with performance monitoring
            self.face_service.cleanup_temp_files()
            
            # Monitor memory after processing
            memory_after = self.performance_optimizer.monitor_memory_usage()
            self.logger.info(f"Memory after processing: {memory_after['rss_mb']:.1f} MB")
            
            # Calculate performance metrics
            batch_stats['performance_metrics'] = {
                'memory_usage_mb': memory_after['rss_mb'],
                'memory_increase_mb': memory_after['rss_mb'] - memory_before['rss_mb'],
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            batch_stats['status'] = 'error'
            batch_stats['error_message'] = str(e)
        
        # Calculate processing time
        batch_stats['processing_time'] = time.time() - start_time
        
        # Update system statistics
        self._update_system_stats(batch_stats)
        
        # Log batch statistics
        self.mysql_service.log_processing_batch(batch_stats)
        
        # Log performance statistics
        self.performance_optimizer.log_performance_stats()
        
        # Print batch summary
        self._print_batch_summary(batch_stats)
        
        # Log duplicate events if any
        if batch_stats['duplicate_events']:
            self.logger.info(f"[DUPLICATE] Batch had {len(batch_stats['duplicate_events'])} duplicate events")
    
    def process_temp_faces_enhanced(self, batch_stats: Dict[str, Any]):
        """Enhanced temporary faces processing with performance optimization and duplicate detection"""
        # Get all unprocessed temp faces
        temp_faces = self.mysql_service.get_unprocessed_temp_faces()
        
        if not temp_faces:
            return
        
        # Get all known encodings
        known_encodings = self.mysql_service.get_all_unique_encodings()
        
        self.logger.info(f"Processing {len(temp_faces)} temp faces against {len(known_encodings)} known persons")
        
        # Extract face encodings for duplicate detection
        face_encodings = [temp_face['face_encoding'] for temp_face in temp_faces if temp_face['face_encoding']]
        
        # Check for face duplicates
        if face_encodings and known_encodings:
            duplicate_count, duplicate_person_ids = self.duplicate_detector.detect_face_duplicates(
                face_encodings, known_encodings
            )
            
            if duplicate_count > 0:
                self.logger.info(f"[DUPLICATE] Found {duplicate_count} duplicate faces")
                
                # Calculate sleep duration for face duplicates
                sleep_duration = self.duplicate_detector.calculate_sleep_duration(duplicate_count, 'face')
                
                # Log duplicate event
                from utils.duplicate_detector import DuplicateEvent
                event = DuplicateEvent(
                    event_time=datetime.now(),
                    duplicate_type='face',
                    duplicate_count=duplicate_count,
                    sleep_duration=sleep_duration,
                    description=f"Found {duplicate_count} duplicate faces"
                )
                self.duplicate_detector.log_duplicate_event(event)
                batch_stats['duplicate_events'].append(event)
                
                # Enter sleep mode if threshold exceeded
                if duplicate_count >= self.duplicate_detector.config.max_duplicate_threshold:
                    self.duplicate_detector.enter_sleep_mode(
                        f"Face duplicates: {duplicate_count}", 
                        sleep_duration
                    )
                    batch_stats['status'] = 'sleep_mode'
                    return
        
        # Process each temp face
        for temp_face in temp_faces:
            try:
                face_encoding = temp_face['face_encoding']
                if not face_encoding:
                    continue
                import numpy as np
                face_encoding_array = np.array(face_encoding)

                # Use enhanced person matching with FAISS support
                matching_person_id = self.face_service.find_matching_person_enhanced(
                    face_encoding_array,
                    known_encodings,
                    use_faiss=self.enable_faiss
                )

                if matching_person_id:
                    # Update existing person
                    self.mysql_service.update_person_visit(matching_person_id)
                    batch_stats['existing_persons'] += 1
                    self.logger.info(f"[UPDATE] Updated visit for person: {matching_person_id}")
                else:
                    # Create new person
                    person_id = self.mysql_service.insert_new_person(
                        face_encoding,
                        temp_face['original_image_url'],
                        0.9
                    )
                    if person_id:
                        batch_stats['new_persons'] += 1
                        self.logger.info(f"[NEW] Created new person: {person_id}")
                        # Add to known encodings for this batch
                        known_encodings.append((person_id, face_encoding))

                        # --- Attribute extraction and DB insert ---
                        # Try to load the cropped face image
                        crop_path = temp_face.get('cropped_face_path') or temp_face.get('cropped_path')
                        if not crop_path:
                            crop_path = temp_face.get('cropped_path')
                        if crop_path and os.path.exists(crop_path):
                            import cv2
                            face_img = cv2.imread(crop_path)
                            if face_img is not None:
                                attributes = self.face_service.extract_attributes(face_img)
                                # Insert only if not already present
                                if not self.mysql_service.customer_data_exists(person_id):
                                    self.mysql_service.insert_customer_data(
                                        person_id,
                                        attributes.get('age', None),
                                        attributes.get('gender', None),
                                        attributes.get('skin_tone', None),
                                        attributes.get('hair_status', None)
                                    )
                        else:
                            self.logger.warning(f"Cropped face image not found for attribute extraction: {crop_path}")

                # Mark as processed
                self.mysql_service.mark_temp_face_processed(temp_face['id'])

            except Exception as e:
                self.logger.error(f"Error processing temp face {temp_face['id']}: {e}")
                continue
    
    def _update_system_stats(self, batch_stats: Dict[str, Any]):
        """Update system-wide statistics"""
        self.system_stats['total_batches'] += 1
        self.system_stats['total_images_processed'] += batch_stats['images_processed']
        self.system_stats['total_faces_detected'] += batch_stats['faces_detected']
        self.system_stats['total_new_persons'] += batch_stats['new_persons']
        self.system_stats['total_existing_persons'] += batch_stats['existing_persons']
        self.system_stats['total_processing_time'] += batch_stats['processing_time']
    
    def _print_batch_summary(self, batch_stats: Dict[str, Any]):
        """Print detailed batch summary"""
        self.logger.info("=" * 60)
        self.logger.info("[BATCH] BATCH PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"[ID] Batch ID: {batch_stats['batch_id']}")
        self.logger.info(f"[IMAGES] Images processed: {batch_stats['images_processed']}")
        self.logger.info(f"[FACES] Faces detected: {batch_stats['faces_detected']}")
        self.logger.info(f"[NEW] New persons: {batch_stats['new_persons']}")
        self.logger.info(f"[RETURN] Returning persons: {batch_stats['existing_persons']}")
        self.logger.info(f"[TIME] Processing time: {batch_stats['processing_time']:.2f}s")
        self.logger.info(f"[STATUS] Status: {batch_stats['status'].upper()}")
        
        # Performance metrics
        if 'performance_metrics' in batch_stats:
            perf = batch_stats['performance_metrics']
            self.logger.info(f"[MEMORY] Memory usage: {perf.get('memory_usage_mb', 0):.1f} MB")
            self.logger.info(f"[INCREASE] Memory increase: {perf.get('memory_increase_mb', 0):.1f} MB")
        
        self.logger.info("=" * 60)
    
    def print_system_statistics(self):
        """Print comprehensive system statistics"""
        # Get database statistics
        db_stats = self.mysql_service.get_statistics()
        
        # Get performance statistics
        perf_stats = self.performance_optimizer.get_performance_stats()
        
        # Calculate system uptime
        uptime = datetime.now() - self.system_stats['start_time']
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("[STATS] ENHANCED SYSTEM STATISTICS")
        self.logger.info("=" * 70)
        
        # Database statistics
        self.logger.info("[DB] DATABASE STATISTICS:")
        self.logger.info(f"  Total Unique Persons: {db_stats.get('total_unique_persons', 0)}")
        self.logger.info(f"  Total Visits: {db_stats.get('total_visits', 0)}")
        self.logger.info(f"  Recent Visitors (24h): {db_stats.get('recent_visitors_24h', 0)}")
        
        # System statistics
        self.logger.info("\n[SYSTEM] SYSTEM STATISTICS:")
        self.logger.info(f"  Total Batches: {self.system_stats['total_batches']}")
        self.logger.info(f"  Total Images Processed: {self.system_stats['total_images_processed']}")
        self.logger.info(f"  Total Faces Detected: {self.system_stats['total_faces_detected']}")
        self.logger.info(f"  Total New Persons: {self.system_stats['total_new_persons']}")
        self.logger.info(f"  Total Existing Persons: {self.system_stats['total_existing_persons']}")
        self.logger.info(f"  Total Processing Time: {self.system_stats['total_processing_time']:.2f}s")
        self.logger.info(f"  System Uptime: {uptime}")
        
        # Performance statistics
        self.logger.info("\n[PERF] PERFORMANCE STATISTICS:")
        self.logger.info(f"  Memory Usage: {perf_stats['memory_usage_mb']:.1f} MB ({perf_stats['memory_percent']:.1f}%)")
        self.logger.info(f"  Average Processing Time: {perf_stats['avg_processing_time']:.3f}s")
        self.logger.info(f"  Cache Hit Rate: {perf_stats['cache_hit_rate']:.2%}")
        self.logger.info(f"  FAISS Available: {perf_stats['faiss_available']}")
        
        # Duplicate detection statistics
        duplicate_stats = self.duplicate_detector.get_duplicate_statistics()
        self.logger.info("\n[DUPLICATE] DUPLICATE DETECTION STATISTICS:")
        self.logger.info(f"  Total Duplicates (24h): {duplicate_stats['total_duplicates_24h']}")
        self.logger.info(f"  Image Duplicates: {duplicate_stats['image_duplicates']}")
        self.logger.info(f"  Face Duplicates: {duplicate_stats['face_duplicates']}")
        self.logger.info(f"  Batch Duplicates: {duplicate_stats['batch_duplicates']}")
        self.logger.info(f"  Sleep Mode: {'ACTIVE' if duplicate_stats['is_sleeping'] else 'INACTIVE'}")
        if duplicate_stats['is_sleeping']:
            self.logger.info(f"  Remaining Sleep: {duplicate_stats['remaining_sleep_seconds']}s")
            self.logger.info(f"  Sleep Reason: {duplicate_stats['sleep_reason']}")
        
        # Configuration
        self.logger.info("\n[CONFIG] CONFIGURATION:")
        self.logger.info(f"  Face Threshold: {self.face_service.threshold}")
        self.logger.info(f"  Max Batch Size: {self.max_batch_size}")
        self.logger.info(f"  Processing Interval: {self.processing_interval}s")
        self.logger.info(f"  Quality Check: {self.enable_quality_check}")
        self.logger.info(f"  FAISS Enabled: {self.enable_faiss}")
        self.logger.info(f"  Sleep Mode: {self.duplicate_detector.config.enable_sleep_mode}")
        self.logger.info(f"  Max Duplicate Threshold: {self.duplicate_detector.config.max_duplicate_threshold}")
        
        self.logger.info("=" * 70 + "\n")
    
    def run_continuous_enhanced(self):
        """Run the enhanced system continuously with monitoring"""
        self.logger.info("[START] Starting Enhanced Face Recognition System...")
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Schedule the processing
        schedule.every(self.processing_interval).seconds.do(self.process_batch_enhanced)
        schedule.every(30).seconds.do(self.print_system_statistics)
        schedule.every(1).hours.do(self.duplicate_detector.cleanup_old_processed_images, 7)
        
        # Initial statistics
        self.print_system_statistics()
        
        # Run scheduler
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("[STOP] System shutdown requested by user")
            self._shutdown_cleanup()
        except Exception as e:
            self.logger.error(f"[ERROR] System error: {e}")
            self._shutdown_cleanup()
    
    def _shutdown_cleanup(self):
        """Clean shutdown with cleanup"""
        self.logger.info("[CLEANUP] Performing cleanup...")
        
        # Final statistics
        self.print_system_statistics()
        
        # Cleanup temp files
        self.face_service.cleanup_temp_files()
        
        # Force garbage collection
        self.performance_optimizer.force_garbage_collection()
        
        self.logger.info("[DONE] System shutdown complete")

if __name__ == "__main__":
    system = EnhancedFaceRecognitionSystem()
    system.run_continuous_enhanced()
