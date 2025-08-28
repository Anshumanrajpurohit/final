import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
import schedule

from services.supabase_service import SupabaseService
from services.mysql_service import MySQLService
from services.face_service import FaceService

load_dotenv()

class FaceRecognitionSystem:
    def __init__(self):
        self.supabase_service = SupabaseService()
        self.mysql_service = MySQLService()
        self.face_service = FaceService(threshold=float(os.getenv('FACE_THRESHOLD', 0.6)))
        self.max_batch_size = int(os.getenv('MAX_BATCH_SIZE', 10))
    
    def process_batch(self):
        """Main processing function"""
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        print(f"[{datetime.now()}] Starting batch processing: {batch_id}")
        
        # Initialize batch statistics
        batch_stats = {
            'batch_id': batch_id,
            'images_processed': 0,
            'faces_detected': 0,
            'new_persons': 0,
            'existing_persons': 0,
            'processing_time': 0,
            'status': 'success'
        }
        
        try:
            # Step 1: Fetch recent images from Supabase
            recent_images = self.supabase_service.get_recent_images(self.max_batch_size)
            print(f"Found {len(recent_images)} recent images")
            
            if not recent_images:
                print("No new images to process")
                return
            
            # Step 2: Process each image
            for image_info in recent_images:
                try:
                    image_name = image_info['name']
                    local_path = f"temp_images/{image_name}"
                    
                    # Download image
                    if self.supabase_service.download_image(image_name, local_path):
                        batch_stats['images_processed'] += 1
                        
                        # Step 3: Detect and crop faces
                        cropped_faces = self.face_service.detect_and_crop_faces(local_path)
                        batch_stats['faces_detected'] += len(cropped_faces)
                        
                        # Step 4: Process each detected face
                        for crop_path, face_encoding in cropped_faces:
                            # Insert into temp_faces table
                            image_url = self.supabase_service.get_image_url(image_name)
                            self.mysql_service.insert_temp_face(image_url, crop_path, face_encoding)
                        
                        # Clean up original downloaded image
                        if os.path.exists(local_path):
                            os.remove(local_path)
                            
                except Exception as e:
                    print(f"Error processing image {image_info.get('name', 'unknown')}: {e}")
                    continue
            
            # Step 5: Process temp faces (compare embeddings)
            self.process_temp_faces(batch_stats)
            
            # Step 6: Clear processed temp faces
            self.mysql_service.clear_processed_temp_faces()
            
            # Step 7: Clean up temp files
            self.face_service.cleanup_temp_files()
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            batch_stats['status'] = 'error'
        
        # Calculate processing time
        batch_stats['processing_time'] = time.time() - start_time
        
        # Log batch statistics
        self.mysql_service.log_processing_batch(batch_stats)
        
        print(f"Batch completed in {batch_stats['processing_time']:.2f}s:")
        print(f"  📸 Images processed: {batch_stats['images_processed']}")
        print(f"  👤 Faces detected: {batch_stats['faces_detected']}")
        print(f"  ✨ New persons: {batch_stats['new_persons']}")  
        print(f"  🔄 Returning persons: {batch_stats['existing_persons']}")
        print(f"  📊 Status: {batch_stats['status'].upper()}")
    
    def process_temp_faces(self, batch_stats):
        """Process temporary faces and update unique persons"""
        # Get all unprocessed temp faces
        temp_faces = self.mysql_service.get_unprocessed_temp_faces()
        
        if not temp_faces:
            return
        
        # Get all known encodings
        known_encodings = self.mysql_service.get_all_unique_encodings()
        
        for temp_face in temp_faces:
            try:
                face_encoding = temp_face['face_encoding']
                if not face_encoding:
                    continue
                
                import numpy as np
                face_encoding_array = np.array(face_encoding)
                
                # Try to find a match
                matching_person_id = self.face_service.find_matching_person(
                    face_encoding_array, known_encodings
                )
                
                if matching_person_id:
                    # Update existing person
                    self.mysql_service.update_person_visit(matching_person_id)
                    batch_stats['existing_persons'] += 1
                    print(f"Updated visit for person: {matching_person_id}")
                else:
                    # Create new person
                    person_id = self.mysql_service.insert_new_person(
                        face_encoding, 
                        temp_face['original_image_url'], 
                        0.9
                    )
                    if person_id:
                        batch_stats['new_persons'] += 1
                        print(f"Created new person: {person_id}")
                        # Add to known encodings for this batch
                        known_encodings.append((person_id, face_encoding))
                
                # Mark as processed
                self.mysql_service.mark_temp_face_processed(temp_face['id'])
                
            except Exception as e:
                print(f"Error processing temp face {temp_face['id']}: {e}")
                continue
    
    def print_statistics(self):
        """Print current system statistics"""
        stats = self.mysql_service.get_statistics()
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        print(f"Total Unique Persons: {stats.get('total_unique_persons', 0)}")
        print(f"Total Visits: {stats.get('total_visits', 0)}")
        print(f"Recent Visitors (24h): {stats.get('recent_visitors_24h', 0)}")
        print("="*50 + "\n")
    
    def run_continuous(self):
        """Run the system continuously"""
        print("Starting Face Recognition System...")
        
        # Schedule the processing
        interval = int(os.getenv('PROCESSING_INTERVAL', 5))
        schedule.every(interval).seconds.do(self.process_batch)
        schedule.every(30).seconds.do(self.print_statistics)
        
        # Initial statistics
        self.print_statistics()
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run_continuous()