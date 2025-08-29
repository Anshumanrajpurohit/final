"""
Enhanced Face Recognition System (resilient)
- Uses MediaPipe via EnhancedFaceService
- Safe against missing optional methods
- No hard-coded local image paths
"""
import os
import time
import logging
import uuid
import datetime
import signal
from typing import Dict, Any

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/face_recognition.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Services
from services.enhanced_face_service import EnhancedFaceService
from services.enhanced_mysql_service import EnhancedMySQLService
from services.supabase_service import SupabaseService
from utils.duplicate_detector import DuplicateDetector
from utils.performance_optimizer import PerformanceOptimizer


class EnhancedFaceRecognitionSystem:
    def __init__(self):
        self.running = True
        self.batch_size = 10
        self.processing_interval = 5
        self.config = self._load_config()

        os.makedirs("temp_images", exist_ok=True)

        # Init services (in dependency order)
        self.face_service = EnhancedFaceService()
        self.mysql_service = EnhancedMySQLService(self.config)
        self.supabase_service = SupabaseService()
        self.duplicate_detector = DuplicateDetector(self.mysql_service)
        self.performance_optimizer = PerformanceOptimizer()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.stats = {
            "start_time": datetime.datetime.now(),
            "total_batches": 0,
            "total_images": 0,
            "total_faces": 0,
            "total_new_persons": 0,
            "total_existing_persons": 0,
            "total_processing_time": 0.0,
        }

        logger.info("[START] Enhanced Face Recognition System initialized")
        logger.info(
            f"Configuration: batch_size={self.batch_size}, interval={self.processing_interval}s, "
            f"quality_check={self.config.get('USE_QUALITY_CHECK', True)}, faiss={self.config.get('USE_FAISS', True)}"
        )
        logger.info(
            f"Sleep Mode: enabled={self.config.get('USE_SLEEP_MODE', True)}, "
            f"sleep_duration={self.config.get('SLEEP_DURATION', 300)}s, max_duplicates={self.config.get('MAX_DUPLICATES', 3)}"
        )

    def _load_config(self) -> Dict[str, Any]:
        cfg = {
            "USE_QUALITY_CHECK": True,
            "USE_AGE_GENDER": True,
            "USE_FAISS": True,
            "USE_SLEEP_MODE": True,
            "SLEEP_DURATION": 300,
            "FACE_THRESHOLD": 0.6,
            "MAX_DUPLICATES": 3,
            "DB_HOST": "localhost",
            "DB_USER": "root",
            "DB_PASSWORD": "",
            "DB_NAME": "face_recognition_db",
        }
        env_path = ".env"
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        vs = v.strip()
                        l = vs.lower()
                        if l in ("true", "yes", "1"):
                            cfg[k] = True
                        elif l in ("false", "no", "0"):
                            cfg[k] = False
                        else:
                            try:
                                cfg[k] = float(vs) if "." in vs else int(vs)
                            except ValueError:
                                cfg[k] = vs
            except Exception as e:
                logger.warning(f"Config load warning: {e}")
        return cfg

    def _signal_handler(self, *_):
        logger.info("[STOP] System shutdown requested by user")
        self.running = False

    def start(self):
        logger.info("[START] Starting Enhanced Face Recognition System...")
        while self.running:
            try:
                batch_id = str(uuid.uuid4())
                logger.info(f"[START] Starting enhanced batch processing: {batch_id}")

                batch_stats = {
                    "batch_id": batch_id,
                    "start_time": datetime.datetime.now(),
                    "images_processed": 0,
                    "faces_detected": 0,
                    "new_persons": 0,
                    "existing_persons": 0,
                    "duplicate_events": 0,
                }

                # Sleep-mode check (supports multiple APIs)
                sleep_active = False
                sleep_remaining = 0
                if getattr(self.duplicate_detector, "is_in_sleep_mode", None):
                    sleep_active = self.duplicate_detector.is_in_sleep_mode()
                    if getattr(self.duplicate_detector, "get_sleep_remaining", None):
                        sleep_remaining = self.duplicate_detector.get_sleep_remaining()
                elif getattr(self.duplicate_detector, "is_sleeping", None):
                    sleep_active = self.duplicate_detector.is_sleeping()
                    if getattr(self.duplicate_detector, "get_sleep_status", None):
                        try:
                            status = self.duplicate_detector.get_sleep_status()
                            sleep_remaining = int(status.get('remaining_seconds', 0))
                        except Exception:
                            sleep_remaining = 0
                if sleep_active:
                    logger.info(f"[SLEEP] System in sleep mode - {sleep_remaining}s remaining")
                    time.sleep(min(5, max(1, sleep_remaining)))
                    continue

                t0 = time.time()
                self._process_images(batch_stats)
                dt = time.time() - t0

                self.stats["total_batches"] += 1
                self.stats["total_images"] += batch_stats["images_processed"]
                self.stats["total_faces"] += batch_stats["faces_detected"]
                self.stats["total_new_persons"] += batch_stats["new_persons"]
                self.stats["total_existing_persons"] += batch_stats["existing_persons"]
                self.stats["total_processing_time"] += dt

                if batch_stats["images_processed"] > 0:
                    logger.info("============================================================")
                    logger.info("[BATCH] BATCH PROCESSING SUMMARY")
                    logger.info("============================================================")
                    logger.info(f"[ID] Batch ID: {batch_id}")
                    logger.info(f"[IMAGES] Images processed: {batch_stats['images_processed']}")
                    logger.info(f"[FACES] Faces detected: {batch_stats['faces_detected']}")
                    logger.info(f"[NEW] New persons: {batch_stats['new_persons']}")
                    logger.info(f"[RETURN] Returning persons: {batch_stats['existing_persons']}")
                    logger.info(f"[TIME] Processing time: {dt:.2f}s")

                    mem_usage = 0.0
                    if getattr(self.performance_optimizer, "get_memory_usage", None):
                        mem_usage = self.performance_optimizer.get_memory_usage()
                    logger.info(f"[MEMORY] Memory usage: {mem_usage:.1f} MB")

                    if getattr(self.performance_optimizer, "get_memory_increase", None):
                        logger.info(f"[INCREASE] Memory increase: {self.performance_optimizer.get_memory_increase():.1f} MB")
                    logger.info("============================================================")

                if batch_stats["duplicate_events"] > 0:
                    logger.info(f"[DUPLICATE] Batch had {batch_stats['duplicate_events']} duplicate events")

                time.sleep(self.processing_interval)

                if self.stats["total_batches"] % 12 == 0:
                    self._print_statistics()
            except Exception as e:
                logger.exception(f"Error in main loop: {e}")
                time.sleep(5)

        self._cleanup()

    def _ensure_logger(self):
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(__name__)

    def _process_images(self, batch_stats: Dict[str, Any]):
        self._ensure_logger()
        try:
            mem_before = 0.0
            if getattr(self.performance_optimizer, "get_memory_usage", None):
                mem_before = self.performance_optimizer.get_memory_usage()
            logger.info(f"Memory before processing: {mem_before:.1f} MB")

            # Prefer Supabase
            recent_images = []
            get_recent = getattr(self.supabase_service, "get_recent_images", None)
            if get_recent:
                try:
                    recent_images = get_recent(limit=self.batch_size) or []
                except Exception as e:
                    logger.warning(f"Supabase get_recent_images failed: {e}")

            # Optional fallback: local temp_images (no hard-coded image)
            if not recent_images:
                try:
                    local_files = [
                        f for f in os.listdir("temp_images")
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    local_files.sort()
                    recent_images = [{"name": f, "_local": True} for f in local_files[: self.batch_size]]
                    if recent_images:
                        logger.info(f"[FALLBACK] Using {len(recent_images)} local images from temp_images")
                except Exception as e:
                    logger.warning(f"Local fallback scan failed: {e}")
                    recent_images = []

            if not recent_images:
                logger.info("No new images found")
                return False

            logger.info(f"[FOUND] Found {len(recent_images)} images to process")
            logger.info(f"[AGE/GENDER] Age and gender detection {'enabled' if self.config.get('USE_AGE_GENDER', True) else 'disabled'}")

            for img_info in recent_images:
                image_name = img_info.get("name") or img_info.get("filename") or ""
                if not image_name:
                    continue
                is_local = bool(img_info.get("_local"))

                # Duplicate check (if available)
                is_dup = False
                image_hash = None
                if getattr(self.duplicate_detector, "generate_image_hash", None):
                    try:
                        # For remote images, we haven't downloaded yet; use name as proxy hash
                        local_tmp = os.path.join("temp_images", image_name)
                        image_hash = self.duplicate_detector.generate_image_hash(local_tmp) if os.path.exists(local_tmp) else image_name
                    except Exception:
                        image_hash = image_name
                if getattr(self.duplicate_detector, "is_image_processed", None):
                    try:
                        is_dup = self.duplicate_detector.is_image_processed(image_hash or image_name, image_name)
                    except Exception as e:
                        logger.warning(f"Duplicate check failed: {e}")
                if is_dup:
                    batch_stats["duplicate_events"] += 1
                    logger.info(f"[DUPLICATE] Image already processed: {image_name}")
                    continue

                local_path = os.path.join("temp_images", image_name)
                downloaded = False
                if not is_local:
                    if getattr(self.supabase_service, "download_image", None):
                        downloaded = self.supabase_service.download_image(image_name, local_path)
                    if not downloaded:
                        logger.warning(f"Failed to download image: {image_name}")
                        continue
                else:
                    if not os.path.exists(local_path):
                        logger.warning(f"Local file missing: {local_path}")
                        continue

                try:
                    # Detect and process faces
                    results = self.face_service.process_batch(local_path)
                    # Count only successful encodings
                    success_faces = sum(1 for r in results if r.get("success"))
                    face_count = success_faces
                    # Update batch stats
                    batch_stats["faces_detected"] += face_count
                    batch_stats["images_processed"] += 1
                    self.logger.info(f"[ENCODINGS] Successful face encodings: {face_count}")
                    # Insert to DB per face (if available)
                    for r in results:
                        if not r.get("success"):
                            continue
                        data = r.get("data", {})
                        face_encoding = data.get("face_encoding")
                        if face_encoding is None:
                            continue
                        if hasattr(face_encoding, "tolist"):
                            face_encoding = face_encoding.tolist()

                        insert = getattr(self.mysql_service, "insert_face_data", None)
                        if insert:
                            try:
                                image_url = ""
                                if getattr(self.supabase_service, "get_image_url", None) and not is_local:
                                    image_url = self.supabase_service.get_image_url(image_name)

                                res = insert(
                                    image_url=image_url,
                                    image_path=local_path,
                                    face_encoding=face_encoding,
                                    crop_path=data.get("crop_path", ""),
                                    quality_score=data.get("quality_score", 0.0),
                                    age=data.get("age_range", "unknown"),
                                    gender=data.get("gender", "unknown"),
                                    x=data.get("x", 0),
                                    y=data.get("y", 0),
                                    width=data.get("w", 0),
                                    height=data.get("h", 0),
                                )
                                if res and res.get("new_person"):
                                    batch_stats["new_persons"] += 1
                                else:
                                    batch_stats["existing_persons"] += 1
                            except Exception as e:
                                logger.exception(f"DB insert failed: {e}")

                    # Mark processed (if available)
                    if getattr(self.duplicate_detector, "mark_image_processed", None):
                        try:
                            image_url = ""
                            if getattr(self.supabase_service, "get_image_url", None) and not is_local:
                                image_url = self.supabase_service.get_image_url(image_name)
                            # Compute actual hash if file exists now
                            if getattr(self.duplicate_detector, "generate_image_hash", None) and os.path.exists(local_path):
                                try:
                                    image_hash = self.duplicate_detector.generate_image_hash(local_path)
                                except Exception:
                                    pass
                            self.duplicate_detector.mark_image_processed(
                                image_hash or image_name, image_name, image_url, face_count
                            )
                        except Exception as e:
                            logger.warning(f"Mark processed failed: {e}")

                except Exception as e:
                    logger.exception(f"Error processing image {image_name}: {e}")
                finally:
                    # Only delete downloaded files (never delete local originals)
                    if downloaded:
                        try:
                            if os.path.exists(local_path):
                                os.remove(local_path)
                        except Exception as e:
                            logger.warning(f"Temp cleanup failed for {local_path}: {e}")

            mem_after = 0.0
            if getattr(self.performance_optimizer, "get_memory_usage", None):
                mem_after = self.performance_optimizer.get_memory_usage()
            logger.info(f"Memory after processing: {mem_after:.1f} MB")
            return True
        except Exception as e:
            logger.exception(f"Error in _process_images: {e}")
            return False

    def _print_statistics(self):
        try:
            now = datetime.datetime.now()
            uptime = now - self.stats["start_time"]

            logger.info("\n======================================================================")
            logger.info("[STATS] ENHANCED SYSTEM STATISTICS")
            logger.info("======================================================================")

            # DB stats (optional)
            total_persons = 0
            total_visits = 0
            recent_visitors = 0
            if getattr(self.mysql_service, "get_total_persons", None):
                total_persons = self.mysql_service.get_total_persons()
            if getattr(self.mysql_service, "get_total_visits", None):
                total_visits = self.mysql_service.get_total_visits()
            if getattr(self.mysql_service, "get_recent_visitors", None):
                recent_visitors = self.mysql_service.get_recent_visitors(hours=24)
            logger.info(f"[DB] Total Unique Persons: {total_persons}")
            logger.info(f"[DB] Total Visits: {total_visits}")
            logger.info(f"[DB] Recent Visitors (24h): {recent_visitors}")

            # System stats
            avg_time = self.stats["total_processing_time"] / max(1, self.stats["total_batches"])
            logger.info("\n[SYSTEM] SYSTEM STATISTICS:")
            logger.info(f"  Total Batches: {self.stats['total_batches']}")
            logger.info(f"  Total Images Processed: {self.stats['total_images']}")
            logger.info(f"  Total Faces Detected: {self.stats['total_faces']}")
            logger.info(f"  Total New Persons: {self.stats['total_new_persons']}")
            logger.info(f"  Total Existing Persons: {self.stats['total_existing_persons']}")
            logger.info(f"  Total Processing Time: {self.stats['total_processing_time']:.2f}s")
            logger.info(f"  System Uptime: {uptime}")

            # Performance stats (optional)
            mem_usage = 0.0
            mem_percent = 0.0
            if getattr(self.performance_optimizer, "get_memory_usage", None):
                mem_usage = self.performance_optimizer.get_memory_usage()
            if getattr(self.performance_optimizer, "get_memory_percent", None):
                mem_percent = self.performance_optimizer.get_memory_percent()
            logger.info("\n[PERF] PERFORMANCE STATISTICS:")
            logger.info(f"  Memory Usage: {mem_usage:.1f} MB ({mem_percent:.1f}%)")

            if getattr(self.mysql_service, "get_cache_hit_rate", None):
                try:
                    cache_hit = self.mysql_service.get_cache_hit_rate() * 100
                    logger.info(f"  Cache Hit Rate: {cache_hit:.2f}%")
                except Exception:
                    pass
            if getattr(self.mysql_service, "is_faiss_available", None):
                logger.info(f"  FAISS Available: {self.mysql_service.is_faiss_available()}")

            # Duplicate stats (optional)
            duplicates_24h = 0
            if getattr(self.duplicate_detector, "get_duplicate_count", None):
                try:
                    duplicates_24h = self.duplicate_detector.get_duplicate_count(None, hours=24)
                except Exception:
                    pass
            sleep_status = False
            if getattr(self.duplicate_detector, "is_in_sleep_mode", None):
                sleep_status = self.duplicate_detector.is_in_sleep_mode()
            sleep_remaining = 0
            if getattr(self.duplicate_detector, "get_sleep_remaining", None):
                sleep_remaining = self.duplicate_detector.get_sleep_remaining()

            logger.info("\n[DUPLICATE] DUPLICATE DETECTION STATISTICS:")
            logger.info(f"  Total Duplicates (24h): {duplicates_24h}")
            logger.info(f"  Sleep Mode: {'ACTIVE' if sleep_status else 'INACTIVE'}")
            if sleep_status:
                logger.info(f"  Remaining Sleep: {sleep_remaining}s")

            # Config
            logger.info("\n[CONFIG] CONFIGURATION:")
            logger.info(f"  Face Threshold: {self.config.get('FACE_THRESHOLD', 0.6)}")
            logger.info(f"  Max Batch Size: {self.batch_size}")
            logger.info(f"  Processing Interval: {self.processing_interval}s")
            logger.info(f"  Quality Check: {self.config.get('USE_QUALITY_CHECK', True)}")
            logger.info(f"  FAISS Enabled: {self.config.get('USE_FAISS', True)}")
            logger.info(f"  Sleep Mode: {self.config.get('USE_SLEEP_MODE', True)}")
            logger.info("======================================================================\n")
        except Exception as e:
            logger.exception(f"Error printing statistics: {e}")

    def _cleanup(self):
        logger.info("[CLEANUP] Performing cleanup...")
        try:
            self.face_service.cleanup_temp_files()
        except Exception as e:
            logger.warning(f"Temp cleanup error: {e}")
        try:
            self._print_statistics()
        except Exception:
            pass
        if getattr(self.performance_optimizer, "optimize_memory", None):
            try:
                self.performance_optimizer.optimize_memory()
            except Exception:
                pass
        logger.info("[DONE] System shutdown complete")


def main():
    system = EnhancedFaceRecognitionSystem()
    system.start()


if __name__ == "__main__":
    main()
