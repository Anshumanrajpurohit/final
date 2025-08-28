"""
Performance Optimizer for Face Recognition System
Includes batch processing, memory management, and efficient embedding comparison
"""
import numpy as np
import time
import psutil
import gc
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Try to import FAISS for efficient similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")

class PerformanceOptimizer:
    def __init__(self, max_memory_usage: float = 0.8, batch_size: int = 100):
        self.max_memory_usage = max_memory_usage
        self.batch_size = batch_size
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.processing_times = deque(maxlen=100)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_percent': 100 - process.memory_percent()
        }
    
    def should_garbage_collect(self) -> bool:
        """Check if garbage collection is needed"""
        memory_usage = self.monitor_memory_usage()
        return memory_usage['percent'] > (self.max_memory_usage * 100)
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection and clear caches"""
        self.logger.info("[GC] Performing garbage collection...")
        
        # Clear embedding cache
        self.embedding_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        memory_after = self.monitor_memory_usage()
        self.logger.info(f"[OK] Memory after GC: {memory_after['rss_mb']:.1f} MB")
    
    def batch_process_embeddings(self, embeddings: List[np.ndarray], 
                                batch_size: int = None) -> List[Any]:
        """Process embeddings in batches to manage memory"""
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # Process batch
            batch_results = self._process_embedding_batch(batch)
            results.extend(batch_results)
            
            # Check memory usage
            if self.should_garbage_collect():
                self.force_garbage_collection()
        
        return results
    
    def _process_embedding_batch(self, embeddings: List[np.ndarray]) -> List[Any]:
        """Process a single batch of embeddings"""
        # This is a placeholder - implement your specific processing logic
        return [self._process_single_embedding(emb) for emb in embeddings]
    
    def _process_single_embedding(self, embedding: np.ndarray) -> Any:
        """Process a single embedding"""
        # Placeholder - implement your specific logic
        return embedding.tolist()
    
    def create_faiss_index(self, embeddings: List[np.ndarray], 
                          dimension: int = 128) -> Optional[Any]:
        """Create FAISS index for efficient similarity search"""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, using brute force comparison")
            return None
        
        try:
            # Convert embeddings to float32
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Create FAISS index
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            self.logger.info(f"[OK] Created FAISS index with {len(embeddings)} embeddings")
            return index
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to create FAISS index: {e}")
            return None
    
    def search_faiss_index(self, index: Any, query_embedding: np.ndarray, 
                          k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index for similar embeddings"""
        if index is None:
            return np.array([]), np.array([])
        
        try:
            # Ensure query is float32
            query = np.array([query_embedding], dtype=np.float32)
            
            # Search
            distances, indices = index.search(query, k)
            
            return distances[0], indices[0]
            
        except Exception as e:
            self.logger.error(f"[ERROR] FAISS search failed: {e}")
            return np.array([]), np.array([])
    
    def efficient_embedding_comparison(self, query_embedding: np.ndarray,
                                     known_embeddings: List[Tuple[str, np.ndarray]],
                                     threshold: float = 0.6,
                                     use_faiss: bool = True) -> Optional[str]:
        """Efficiently compare embeddings using FAISS or optimized brute force"""
        
        if not known_embeddings:
            return None
        
        # Try FAISS first
        if use_faiss and FAISS_AVAILABLE and len(known_embeddings) > 100:
            return self._faiss_comparison(query_embedding, known_embeddings, threshold)
        else:
            return self._optimized_brute_force_comparison(query_embedding, known_embeddings, threshold)
    
    def _faiss_comparison(self, query_embedding: np.ndarray,
                         known_embeddings: List[Tuple[str, np.ndarray]],
                         threshold: float) -> Optional[str]:
        """Compare using FAISS index"""
        try:
            # Extract embeddings and IDs
            embeddings = [emb[1] for emb in known_embeddings]
            ids = [emb[0] for emb in known_embeddings]
            
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            if index is None:
                return self._optimized_brute_force_comparison(query_embedding, known_embeddings, threshold)
            
            # Search
            distances, indices = self.search_faiss_index(index, query_embedding, k=1)
            
            if len(distances) > 0 and distances[0] < threshold:
                return ids[indices[0]]
            
            return None
            
        except Exception as e:
            self.logger.error(f"FAISS comparison failed: {e}")
            return self._optimized_brute_force_comparison(query_embedding, known_embeddings, threshold)
    
    def _optimized_brute_force_comparison(self, query_embedding: np.ndarray,
                                        known_embeddings: List[Tuple[str, np.ndarray]],
                                        threshold: float) -> Optional[str]:
        """Optimized brute force comparison with early termination"""
        best_match_id = None
        best_distance = float('inf')
        
        # Convert query to numpy array if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        for person_id, known_embedding in known_embeddings:
            try:
                # Convert known embedding to numpy array if needed
                if not isinstance(known_embedding, np.ndarray):
                    known_embedding = np.array(known_embedding)
                
                # Calculate distance
                distance = np.linalg.norm(query_embedding - known_embedding)
                
                # Early termination if we find a very good match
                if distance < threshold * 0.5:
                    return person_id
                
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = person_id
                    
            except Exception as e:
                self.logger.warning(f"Error comparing with person {person_id}: {e}")
                continue
        
        return best_match_id
    
    def parallel_face_processing(self, image_paths: List[str], 
                               max_workers: int = 4) -> List[Tuple[str, List]]:
        """Process multiple images in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_image, path): path 
                for path in image_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append((path, result))
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {e}")
                    results.append((path, []))
        
        return results
    
    def _process_single_image(self, image_path: str) -> List:
        """Process a single image (placeholder - implement your face detection logic)"""
        # This should be implemented with your face detection logic
        # For now, return empty list
        return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        memory_usage = self.monitor_memory_usage()
        
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        cache_hit_rate = 0
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = self.cache_hits / total_cache_requests
        
        return {
            'memory_usage_mb': memory_usage['rss_mb'],
            'memory_percent': memory_usage['percent'],
            'avg_processing_time': avg_processing_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'faiss_available': FAISS_AVAILABLE
        }
    
    def log_performance_stats(self) -> None:
        """Log current performance statistics"""
        stats = self.get_performance_stats()
        
        self.logger.info("[STATS] Performance Statistics:")
        self.logger.info(f"  Memory Usage: {stats['memory_usage_mb']:.1f} MB ({stats['memory_percent']:.1f}%)")
        self.logger.info(f"  Avg Processing Time: {stats['avg_processing_time']:.3f}s")
        self.logger.info(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        self.logger.info(f"  FAISS Available: {stats['faiss_available']}")
    
    def optimize_for_large_dataset(self, dataset_size: int) -> Dict[str, Any]:
        """Provide optimization recommendations for large datasets"""
        recommendations = {
            'use_faiss': dataset_size > 1000,
            'batch_size': min(100, max(10, dataset_size // 100)),
            'max_workers': min(4, max(1, dataset_size // 500)),
            'enable_caching': dataset_size > 100,
            'memory_limit_mb': dataset_size * 0.1  # Rough estimate
        }
        
        self.logger.info("[OPTIMIZE] Optimization Recommendations:")
        for key, value in recommendations.items():
            self.logger.info(f"  {key}: {value}")
        
        return recommendations
