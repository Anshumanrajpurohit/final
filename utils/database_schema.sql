-- Table for storing extracted customer attributes
CREATE TABLE IF NOT EXISTS customerdataextr (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id VARCHAR(255) UNIQUE NOT NULL,
    age INT,
    gender VARCHAR(20),
    skin_tone VARCHAR(50),
    hair_status VARCHAR(20)
);
-- Face Recognition System Database Schema
-- This schema supports the complete face recognition workflow

-- Unique persons table (main table)
CREATE TABLE IF NOT EXISTS unique_persons (
    person_id VARCHAR(36) PRIMARY KEY,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    visit_count INT DEFAULT 1,
    confidence_score DECIMAL(3,2) DEFAULT 0.90,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_last_seen (last_seen),
    INDEX idx_visit_count (visit_count),
    INDEX idx_is_active (is_active)
);

-- Person embeddings table (separate for better performance)
CREATE TABLE IF NOT EXISTS person_embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person_id VARCHAR(36) NOT NULL,
    embedding LONGTEXT NOT NULL, -- JSON encoded face embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES unique_persons(person_id) ON DELETE CASCADE,
    INDEX idx_person_id (person_id)
);

-- Temporary faces table (for batch processing)
CREATE TABLE IF NOT EXISTS temp_faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    original_image_url TEXT NOT NULL,
    cropped_face_path VARCHAR(255) NOT NULL,
    face_encoding LONGTEXT NOT NULL, -- JSON encoded face embedding
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP NULL,
    INDEX idx_processed (processed),
    INDEX idx_created_at (created_at)
);

-- Processing logs table (for monitoring and debugging)
CREATE TABLE IF NOT EXISTS processing_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    batch_id VARCHAR(36) NOT NULL,
    images_processed INT DEFAULT 0,
    faces_detected INT DEFAULT 0,
    new_persons INT DEFAULT 0,
    existing_persons INT DEFAULT 0,
    processing_time DECIMAL(10,3) DEFAULT 0.000,
    status ENUM('success', 'error', 'partial') DEFAULT 'success',
    error_message TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_batch_id (batch_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Image metadata table (optional - for tracking original images)
CREATE TABLE IF NOT EXISTS image_metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_url TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size INT NULL,
    image_width INT NULL,
    image_height INT NULL,
    faces_detected INT DEFAULT 0,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP NULL,
    INDEX idx_filename (filename),
    INDEX idx_processed (processed),
    INDEX idx_created_at (created_at)
);

-- System configuration table (for dynamic settings)
CREATE TABLE IF NOT EXISTS system_config (
    config_key VARCHAR(50) PRIMARY KEY,
    config_value TEXT NOT NULL,
    description TEXT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- NEW: Processed images table for duplicate detection
CREATE TABLE IF NOT EXISTS processed_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_hash VARCHAR(64) NOT NULL,
    image_name VARCHAR(255) NOT NULL,
    image_url TEXT NOT NULL,
    face_count INT DEFAULT 0,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_image_hash (image_hash),
    INDEX idx_image_hash (image_hash),
    INDEX idx_processed_at (processed_at),
    INDEX idx_image_name (image_name)
);

-- NEW: Duplicate events table for sleep mode tracking
CREATE TABLE IF NOT EXISTS duplicate_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duplicate_type ENUM('image', 'face', 'batch') NOT NULL,
    duplicate_count INT DEFAULT 1,
    sleep_duration INT DEFAULT 300,
    description TEXT NULL,
    INDEX idx_event_time (event_time),
    INDEX idx_duplicate_type (duplicate_type)
);

-- NEW: Sleep mode state table
CREATE TABLE IF NOT EXISTS sleep_mode_state (
    id INT AUTO_INCREMENT PRIMARY KEY,
    is_sleeping BOOLEAN DEFAULT FALSE,
    sleep_start_time TIMESTAMP NULL,
    sleep_end_time TIMESTAMP NULL,
    sleep_duration INT DEFAULT 0,
    sleep_reason VARCHAR(100) NULL,
    wake_up_trigger VARCHAR(50) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_is_sleeping (is_sleeping),
    INDEX idx_sleep_start_time (sleep_start_time)
);

-- Insert default configuration
INSERT IGNORE INTO system_config (config_key, config_value, description) VALUES
('face_threshold', '0.6', 'Face recognition similarity threshold'),
('max_batch_size', '10', 'Maximum images to process per batch'),
('processing_interval', '5', 'Processing interval in seconds'),
('temp_file_retention_hours', '24', 'How long to keep temp files'),
('enable_debug_logging', 'true', 'Enable detailed debug logging'),
('duplicate_sleep_duration', '300', 'Sleep duration in seconds when duplicates detected'),
('max_duplicate_threshold', '3', 'Maximum duplicates before sleep mode'),
('image_hash_cache_duration', '86400', 'Image hash cache duration in seconds (24 hours)'),
('sleep_mode_backoff_multiplier', '1.5', 'Sleep duration multiplier for repeated duplicates'),
('enable_sleep_mode', 'true', 'Enable sleep mode for duplicate detection'),
('min_sleep_duration', '60', 'Minimum sleep duration in seconds'),
('max_sleep_duration', '3600', 'Maximum sleep duration in seconds');

-- Create views for easier querying
-- Ensure `is_active` column exists (safe for MySQL 8+)
ALTER TABLE unique_persons ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

CREATE OR REPLACE VIEW person_statistics AS
SELECT 
    COUNT(*) as total_unique_persons,
    SUM(visit_count) as total_visits,
    AVG(visit_count) as avg_visits_per_person,
    COUNT(CASE WHEN last_seen >= DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN 1 END) as recent_visitors_24h,
    COUNT(CASE WHEN last_seen >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as recent_visitors_7d,
    MAX(last_seen) as last_activity
FROM unique_persons 
WHERE is_active = TRUE;

-- Create view for processing performance
CREATE OR REPLACE VIEW processing_performance AS
SELECT 
    DATE(created_at) as processing_date,
    COUNT(*) as total_batches,
    AVG(processing_time) as avg_processing_time,
    SUM(images_processed) as total_images_processed,
    SUM(faces_detected) as total_faces_detected,
    SUM(new_persons) as total_new_persons,
    SUM(existing_persons) as total_existing_persons,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_batches,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as failed_batches
FROM processing_logs 
GROUP BY DATE(created_at)
ORDER BY processing_date DESC;
