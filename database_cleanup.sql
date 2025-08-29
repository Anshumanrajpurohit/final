-- Face Recognition Database Cleanup SQL Script
-- This script removes unwanted tables and cleans up data issues

-- 1. Delete duplicate entries from person_embeddings table
-- Keep the first occurrence of each embedding, delete duplicates
DELETE FROM person_embeddings 
WHERE id IN (
    SELECT id FROM (
        SELECT id, embedding, 
        ROW_NUMBER() OVER(PARTITION BY embedding ORDER BY id) as row_num 
        FROM person_embeddings
    ) t 
    WHERE row_num > 1
);

-- 2. Remove images with zero face count from processed_images
-- These are likely failed face detection attempts
DELETE FROM processed_images 
WHERE face_count = 0;

-- 3. Check for empty tables and remove if needed
-- (Uncomment the appropriate lines below)

-- If customerdataextr is empty and not needed:
-- DROP TABLE IF EXISTS customerdataextr;

-- If temp_faces table has old unprocessed entries:
-- DELETE FROM temp_faces WHERE created_at < DATE_SUB(NOW(), INTERVAL 7 DAY);

-- 4. Check for system configuration and update face detection parameters
-- Update face detection threshold for better accuracy
UPDATE system_config 
SET config_value = '0.55' 
WHERE config_key = 'face_threshold';

-- Update duplicate settings to be more strict
UPDATE system_config 
SET config_value = '5' 
WHERE config_key = 'max_duplicate_threshold';

-- 5. Clean up sleep mode state if needed
-- Remove any stale sleep mode entries
DELETE FROM sleep_mode_state 
WHERE sleep_end_time < NOW();

-- 6. Optimize tables for better performance
OPTIMIZE TABLE person_embeddings;
OPTIMIZE TABLE processed_images;
OPTIMIZE TABLE unique_persons;
OPTIMIZE TABLE processing_logs;
