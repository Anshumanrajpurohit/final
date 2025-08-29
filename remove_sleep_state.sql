-- SQL Script to remove sleep state functionality
-- This script will disable sleep mode and clear all sleep-related tables

-- 1. Disable sleep mode in system_config
UPDATE system_config 
SET config_value = 'false' 
WHERE config_key = 'enable_sleep_mode';

-- 2. Set sleep duration to 0
UPDATE system_config 
SET config_value = '0' 
WHERE config_key = 'duplicate_sleep_duration';

-- 3. Set max duplicate threshold to a high value to prevent triggering
UPDATE system_config 
SET config_value = '100' 
WHERE config_key = 'max_duplicate_threshold';

-- 4. Set sleep mode backoff multiplier to 0
UPDATE system_config 
SET config_value = '0' 
WHERE config_key = 'sleep_mode_backoff_multiplier';

-- 5. Set min sleep duration to 0
UPDATE system_config 
SET config_value = '0' 
WHERE config_key = 'min_sleep_duration';

-- 6. Set max sleep duration to 0
UPDATE system_config 
SET config_value = '0' 
WHERE config_key = 'max_sleep_duration';

-- 7. Truncate the sleep_mode_state table
TRUNCATE TABLE sleep_mode_state;

-- 8. Truncate the duplicate_events table
TRUNCATE TABLE duplicate_events;

-- 9. Option: Drop the sleep_mode_state table entirely (uncomment if needed)
-- DROP TABLE IF EXISTS sleep_mode_state;

-- 10. Confirm settings
SELECT config_key, config_value 
FROM system_config 
WHERE config_key LIKE '%sleep%' OR config_key LIKE '%duplicate%';
