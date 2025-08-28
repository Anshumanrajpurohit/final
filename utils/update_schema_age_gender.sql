-- Add age and gender columns to unique_persons table if they don't exist
ALTER TABLE unique_persons
ADD COLUMN IF NOT EXISTS age INT NULL,
ADD COLUMN IF NOT EXISTS gender VARCHAR(1) NULL,
ADD COLUMN IF NOT EXISTS age_updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP;
