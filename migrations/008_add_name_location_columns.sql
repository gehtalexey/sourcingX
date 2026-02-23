-- Migration: Add name and location columns to profiles table
-- These columns were missing but the code tries to save them

-- Add name column
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS name TEXT;

-- Add location column
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS location TEXT;

-- Add original_url column (for matching with loaded data)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS original_url TEXT;

-- Create index on name for searching
CREATE INDEX IF NOT EXISTS idx_profiles_name ON profiles(name);

-- Create index on location for filtering
CREATE INDEX IF NOT EXISTS idx_profiles_location ON profiles(location);
