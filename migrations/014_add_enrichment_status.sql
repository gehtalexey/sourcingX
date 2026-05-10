-- Migration: Add enrichment_status column to track Crustdata enrichment attempts
-- This allows tracking profiles that Crustdata doesn't have data for

-- Add enrichment_status column
-- Values: 'enriched' (success), 'not_found' (Crustdata has no data), 'failed' (API error)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS enrichment_status TEXT DEFAULT 'enriched';

-- Add index for filtering by enrichment status
CREATE INDEX IF NOT EXISTS idx_profiles_enrichment_status ON profiles(enrichment_status);

-- Add enrichment_attempted_at to track when we last tried to enrich
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS enrichment_attempted_at TIMESTAMPTZ;

-- Backfill existing profiles as 'enriched' (they all have raw_data from successful enrichment)
UPDATE profiles
SET enrichment_status = 'enriched',
    enrichment_attempted_at = enriched_at
WHERE enrichment_status IS NULL AND raw_data IS NOT NULL;
