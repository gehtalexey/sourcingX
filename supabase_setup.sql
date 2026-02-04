-- LinkedIn Enricher Database Schema (v2 - Crustdata Only)
-- Run this in Supabase SQL Editor for fresh setup
--
-- This schema stores only Crustdata-enriched profiles.
-- PhantomBuster data is used for UI preview only, not stored in DB.

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main profiles table - stores Crustdata enriched profiles only
CREATE TABLE profiles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  linkedin_url TEXT UNIQUE NOT NULL,

  -- Basic info (from Crustdata)
  first_name TEXT,
  last_name TEXT,
  headline TEXT,
  location TEXT,
  summary TEXT,

  -- Current position (extracted for easy querying)
  current_title TEXT,
  current_company TEXT,
  current_years_in_role NUMERIC,
  current_years_at_company NUMERIC,

  -- Full arrays from Crustdata (for advanced filtering)
  positions JSONB,          -- [{title, company_name, start_date, end_date, ...}, ...]
  education_list JSONB,     -- [{school, degree, field_of_study, ...}, ...]
  skills_list JSONB,        -- ["skill1", "skill2", ...]

  -- Flattened for display
  skills TEXT,              -- Comma-separated skills string
  education TEXT,           -- Most recent school name

  -- Profile metadata
  connections_count INTEGER,
  followers_count INTEGER,
  profile_picture_url TEXT,

  -- Raw Crustdata response (full backup)
  raw_data JSONB,

  -- Screening results (from OpenAI)
  screening_score INTEGER,
  screening_fit_level TEXT,  -- 'Strong Fit', 'Good Fit', 'Partial Fit', 'Not a Fit'
  screening_summary TEXT,
  screening_reasoning TEXT,

  -- Email enrichment
  email TEXT,
  email_source TEXT,  -- 'salesql', 'crustdata', 'manual'

  -- Pipeline status (no 'scraped' - only enriched profiles stored)
  status TEXT DEFAULT 'enriched' CHECK (status IN ('enriched', 'screened', 'contacted', 'archived')),

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  enriched_at TIMESTAMPTZ DEFAULT NOW(),
  screened_at TIMESTAMPTZ,
  contacted_at TIMESTAMPTZ
);

-- Indexes for fast lookups
CREATE INDEX idx_profiles_linkedin_url ON profiles(linkedin_url);
CREATE INDEX idx_profiles_status ON profiles(status);
CREATE INDEX idx_profiles_screening_fit ON profiles(screening_fit_level);
CREATE INDEX idx_profiles_current_company ON profiles(current_company);
CREATE INDEX idx_profiles_enriched_at ON profiles(enriched_at);

-- GIN indexes for JSONB array searches
CREATE INDEX idx_profiles_positions ON profiles USING GIN (positions);
CREATE INDEX idx_profiles_skills ON profiles USING GIN (skills_list);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- View: Profiles needing screening (enriched but not screened)
CREATE VIEW profiles_needing_screening AS
SELECT *
FROM profiles
WHERE status = 'enriched'
  AND screening_score IS NULL;

-- View: Pipeline funnel stats
CREATE VIEW pipeline_stats AS
SELECT
  COUNT(*) FILTER (WHERE status = 'enriched' AND screening_score IS NULL) AS pending_screening,
  COUNT(*) FILTER (WHERE status = 'screened') AS screened,
  COUNT(*) FILTER (WHERE status = 'contacted') AS contacted,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Strong Fit') AS strong_fit,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Good Fit') AS good_fit,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Partial Fit') AS partial_fit,
  COUNT(*) FILTER (WHERE screening_fit_level = 'Not a Fit') AS not_a_fit,
  COUNT(*) AS total
FROM profiles;

-- API Usage Logs (for tracking Crustdata/OpenAI/SalesQL costs)
CREATE TABLE IF NOT EXISTS api_usage_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  provider TEXT NOT NULL,        -- 'crustdata', 'openai', 'salesql', 'phantombuster'
  operation TEXT NOT NULL,       -- 'enrich', 'screen', 'email_lookup', 'scrape'
  request_count INTEGER DEFAULT 1,
  credits_used NUMERIC,
  tokens_input INTEGER,
  tokens_output INTEGER,
  cost_usd NUMERIC(10, 6),
  status TEXT DEFAULT 'success', -- 'success' or 'error'
  error_message TEXT,
  response_time_ms INTEGER,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_usage_logs_provider ON api_usage_logs(provider);
CREATE INDEX idx_usage_logs_created_at ON api_usage_logs(created_at);
