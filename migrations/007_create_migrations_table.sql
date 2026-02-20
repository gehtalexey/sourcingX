-- Migration: Create migrations tracking table
-- This table tracks which migrations have been applied to the database
--
-- Run this FIRST in Supabase SQL Editor to enable migration tracking

-- Create migrations table to track applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    checksum TEXT,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    applied_by TEXT DEFAULT current_user
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_migrations_name ON schema_migrations(migration_name);
CREATE INDEX IF NOT EXISTS idx_migrations_applied_at ON schema_migrations(applied_at);

-- Enable RLS
ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to view and manage migrations
CREATE POLICY "Allow authenticated access to schema_migrations" ON schema_migrations
    FOR ALL TO authenticated
    USING (true)
    WITH CHECK (true);

-- Grant permissions
GRANT ALL ON schema_migrations TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE schema_migrations_id_seq TO authenticated;

-- Comments for documentation
COMMENT ON TABLE schema_migrations IS 'Tracks which database migrations have been applied';
COMMENT ON COLUMN schema_migrations.migration_name IS 'Unique name of the migration file (e.g., 007_create_migrations_table.sql)';
COMMENT ON COLUMN schema_migrations.checksum IS 'SHA256 hash of migration file contents for integrity verification';
COMMENT ON COLUMN schema_migrations.execution_time_ms IS 'Time taken to execute the migration in milliseconds';
COMMENT ON COLUMN schema_migrations.success IS 'Whether the migration completed successfully';
COMMENT ON COLUMN schema_migrations.error_message IS 'Error message if migration failed';
COMMENT ON COLUMN schema_migrations.applied_by IS 'Database user who applied the migration';

-- Insert record for this migration itself
INSERT INTO schema_migrations (migration_name, checksum)
VALUES ('007_create_migrations_table.sql', 'initial')
ON CONFLICT (migration_name) DO NOTHING;
