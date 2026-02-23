"""
Database Migration Tracking Utility for SourcingX

Provides utilities to:
1. Track which migrations have been applied
2. Run pending migrations safely
3. Generate migration checksums for integrity
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from db import SupabaseClient, get_supabase_client


# ============================================================================
# MIGRATION TRACKING
# ============================================================================

def get_applied_migrations(client: SupabaseClient) -> list:
    """Get list of migrations that have been applied.

    Returns:
        List of dicts with migration_name, applied_at, success, checksum
    """
    try:
        result = client.select('schema_migrations', '*', limit=1000)
        return sorted(result, key=lambda x: x.get('migration_name', ''))
    except Exception as e:
        # Table might not exist yet
        print(f"[Migrations] Could not fetch applied migrations: {e}")
        return []


def is_migration_applied(client: SupabaseClient, migration_name: str) -> bool:
    """Check if a specific migration has been applied.

    Args:
        client: SupabaseClient instance
        migration_name: Name of the migration file (e.g., '007_create_migrations_table.sql')

    Returns:
        True if migration was applied successfully
    """
    try:
        result = client.select(
            'schema_migrations',
            'success',
            {'migration_name': f'eq.{migration_name}', 'success': 'eq.true'},
            limit=1
        )
        return len(result) > 0
    except Exception:
        return False


def get_migration_files(migrations_dir: str = None) -> list:
    """Get all migration files from the migrations directory.

    Args:
        migrations_dir: Path to migrations directory. Defaults to ./migrations

    Returns:
        List of (filename, filepath) tuples sorted by name
    """
    if migrations_dir is None:
        migrations_dir = Path(__file__).parent / 'migrations'
    else:
        migrations_dir = Path(migrations_dir)

    if not migrations_dir.exists():
        return []

    files = []
    for f in migrations_dir.glob('*.sql'):
        files.append((f.name, str(f)))

    return sorted(files, key=lambda x: x[0])


def calculate_checksum(filepath: str) -> str:
    """Calculate SHA256 checksum of a migration file.

    Args:
        filepath: Path to the migration file

    Returns:
        SHA256 hex digest
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_pending_migrations(client: SupabaseClient, migrations_dir: str = None) -> list:
    """Get list of migrations that haven't been applied yet.

    Args:
        client: SupabaseClient instance
        migrations_dir: Path to migrations directory

    Returns:
        List of (filename, filepath, checksum) tuples for pending migrations
    """
    applied = {m['migration_name'] for m in get_applied_migrations(client)}
    all_files = get_migration_files(migrations_dir)

    pending = []
    for filename, filepath in all_files:
        if filename not in applied:
            checksum = calculate_checksum(filepath)
            pending.append((filename, filepath, checksum))

    return pending


def record_migration(client: SupabaseClient, migration_name: str, checksum: str,
                     execution_time_ms: int = None, success: bool = True,
                     error_message: str = None) -> bool:
    """Record that a migration has been applied.

    Args:
        client: SupabaseClient instance
        migration_name: Name of the migration file
        checksum: SHA256 checksum of the file
        execution_time_ms: How long it took to execute
        success: Whether migration succeeded
        error_message: Error message if failed

    Returns:
        True if recorded successfully
    """
    try:
        data = {
            'migration_name': migration_name,
            'checksum': checksum,
            'success': success,
            'applied_at': datetime.utcnow().isoformat(),
        }
        if execution_time_ms is not None:
            data['execution_time_ms'] = execution_time_ms
        if error_message:
            data['error_message'] = error_message

        client.upsert('schema_migrations', data, on_conflict='migration_name')
        return True
    except Exception as e:
        print(f"[Migrations] Failed to record migration: {e}")
        return False


def verify_migration_integrity(client: SupabaseClient, migrations_dir: str = None) -> dict:
    """Verify that applied migrations haven't been modified.

    Compares stored checksums with current file checksums.

    Returns:
        Dict with 'valid', 'modified', 'missing' lists of migration names
    """
    applied = {m['migration_name']: m.get('checksum') for m in get_applied_migrations(client)}
    all_files = {name: filepath for name, filepath in get_migration_files(migrations_dir)}

    result = {'valid': [], 'modified': [], 'missing': []}

    for name, stored_checksum in applied.items():
        if name not in all_files:
            result['missing'].append(name)
        elif stored_checksum and stored_checksum != 'initial':
            current_checksum = calculate_checksum(all_files[name])
            if current_checksum == stored_checksum:
                result['valid'].append(name)
            else:
                result['modified'].append(name)
        else:
            # No checksum stored or 'initial' placeholder
            result['valid'].append(name)

    return result


def get_migration_status(client: SupabaseClient = None, migrations_dir: str = None) -> dict:
    """Get comprehensive status of all migrations.

    Returns:
        Dict with applied, pending, and integrity info
    """
    if client is None:
        client = get_supabase_client()
    if client is None:
        return {'error': 'Could not connect to database'}

    applied = get_applied_migrations(client)
    pending = get_pending_migrations(client, migrations_dir)
    integrity = verify_migration_integrity(client, migrations_dir)

    return {
        'applied': [m['migration_name'] for m in applied],
        'pending': [p[0] for p in pending],
        'integrity': integrity,
        'summary': {
            'total_applied': len(applied),
            'total_pending': len(pending),
            'modified_files': len(integrity['modified']),
            'missing_files': len(integrity['missing']),
        }
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_migration_status():
    """Print migration status to console."""
    client = get_supabase_client()
    if not client:
        print("[Migrations] ERROR: Could not connect to database")
        return

    status = get_migration_status(client)

    print("\n=== Migration Status ===\n")

    print(f"Applied: {status['summary']['total_applied']}")
    for name in status['applied']:
        print(f"  [OK] {name}")

    print(f"\nPending: {status['summary']['total_pending']}")
    for name in status['pending']:
        print(f"  [ ] {name}")

    if status['integrity']['modified']:
        print("\nWARNING: Modified migrations detected:")
        for name in status['integrity']['modified']:
            print(f"  [!] {name}")

    if status['integrity']['missing']:
        print("\nWARNING: Missing migration files:")
        for name in status['integrity']['missing']:
            print(f"  [?] {name}")

    print()


if __name__ == '__main__':
    print_migration_status()
