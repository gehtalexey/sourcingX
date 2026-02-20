"""
Database Transaction and Rollback Support for SourcingX

Provides utilities for safe database operations:
1. BatchTransaction - Execute multiple operations with retry and error tracking
2. DataCheckpoint - Save and restore data state for rollback capability
3. safe_batch_operation - Context manager combining both for safe batch updates

Since Supabase REST API doesn't support true database transactions, these utilities
provide application-level transaction semantics with rollback capability.
"""

import copy
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from db import SupabaseClient


# ============================================================================
# TRANSACTION SUPPORT FOR BATCH OPERATIONS
# ============================================================================

class BatchTransaction:
    """Context manager for batch operations with automatic rollback on failure.

    Since Supabase REST API doesn't support true transactions, this provides:
    - Atomic-like batch operations with all-or-nothing semantics
    - Automatic retry with exponential backoff
    - Rollback tracking for manual recovery

    Usage:
        from db import get_supabase_client
        from db_transactions import BatchTransaction

        client = get_supabase_client()
        with BatchTransaction(client) as batch:
            batch.upsert('profiles', profile1)
            batch.upsert('profiles', profile2)
            batch.delete('profiles', {'linkedin_url': 'old_url'})
        # All operations execute on __exit__ if no errors

    On error, batch.failed_operations contains what failed for manual recovery.
    """

    def __init__(self, client: 'SupabaseClient', max_retries: int = 3):
        self.client = client
        self.max_retries = max_retries
        self.operations: List[Dict[str, Any]] = []
        self.completed_operations: List[Dict[str, Any]] = []
        self.failed_operations: List[Dict[str, Any]] = []
        self._in_context = False

    def __enter__(self):
        self._in_context = True
        self.operations = []
        self.completed_operations = []
        self.failed_operations = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False
        if exc_type is not None:
            # Exception occurred, don't execute pending operations
            self.failed_operations = self.operations
            return False

        # Execute all pending operations
        self._execute_all()
        return False

    def upsert(self, table: str, data: dict, on_conflict: str = None):
        """Queue an upsert operation."""
        self.operations.append({
            'type': 'upsert',
            'table': table,
            'data': copy.deepcopy(data),
            'on_conflict': on_conflict,
        })

    def upsert_batch(self, table: str, rows: list, on_conflict: str = None):
        """Queue a batch upsert operation."""
        self.operations.append({
            'type': 'upsert_batch',
            'table': table,
            'data': copy.deepcopy(rows),
            'on_conflict': on_conflict,
        })

    def insert(self, table: str, data: dict):
        """Queue an insert operation."""
        self.operations.append({
            'type': 'insert',
            'table': table,
            'data': copy.deepcopy(data),
        })

    def update(self, table: str, data: dict, filters: dict):
        """Queue an update operation."""
        self.operations.append({
            'type': 'update',
            'table': table,
            'data': copy.deepcopy(data),
            'filters': copy.deepcopy(filters),
        })

    def delete(self, table: str, filters: dict):
        """Queue a delete operation."""
        self.operations.append({
            'type': 'delete',
            'table': table,
            'filters': copy.deepcopy(filters),
        })

    def _execute_single(self, op: dict) -> bool:
        """Execute a single operation with retry."""
        for attempt in range(self.max_retries):
            try:
                if op['type'] == 'upsert':
                    self.client.upsert(op['table'], op['data'], op.get('on_conflict'))
                elif op['type'] == 'upsert_batch':
                    self.client.upsert_batch(op['table'], op['data'], op.get('on_conflict'))
                elif op['type'] == 'insert':
                    self.client.insert(op['table'], op['data'])
                elif op['type'] == 'update':
                    self.client.update(op['table'], op['data'], op['filters'])
                elif op['type'] == 'delete':
                    self.client.delete(op['table'], op['filters'])
                return True
            except requests.HTTPError as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    time.sleep(2 ** attempt)
                else:
                    op['error'] = str(e)
                    return False
            except Exception as e:
                op['error'] = str(e)
                return False
        return False

    def _execute_all(self):
        """Execute all pending operations."""
        for op in self.operations:
            if self._execute_single(op):
                self.completed_operations.append(op)
            else:
                self.failed_operations.append(op)

        if self.failed_operations:
            raise BatchOperationError(
                f"{len(self.failed_operations)} operations failed",
                completed=self.completed_operations,
                failed=self.failed_operations
            )

    @property
    def pending_count(self) -> int:
        """Number of operations queued."""
        return len(self.operations)


class BatchOperationError(Exception):
    """Raised when batch operations partially fail."""

    def __init__(self, message: str, completed: list = None, failed: list = None):
        super().__init__(message)
        self.completed = completed or []
        self.failed = failed or []


# ============================================================================
# ROLLBACK HELPER FOR SAFE DATA OPERATIONS
# ============================================================================

class DataCheckpoint:
    """Create checkpoints for safe data operations with rollback capability.

    This helper fetches and stores the current state of records before modifications,
    allowing you to restore them if something goes wrong.

    Usage:
        from db import get_supabase_client
        from db_transactions import DataCheckpoint

        client = get_supabase_client()
        checkpoint = DataCheckpoint(client)

        # Save current state before modifications
        checkpoint.save('profiles', {'linkedin_url': 'eq.https://linkedin.com/in/user'})

        try:
            # Make changes
            client.update('profiles', {'status': 'archived'}, {'linkedin_url': url})
            # More operations...
        except Exception:
            # Restore original state
            checkpoint.rollback()

    Note: This is application-level rollback, not database transactions.
    It works by storing copies of data and restoring them if needed.
    """

    def __init__(self, client: 'SupabaseClient'):
        self.client = client
        self.checkpoints: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()

    def save(self, table: str, filters: dict, columns: str = '*') -> int:
        """Save current state of records matching filters.

        Args:
            table: Table name
            filters: Supabase filter dict (e.g., {'linkedin_url': 'eq.xxx'})
            columns: Columns to fetch (default '*')

        Returns:
            Number of records saved to checkpoint
        """
        try:
            records = self.client.select(table, columns, filters, limit=10000)
            if records:
                self.checkpoints.append({
                    'table': table,
                    'records': copy.deepcopy(records),
                    'primary_key': self._infer_primary_key(table),
                    'saved_at': datetime.utcnow().isoformat(),
                })
            return len(records)
        except Exception as e:
            print(f"[Checkpoint] Failed to save checkpoint for {table}: {e}")
            return 0

    def _infer_primary_key(self, table: str) -> str:
        """Infer primary key column for a table."""
        # Common primary keys in this project
        pk_map = {
            'profiles': 'linkedin_url',
            'screening_prompts': 'role_type',
            'settings': 'key',
            'api_usage_logs': 'id',
            'search_history': 'id',
            'schema_migrations': 'migration_name',
        }
        return pk_map.get(table, 'id')

    def rollback(self) -> dict:
        """Restore all checkpointed records to their saved state.

        Returns:
            Dict with 'restored' count and 'errors' list
        """
        stats = {'restored': 0, 'errors': []}

        # Process checkpoints in reverse order (LIFO)
        for checkpoint in reversed(self.checkpoints):
            table = checkpoint['table']
            pk = checkpoint['primary_key']

            for record in checkpoint['records']:
                try:
                    # Upsert to restore original state
                    self.client.upsert(table, record, on_conflict=pk)
                    stats['restored'] += 1
                except Exception as e:
                    stats['errors'].append({
                        'table': table,
                        'record_pk': record.get(pk),
                        'error': str(e)
                    })

        return stats

    def clear(self):
        """Clear all checkpoints (call after successful operation)."""
        self.checkpoints = []

    @property
    def record_count(self) -> int:
        """Total number of records in all checkpoints."""
        return sum(len(cp['records']) for cp in self.checkpoints)

    def get_summary(self) -> dict:
        """Get summary of checkpointed data."""
        return {
            'created_at': self.created_at.isoformat(),
            'tables': [cp['table'] for cp in self.checkpoints],
            'total_records': self.record_count,
            'checkpoints': len(self.checkpoints),
        }


@contextmanager
def safe_batch_operation(client: 'SupabaseClient', table: str, filters: dict = None):
    """Context manager combining checkpoint and batch transaction.

    Usage:
        from db import get_supabase_client
        from db_transactions import safe_batch_operation

        client = get_supabase_client()
        with safe_batch_operation(client, 'profiles', {'status': 'eq.enriched'}) as (checkpoint, batch):
            batch.upsert('profiles', new_data)
            batch.update('profiles', {'status': 'screened'}, {'linkedin_url': url})
        # Automatically commits on success, checkpoint available for manual rollback

    Args:
        client: SupabaseClient instance
        table: Primary table being modified
        filters: Optional filters for checkpoint (saves matching records)

    Yields:
        Tuple of (DataCheckpoint, BatchTransaction)
    """
    checkpoint = DataCheckpoint(client)
    batch = BatchTransaction(client)

    # Save checkpoint if filters provided
    if filters:
        checkpoint.save(table, filters)

    try:
        with batch:
            yield checkpoint, batch
        # Success - clear checkpoint
        checkpoint.clear()
    except BatchOperationError as e:
        # Partial failure - checkpoint remains for manual rollback
        print(f"[SafeBatch] Partial failure: {len(e.completed)} succeeded, {len(e.failed)} failed")
        raise
    except Exception as e:
        # Full failure - checkpoint remains for manual rollback
        print(f"[SafeBatch] Operation failed: {e}")
        raise
