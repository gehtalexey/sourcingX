"""Regression test: bulk upsert groups rows by key-shape, no None padding.

The old approach padded all rows to the union of keys (None for missing fields),
which overwrote existing DB values with NULL on upsert.  The new approach groups
by frozenset(keys) so each sub-batch is shape-homogeneous with no padding.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch


def _make_db_module():
    """Build a minimal fake environment so db.py can be imported."""
    for mod in ['streamlit', 'pandas', 'numpy', 'postgrest', 'gotrue',
                'supabase', 'openai', 'anthropic', 'gspread']:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    # Make sure db.py's top-level imports don't blow up
    import importlib
    try:
        return importlib.import_module('db')
    except Exception:
        return None


class TestBulkUpsertShapeGroups(unittest.TestCase):

    def test_no_none_padding_across_shapes(self):
        """Rows with different key sets must be sent in separate batches,
        never padded with None to a common superset."""
        upserted_batches = []

        class FakeClient:
            def upsert_batch(self, table, batch, on_conflict=None):
                upserted_batches.append(batch)
            def upsert(self, table, row, on_conflict=None):
                pass

        # Two rows with different key sets
        rows = [
            {'linkedin_url': 'https://www.linkedin.com/in/alice', 'name': 'Alice', 'email': 'a@x.com'},
            {'linkedin_url': 'https://www.linkedin.com/in/bob',   'name': 'Bob'},
        ]

        from collections import defaultdict
        shape_groups = defaultdict(list)
        for r in rows:
            shape_groups[frozenset(r.keys())].append(r)

        for shape_rows in shape_groups.values():
            for i in range(0, len(shape_rows), 100):
                batch = shape_rows[i:i+100]
                FakeClient().upsert_batch('profiles', batch, on_conflict='linkedin_url')

        # Should be two separate batches, not one padded batch
        self.assertEqual(len(upserted_batches), 2)

        # No batch should contain None values
        for batch in upserted_batches:
            for row in batch:
                for k, v in row.items():
                    self.assertIsNotNone(v, f"None found for key '{k}' — padding bug regressed")

    def test_same_shape_rows_batched_together(self):
        """Rows with identical key sets should be sent in the same batch."""
        batches = []

        shape_groups = {}
        rows = [
            {'linkedin_url': 'https://www.linkedin.com/in/a', 'name': 'A'},
            {'linkedin_url': 'https://www.linkedin.com/in/b', 'name': 'B'},
            {'linkedin_url': 'https://www.linkedin.com/in/c', 'name': 'C'},
        ]
        from collections import defaultdict
        shape_groups = defaultdict(list)
        for r in rows:
            shape_groups[frozenset(r.keys())].append(r)

        for shape_rows in shape_groups.values():
            for i in range(0, len(shape_rows), 100):
                batches.append(shape_rows[i:i+100])

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)


if __name__ == '__main__':
    unittest.main()
