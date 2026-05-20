"""Regression test: combined date range doesn't silently drop lower bound.

Old bug: when date_after AND date_before were both set, both were pushed as
simple conditions (enriched_at.gte.X and enriched_at.lte.Y).  The loop that
converts simple conditions to params dict wrote both to params['enriched_at'],
with the second write overwriting the first — lower bound silently dropped.

Fix: emit the combined range as a single and(gte,lte) expression so it goes
through the complex_conditions path, preserving both bounds.
"""
import unittest


def _build_date_conditions(filters):
    """Minimal reproduction of the search_profiles_boolean date-filter logic."""
    and_conditions = []
    params = {}

    if filters.get('date_after') and filters.get('date_before'):
        and_conditions.append(
            f"and(enriched_at.gte.{filters['date_after']},"
            f"enriched_at.lte.{filters['date_before']})"
        )
    elif filters.get('date_after'):
        params['enriched_at'] = f"gte.{filters['date_after']}"
    elif filters.get('date_before'):
        params['enriched_at'] = f"lte.{filters['date_before']}"

    complex_conditions = []
    for cond in and_conditions:
        if cond.startswith('or(') or cond.startswith('and('):
            complex_conditions.append(cond)
        else:
            parts = cond.split('.', 1)
            if len(parts) == 2:
                params[parts[0]] = parts[1]

    return params, complex_conditions


class TestDateRangeFilter(unittest.TestCase):

    def test_both_bounds_go_to_complex_conditions(self):
        params, complex_conds = _build_date_conditions({
            'date_after': '2024-01-01',
            'date_before': '2024-12-31',
        })
        # Neither bound should be in simple params (that's where they collide)
        self.assertNotIn('enriched_at', params)
        # Both should appear in a single and() complex condition
        self.assertEqual(len(complex_conds), 1)
        self.assertIn('enriched_at.gte.2024-01-01', complex_conds[0])
        self.assertIn('enriched_at.lte.2024-12-31', complex_conds[0])

    def test_only_date_after(self):
        params, complex_conds = _build_date_conditions({'date_after': '2024-01-01'})
        self.assertEqual(params.get('enriched_at'), 'gte.2024-01-01')
        self.assertEqual(complex_conds, [])

    def test_only_date_before(self):
        params, complex_conds = _build_date_conditions({'date_before': '2024-12-31'})
        self.assertEqual(params.get('enriched_at'), 'lte.2024-12-31')
        self.assertEqual(complex_conds, [])

    def test_no_date_filters(self):
        params, complex_conds = _build_date_conditions({})
        self.assertNotIn('enriched_at', params)
        self.assertEqual(complex_conds, [])


if __name__ == '__main__':
    unittest.main()
