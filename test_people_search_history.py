"""Tests for People-DB search-history helpers (migration 023).

Covers the pure functions that back the Search tab's recent-searches list and
the "you already ran this" repeat guard:
  - hash_search_filters: stable, order/case-insensitive, ignores sort/limit + empties
  - summarize_search_filters: short human label

The DB read/write helpers (record/find/list/delete) wrap the same REST client
as the rest of db.py and need live Supabase creds, so they're exercised
manually rather than here.
"""

from db import hash_search_filters, summarize_search_filters, record_people_search


# --- hash_search_filters -----------------------------------------------------

def test_hash_is_case_and_order_insensitive():
    a = {
        'crust_search_title': 'Team Lead',
        'crust_search_skills': 'react, node',
        'crust_search_seniority': ['Senior', 'Manager'],
    }
    b = {
        'crust_search_title': 'team lead',
        'crust_search_skills': 'node, react',
        'crust_search_seniority': ['Manager', 'Senior'],
    }
    assert hash_search_filters(a) == hash_search_filters(b)


def test_hash_ignores_sort_and_limit():
    base = {'crust_search_title': 'backend'}
    with_sort_limit = {
        'crust_search_title': 'backend',
        'crust_search_sort': 'Connections (most first)',
        'crust_search_limit': 1000,
    }
    assert hash_search_filters(base) == hash_search_filters(with_sort_limit)


def test_hash_ignores_empty_zero_false_fields():
    sparse = {'crust_search_title': 'qa'}
    padded = {
        'crust_search_title': 'qa',
        'crust_search_company': '',
        'crust_search_exp_min': 0,
        'crust_search_exact_company': False,
        'crust_search_seniority': [],
        'crust_search_location': '   ',
    }
    assert hash_search_filters(sparse) == hash_search_filters(padded)


def test_hash_distinguishes_real_differences():
    a = {'crust_search_title': 'team lead', 'crust_search_seniority': ['Senior']}
    b = {'crust_search_title': 'team lead', 'crust_search_seniority': ['Mid']}
    assert hash_search_filters(a) != hash_search_filters(b)


def test_hash_distinguishes_extra_filter():
    a = {'crust_search_title': 'devops'}
    b = {'crust_search_title': 'devops', 'crust_search_country': 'Israel'}
    assert hash_search_filters(a) != hash_search_filters(b)


def test_empty_filters_hash_is_stable():
    assert hash_search_filters({}) == hash_search_filters(None)
    assert hash_search_filters({}) == hash_search_filters({'crust_search_company': ''})


def test_hash_is_hex_sha256():
    h = hash_search_filters({'crust_search_title': 'x'})
    assert len(h) == 64
    int(h, 16)  # raises if not hex


def test_bool_true_changes_hash():
    off = {'crust_search_title': 'sales'}
    on = {'crust_search_title': 'sales', 'crust_search_has_email': True}
    assert hash_search_filters(off) != hash_search_filters(on)


# --- summarize_search_filters ------------------------------------------------

def test_summary_joins_key_fields():
    s = summarize_search_filters({
        'crust_search_title': 'Team Lead',
        'crust_search_company': 'Wiz',
        'crust_search_location': 'Tel Aviv',
        'crust_search_seniority': ['Senior'],
    })
    assert 'Team Lead' in s
    assert '@Wiz' in s
    assert 'Tel Aviv' in s
    assert 'Senior' in s
    assert ' · ' in s


def test_summary_falls_back_to_country_then_city():
    assert 'Israel' in summarize_search_filters({'crust_search_country': 'Israel'})
    assert 'Berlin' in summarize_search_filters({'crust_search_geo_city': 'Berlin'})


def test_summary_empty_is_friendly():
    assert summarize_search_filters({}) == 'All profiles (no filters)'


def test_summary_truncates_when_very_long():
    s = summarize_search_filters({'crust_search_title': 'x' * 300})
    assert len(s) <= 140
    assert s.endswith('…')


# --- record_people_search: result_urls storage (migration 024) ---------------
# A tiny fake REST client captures the insert/update payloads so we can assert
# the new result_urls handling without a live Supabase connection.

class _FakeClient:
    def __init__(self, existing=None):
        self._existing = existing if existing is not None else []
        self.inserted = []
        self.updated = []

    def select(self, *args, **kwargs):
        return self._existing

    def insert(self, table, row, *args, **kwargs):
        self.inserted.append(row)
        return [row]

    def update(self, table, updates, where, *args, **kwargs):
        self.updated.append(updates)
        return [updates]


def test_record_insert_includes_result_urls_when_provided():
    c = _FakeClient(existing=[])
    urls = ['https://www.linkedin.com/in/a', 'https://www.linkedin.com/in/b']
    ok = record_people_search(c, 'alice', {'crust_search_title': 'x'},
                              'hash1', 'x', 5, result_urls=urls)
    assert ok
    assert len(c.inserted) == 1
    assert c.inserted[0].get('result_urls') == urls


def test_record_insert_omits_result_urls_when_none():
    c = _FakeClient(existing=[])
    record_people_search(c, 'alice', {'crust_search_title': 'x'}, 'hash1', 'x', 5)
    assert len(c.inserted) == 1
    # Column left untouched so an un-migrated DB (no result_urls column) still works.
    assert 'result_urls' not in c.inserted[0]


def test_record_update_includes_result_urls_and_bumps_run_count():
    c = _FakeClient(existing=[{'id': 'r1', 'run_count': 1}])
    record_people_search(c, 'alice', {}, 'h', 's', 3, result_urls=['u'])
    assert len(c.updated) == 1
    assert c.updated[0].get('result_urls') == ['u']
    assert c.updated[0].get('run_count') == 2


def test_record_update_omits_result_urls_when_none():
    c = _FakeClient(existing=[{'id': 'r1', 'run_count': 2}])
    record_people_search(c, 'alice', {}, 'h', 's', 3)
    assert len(c.updated) == 1
    # A repeat run with no captured links must NOT wipe previously stored links.
    assert 'result_urls' not in c.updated[0]


def test_record_returns_false_without_client_or_username():
    assert record_people_search(None, 'alice', {}, 'h', 's', 1) is False
    assert record_people_search(_FakeClient(), '', {}, 'h', 's', 1) is False
