"""
Regression test: the suite must never reach the live Supabase database.

The autouse ``_no_live_supabase`` fixture in conftest.py makes every
Supabase client lookup return None, and a session-wide guard on
requests' HTTPAdapter refuses any request to a *.supabase.co host.
Before this, dashboard tests wrote junk rows into the live api_usage_logs
table through get_usage_tracker().
"""

import pytest
import requests

# Imported at collection time, like the real dashboard tests, so the autouse
# fixture sees it already loaded.
dashboard = pytest.importorskip("dashboard")


def test_get_supabase_client_returns_none():
    import db
    assert db.get_supabase_client() is None


def test_dashboard_usage_tracker_returns_none():
    assert dashboard.get_supabase_client() is None
    assert dashboard._get_db_client() is None
    assert dashboard.get_usage_tracker() is None


def test_real_request_to_supabase_host_is_blocked():
    # Fake host: the guard raises before any socket is opened.
    with pytest.raises(RuntimeError, match="tests must not reach live Supabase"):
        requests.post(
            "https://example.supabase.co/rest/v1/api_usage_logs",
            json={"provider": "test"},
            timeout=1,
        )
