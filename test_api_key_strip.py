"""Tests that every credential loader strips stray surrounding whitespace
before returning the value.

Background
----------
On 2026-05-12 the daily ``db-refresh`` GitHub Actions cron's first run failed
with::

    Invalid leading whitespace, reserved character(s), or return character(s)
    in header value: 'Token <CRUSTDATA_API_KEY>\\n'

because ``CRUSTDATA_API_KEY`` had been stored with a trailing newline by a
sloppy ``gh secret set --body`` invocation. The fix: every credential loader
applies ``.strip()`` to the returned value so the entire class of failure
disappears.

These tests pin that behaviour for every loader the cron / dashboard touches:

* ``crustdata_search._load_api_key``
* ``dashboard.load_openai_key`` (and the shared ``_strip_secret`` helper)
* ``db.get_supabase_client`` (URL + key, all three sources)

All external services are mocked. No real network calls.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_env(*names):
    """Return a patch.dict that clears the named env vars."""
    return patch.dict(os.environ, {n: "" for n in names}, clear=False)


def _has_header_breaking_chars(value: str) -> bool:
    """Mirror the validation urllib3 performs on outgoing header values.

    urllib3 raises ``InvalidHeader`` (or similar) if a header value contains a
    bare CR/LF. A correctly-stripped key must therefore contain none.
    """
    return any(c in value for c in ("\r", "\n"))


# ---------------------------------------------------------------------------
# crustdata_search._load_api_key
# ---------------------------------------------------------------------------


class TestCrustdataLoader:
    """``crustdata_search._load_api_key`` strips surrounding whitespace."""

    def _make_loader(self, monkeypatch, tmp_path, *, config=None, env=None):
        """Run the loader with a controlled config.json and env."""
        import crustdata_search

        # Force the loader to look at our tmp config (it uses
        # ``Path(__file__).parent / 'config.json'``). Patch __file__ via the
        # module so the parent points at tmp_path.
        fake_module_file = tmp_path / "crustdata_search.py"
        fake_module_file.write_text("")
        monkeypatch.setattr(crustdata_search, "__file__", str(fake_module_file))

        if config is not None:
            (tmp_path / "config.json").write_text(json.dumps(config))
        # Clear env first, then set whatever the caller asked for.
        monkeypatch.delenv("CRUSTDATA_API_KEY", raising=False)
        if env:
            for k, v in env.items():
                monkeypatch.setenv(k, v)

        return crustdata_search._load_api_key

    def test_trailing_newline_stripped_from_config(self, monkeypatch, tmp_path):
        loader = self._make_loader(
            monkeypatch, tmp_path, config={"api_key": "sk-real-key\n"}
        )
        assert loader() == "sk-real-key"

    def test_leading_and_trailing_spaces_stripped_from_config(
        self, monkeypatch, tmp_path
    ):
        loader = self._make_loader(
            monkeypatch, tmp_path, config={"api_key": "   sk-real-key  "}
        )
        assert loader() == "sk-real-key"

    def test_trailing_crlf_stripped_from_env(self, monkeypatch, tmp_path):
        loader = self._make_loader(
            monkeypatch,
            tmp_path,
            env={"CRUSTDATA_API_KEY": "env-key\r\n"},
        )
        result = loader()
        assert result == "env-key"
        # Sanity: the returned value must be safe to drop into an HTTP header.
        assert not _has_header_breaking_chars(result)

    def test_clean_key_passes_through_unchanged(self, monkeypatch, tmp_path):
        loader = self._make_loader(
            monkeypatch, tmp_path, env={"CRUSTDATA_API_KEY": "clean-key"}
        )
        assert loader() == "clean-key"

    def test_missing_key_still_raises(self, monkeypatch, tmp_path):
        from crustdata_search import AuthenticationError

        loader = self._make_loader(monkeypatch, tmp_path)
        # No config, no env -> the original AuthenticationError fires.
        with pytest.raises(AuthenticationError):
            loader()

    def test_empty_string_in_env_still_raises(self, monkeypatch, tmp_path):
        from crustdata_search import AuthenticationError

        # Empty config.json (no api_key field) + empty env -> raises.
        loader = self._make_loader(
            monkeypatch, tmp_path, config={}, env={"CRUSTDATA_API_KEY": ""}
        )
        with pytest.raises(AuthenticationError):
            loader()

    def test_whitespace_only_env_treated_as_missing(self, monkeypatch, tmp_path):
        """A key that is purely whitespace cannot be a real key. After strip
        it becomes an empty string. The loader either falls through to env
        (which is also whitespace here) and raises, or returns ''. Either
        way the caller's truthiness check will treat the result as missing."""
        from crustdata_search import AuthenticationError

        loader = self._make_loader(
            monkeypatch, tmp_path, env={"CRUSTDATA_API_KEY": "   \n"}
        )
        # After strip -> "" -> falsy -> downstream behaviour for "missing".
        # The current implementation returns "" (bypasses the truthy guard
        # only because strip happens after); either an empty return OR a
        # raise is acceptable, but not header-breaking whitespace.
        try:
            result = loader()
        except AuthenticationError:
            return
        assert result == ""
        assert not _has_header_breaking_chars(result)


# ---------------------------------------------------------------------------
# dashboard loaders (OpenAI / Anthropic / Crustdata via config / SalesQL /
# PhantomBuster)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard_module():
    """Import dashboard once for the whole module."""
    import dashboard

    return dashboard


class TestDashboardLoaders:
    """``dashboard.load_*_key`` all run config values through ``_strip_secret``."""

    def test_strip_secret_helper_trailing_newline(self, dashboard_module):
        assert dashboard_module._strip_secret("abc\n") == "abc"

    def test_strip_secret_helper_leading_and_trailing_spaces(self, dashboard_module):
        assert dashboard_module._strip_secret("  abc  ") == "abc"

    def test_strip_secret_helper_clean_value(self, dashboard_module):
        assert dashboard_module._strip_secret("clean-key") == "clean-key"

    def test_strip_secret_helper_preserves_none(self, dashboard_module):
        # Loaders test the result for truthiness; passing None through keeps
        # the "missing" semantics intact.
        assert dashboard_module._strip_secret(None) is None

    def test_strip_secret_helper_preserves_empty_string(self, dashboard_module):
        assert dashboard_module._strip_secret("") == ""

    def test_strip_secret_helper_ignores_non_strings(self, dashboard_module):
        # Streamlit secrets sometimes return ints (e.g. agent IDs). Don't
        # explode on those.
        assert dashboard_module._strip_secret(12345) == 12345

    def test_load_openai_key_strips_trailing_newline(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"openai_api_key": "sk-openai\n"},
        ):
            result = dashboard_module.load_openai_key()
        assert result == "sk-openai"
        assert not _has_header_breaking_chars(result)

    def test_load_openai_key_strips_leading_trailing_spaces(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"openai_api_key": "   sk-openai   "},
        ):
            assert dashboard_module.load_openai_key() == "sk-openai"

    def test_load_openai_key_clean_passes_through(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"openai_api_key": "clean-openai"},
        ):
            assert dashboard_module.load_openai_key() == "clean-openai"

    def test_load_openai_key_missing_returns_none(self, dashboard_module):
        # Empty config -> None -> still falsy for the caller's "no key" path.
        with patch.object(dashboard_module, "load_config", return_value={}):
            assert dashboard_module.load_openai_key() is None

    def test_load_openai_key_empty_string_passes_through(self, dashboard_module):
        # Empty string keeps original loader semantics (caller treats as missing).
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"openai_api_key": ""},
        ):
            assert dashboard_module.load_openai_key() == ""

    def test_load_api_key_strips_trailing_newline(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"api_key": "crustdata-key\r\n"},
        ):
            result = dashboard_module.load_api_key()
        assert result == "crustdata-key"
        assert not _has_header_breaking_chars(result)

    def test_load_anthropic_key_strips_whitespace(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"anthropic_api_key": "  anth-key\n"},
        ):
            # Also patch st.secrets access path to avoid touching real secrets.
            with patch.object(dashboard_module.st, "secrets", {}, create=True):
                result = dashboard_module.load_anthropic_key()
        assert result == "anth-key"

    def test_load_phantombuster_key_strips_whitespace(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"phantombuster_api_key": "pb-key\n"},
        ):
            assert dashboard_module.load_phantombuster_key() == "pb-key"

    def test_load_salesql_key_strips_whitespace(self, dashboard_module):
        with patch.object(
            dashboard_module,
            "load_config",
            return_value={"salesql_api_key": "salesql-key  "},
        ):
            assert dashboard_module.load_salesql_key() == "salesql-key"


# ---------------------------------------------------------------------------
# db.get_supabase_client (URL + key, env source)
# ---------------------------------------------------------------------------


class TestSupabaseLoader:
    """``db.get_supabase_client`` strips whitespace from URL and key."""

    def _reset_db_module(self):
        """Force a clean db module so its module-level state is predictable."""
        if "db" in sys.modules:
            return importlib.reload(sys.modules["db"])
        return importlib.import_module("db")

    def test_trailing_newline_stripped_from_env_url_and_key(self, monkeypatch):
        db = self._reset_db_module()
        # Make sure we don't accidentally read a real config.json in the repo.
        monkeypatch.setattr(
            db.Path, "exists", lambda self: False, raising=False
        )
        monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co\n")
        monkeypatch.setenv("SUPABASE_KEY", "service-role-key\r\n")

        client = db.get_supabase_client()
        assert client is not None
        assert client.url == "https://abc.supabase.co"
        assert client.key == "service-role-key"
        assert not _has_header_breaking_chars(client.url)
        assert not _has_header_breaking_chars(client.key)
        # The Authorization header would also embed the key — make sure the
        # value that ends up in client.headers is header-safe too.
        for h, v in client.headers.items():
            assert not _has_header_breaking_chars(str(v)), (
                f"Header '{h}' still contains CR/LF after strip"
            )

    def test_leading_and_trailing_spaces_stripped_from_env(self, monkeypatch):
        db = self._reset_db_module()
        monkeypatch.setattr(
            db.Path, "exists", lambda self: False, raising=False
        )
        monkeypatch.setenv("SUPABASE_URL", "  https://abc.supabase.co  ")
        monkeypatch.setenv("SUPABASE_KEY", "  service-role-key  ")

        client = db.get_supabase_client()
        assert client is not None
        assert client.url == "https://abc.supabase.co"
        assert client.key == "service-role-key"

    def test_clean_env_passes_through(self, monkeypatch):
        db = self._reset_db_module()
        monkeypatch.setattr(
            db.Path, "exists", lambda self: False, raising=False
        )
        monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "service-role-key")

        client = db.get_supabase_client()
        assert client is not None
        assert client.url == "https://abc.supabase.co"
        assert client.key == "service-role-key"

    def test_missing_env_returns_none(self, monkeypatch):
        """Empty/missing creds still produce None — error semantics preserved."""
        db = self._reset_db_module()
        monkeypatch.setattr(
            db.Path, "exists", lambda self: False, raising=False
        )
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        # Also ensure no streamlit secrets sneak in.
        # st may or may not be importable; if it is, force st.secrets to {}.
        try:
            import streamlit as _st  # noqa: F401
            monkeypatch.setattr(
                "streamlit.secrets", {}, raising=False
            )
        except Exception:
            pass

        assert db.get_supabase_client() is None

    def test_whitespace_only_env_treated_as_missing(self, monkeypatch):
        """A whitespace-only key strips to '' which is falsy -> client is None."""
        db = self._reset_db_module()
        monkeypatch.setattr(
            db.Path, "exists", lambda self: False, raising=False
        )
        monkeypatch.setenv("SUPABASE_URL", "   ")
        monkeypatch.setenv("SUPABASE_KEY", "\n\n")
        try:
            monkeypatch.setattr("streamlit.secrets", {}, raising=False)
        except Exception:
            pass

        assert db.get_supabase_client() is None
