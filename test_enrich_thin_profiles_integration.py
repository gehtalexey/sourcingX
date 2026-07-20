"""Tests for dashboard.enrich_thin_profiles_for_batch() — the screening-time
auto-enrichment step that tops up profiles missing skills/summary (e.g. from
the new Crustdata search endpoints) via the new batch-enrich pipeline before
AI Screen sees them.

All network-free: batch_enrich_profiles() and save_enriched_profiles_bulk()
are monkeypatched at the dashboard module level.
"""

import dashboard


def _thin_profile(url, name="Thin Person"):
    return {
        "linkedin_url": url,
        "name": name,
        "raw_crustdata": {"name": name, "skills": [], "summary": ""},
    }


def _rich_profile(url, name="Rich Person"):
    return {
        "linkedin_url": url,
        "name": name,
        "raw_crustdata": {
            "name": name,
            "skills": ["Python"],
            "summary": "Already has everything.",
        },
    }


def _flat_enriched(url, name):
    return {
        "name": name,
        "skills": ["Python", "Kubernetes"],
        "summary": "Filled in via batch enrich.",
        "linkedin_flagship_url": url,
        "current_employers": [],
        "past_employers": [],
    }


class TestEnrichThinProfilesForBatch:
    def test_only_thin_profiles_are_submitted(self, monkeypatch):
        thin = _thin_profile("https://www.linkedin.com/in/thin")
        rich = _rich_profile("https://www.linkedin.com/in/rich")

        captured_urls = {}

        def fake_batch_enrich(urls, api_key=None):
            captured_urls["urls"] = urls
            return {
                "by_url": {urls[0]: _flat_enriched(urls[0], "Thin Person")},
                "requested": 1, "fulfilled": 1, "unmatched": [], "credits_used": 1,
                "batch_ids": ["b1"],
            }

        monkeypatch.setattr(dashboard, "batch_enrich_profiles", fake_batch_enrich)
        monkeypatch.setattr(dashboard, "save_enriched_profiles_bulk", lambda *a, **k: {"saved": 1, "errors": 0, "error_messages": []})

        stats = dashboard.enrich_thin_profiles_for_batch([thin, rich], api_key="test-key", db_client=None)

        assert captured_urls["urls"] == ["https://www.linkedin.com/in/thin"]
        assert stats == {"thin_found": 1, "enriched": 1, "unmatched": 0, "credits_used": 1}
        # The already-rich profile must be untouched.
        assert rich["raw_crustdata"]["name"] == "Rich Person"
        assert rich["raw_crustdata"]["skills"] == ["Python"]

    def test_thin_profile_gets_merged_with_enriched_data(self, monkeypatch):
        thin = _thin_profile("https://www.linkedin.com/in/thin")

        def fake_batch_enrich(urls, api_key=None):
            return {
                "by_url": {urls[0]: _flat_enriched(urls[0], "Thin Person")},
                "requested": 1, "fulfilled": 1, "unmatched": [], "credits_used": 1,
                "batch_ids": ["b1"],
            }

        monkeypatch.setattr(dashboard, "batch_enrich_profiles", fake_batch_enrich)
        monkeypatch.setattr(dashboard, "save_enriched_profiles_bulk", lambda *a, **k: {"saved": 1, "errors": 0, "error_messages": []})

        dashboard.enrich_thin_profiles_for_batch([thin], api_key="test-key", db_client=None)

        assert thin["raw_crustdata"]["skills"] == ["Python", "Kubernetes"]
        assert thin["raw_crustdata"]["summary"] == "Filled in via batch enrich."
        assert "raw_data" not in thin  # stale key cleared, screen_profile() must read raw_crustdata

    def test_save_receives_flat_profiles_not_a_wrapper(self, monkeypatch):
        """Regression test for the Codex-caught bug (2026-07-20): the save
        step must pass the translator's flat dicts directly to
        save_enriched_profiles_bulk(), NOT wrapped as
        {linkedin_url, raw_data, original_url} — a wrapper would make
        _prepare_profile_row() silently write near-empty rows."""
        thin = _thin_profile("https://www.linkedin.com/in/thin")
        flat = _flat_enriched(thin["linkedin_url"], "Thin Person")

        def fake_batch_enrich(urls, api_key=None):
            return {
                "by_url": {urls[0]: flat},
                "requested": 1, "fulfilled": 1, "unmatched": [], "credits_used": 1,
                "batch_ids": ["b1"],
            }

        saved_args = {}

        def fake_save(db_client, profiles, original_url_map=None, batch_size=100, **kwargs):
            saved_args["profiles"] = profiles
            saved_args["original_url_map"] = original_url_map
            return {"saved": len(profiles), "errors": 0, "error_messages": []}

        monkeypatch.setattr(dashboard, "batch_enrich_profiles", fake_batch_enrich)
        monkeypatch.setattr(dashboard, "save_enriched_profiles_bulk", fake_save)

        dashboard.enrich_thin_profiles_for_batch([thin], api_key="test-key", db_client=object())

        assert saved_args["profiles"] == [flat]
        # Every saved item must have name/skills directly at the top level —
        # exactly what _prepare_profile_row() reads — not nested under a
        # "raw_data" key.
        assert saved_args["profiles"][0]["name"] == "Thin Person"
        assert saved_args["profiles"][0]["skills"] == ["Python", "Kubernetes"]
        assert thin["linkedin_url"] in saved_args["original_url_map"].values()

    def test_unmatched_profiles_left_thin_and_counted_not_blocked(self, monkeypatch):
        thin = _thin_profile("https://www.linkedin.com/in/nomatch")

        monkeypatch.setattr(
            dashboard, "batch_enrich_profiles",
            lambda urls, api_key=None: {
                "by_url": {}, "requested": 1, "fulfilled": 0,
                "unmatched": urls, "credits_used": 0, "batch_ids": ["b1"],
            },
        )
        monkeypatch.setattr(dashboard, "save_enriched_profiles_bulk", lambda *a, **k: {"saved": 0, "errors": 0, "error_messages": []})

        stats = dashboard.enrich_thin_profiles_for_batch([thin], api_key="test-key", db_client=None)

        assert stats["unmatched"] == 1
        assert stats["enriched"] == 0
        # Left as-is — still screenable, not dropped from the caller's list.
        assert thin["raw_crustdata"]["skills"] == []

    def test_enrichment_failure_does_not_raise_screening_proceeds(self, monkeypatch):
        thin = _thin_profile("https://www.linkedin.com/in/thin")

        def boom(urls, api_key=None):
            raise RuntimeError("Crustdata is down")

        monkeypatch.setattr(dashboard, "batch_enrich_profiles", boom)

        stats = dashboard.enrich_thin_profiles_for_batch([thin], api_key="test-key", db_client=None)

        assert stats["thin_found"] == 1
        assert stats["unmatched"] == 1
        assert stats["enriched"] == 0

    def test_no_thin_profiles_skips_the_api_call_entirely(self, monkeypatch):
        rich = _rich_profile("https://www.linkedin.com/in/rich")
        called = {"n": 0}
        monkeypatch.setattr(dashboard, "batch_enrich_profiles", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

        stats = dashboard.enrich_thin_profiles_for_batch([rich], api_key="test-key", db_client=None)

        assert called["n"] == 0
        assert stats == {"thin_found": 0, "enriched": 0, "unmatched": 0, "credits_used": 0}

    def test_missing_either_skill_or_summary_alone_is_not_thin(self):
        """Alexey's call, 2026-07-20: only enrich when BOTH are missing, not
        just one — cheaper than an OR check."""
        partial = {
            "linkedin_url": "https://www.linkedin.com/in/partial",
            "raw_crustdata": {"name": "Partial", "skills": ["Python"], "summary": ""},
        }
        stats = dashboard.enrich_thin_profiles_for_batch([partial], api_key="test-key", db_client=None)
        assert stats["thin_found"] == 0


class TestEnrichHookedInUnconditionally:
    """Regression guard (Codex review, 2026-07-20): the call to
    enrich_thin_profiles_for_batch() in the screening batch loop must NOT be
    nested inside the `if missing_before > 0:` block, or profiles that are
    already thin in memory (with nothing upstream flagging them as missing)
    silently skip enrichment. A full Streamlit-driven test of that tab isn't
    practical here, so this checks the source structure directly."""

    def test_enrich_call_is_not_nested_inside_missing_before_block(self):
        import inspect

        lines = inspect.getsource(dashboard).splitlines()

        if_line_no = next(i for i, l in enumerate(lines) if "if missing_before > 0:" in l)
        call_line_no = next(i for i, l in enumerate(lines) if "enrich_thin_profiles_for_batch(" in l and "def " not in l)
        assert call_line_no > if_line_no, "expected the enrich call to appear after the fetch guard"

        if_indent = len(lines[if_line_no]) - len(lines[if_line_no].lstrip())

        # Find where the `if missing_before > 0:` block actually ends: the
        # first subsequent non-blank line whose indentation is <= the if's
        # own indentation (i.e. a dedent back out of the block).
        block_end_line_no = len(lines)
        for i in range(if_line_no + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent <= if_indent:
                block_end_line_no = i
                break

        assert call_line_no >= block_end_line_no, (
            "enrich_thin_profiles_for_batch() call appears to be nested "
            "inside `if missing_before > 0:` — it must run unconditionally, "
            "not gated behind the raw-data fetch step reporting something missing"
        )
