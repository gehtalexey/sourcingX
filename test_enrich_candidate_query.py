"""Tests for the Enrich tab's targeted dedup query builder.

Background
----------
The Enrich tab decides "already enriched" vs "to enrich" for the loaded
profiles. It used to download every profile enriched in the last N months
(~31k rows) and build a giant in-memory lookup on every uncached render —
slow, and slower on every enrichment as the table grew.

`_build_candidate_query_specs` replaces that with targeted queries: only fetch
the DB profiles that could match the input URLs. The risk (flagged in code
review) is that a narrower query misses rows the old full scan would have
caught. These tests pin the matching CONTRACT:

* Exact pass — the normalized URL (and its hyphen-free form) is queried against
  linkedin_url / original_url / original_urls.
* Suffix pass — a clean multi-part slug (``john-doe``) may be stored in the DB
  only under an ID-suffixed form (``john-doe-abc123``). The builder MUST still
  emit a query that fetches that row, or the profile is wrongly marked
  "to enrich" and re-enriched (wasted credits).

The builder is pure (no DB), so these tests run offline. A tiny simulator
(`rows_fetched_by`) interprets the emitted PostgREST filter specs against fake
DB rows, so the tests assert observable behaviour, not string formatting.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import _build_candidate_query_specs

CUTOFF = "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# Simulator: which fake DB rows would a set of query specs fetch?
# ---------------------------------------------------------------------------

def _split_top_level(s: str) -> list:
    """Split an or=() clause body on commas that are not inside (...) or {...}."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return parts


def _values(blob: str) -> list:
    """Pull the quoted values out of an in.(...) / ov.{...} / like."..." clause."""
    return re.findall(r'"([^"]*)"', blob)


def _clause_matches(clause: str, row: dict) -> bool:
    # clause looks like  col.op.<rest>
    col, _, rest = clause.partition(".")
    op, _, arg = rest.partition(".")
    if op == "in":
        wanted = set(_values(arg))
        return row.get(col) in wanted
    if op == "ov":
        wanted = set(_values(arg))
        return bool(set(row.get(col) or []) & wanted)
    if op == "like":
        pat = _values(arg)[0]
        regex = "^" + re.escape(pat).replace(r"\*", ".*") + "$"
        return bool(re.match(regex, row.get(col) or ""))
    raise AssertionError(f"simulator does not handle op {op!r} in {clause!r}")


def rows_fetched_by(specs: list, rows: list) -> list:
    """Return the rows the given query specs would fetch from `rows`."""
    fetched = []
    for row in rows:
        for spec in specs:
            f = spec["filters"]
            assert f.get("enriched_at", "").startswith("gte."), "every spec filters on enriched_at"
            body = f["or"]
            assert body.startswith("(") and body.endswith(")")
            if any(_clause_matches(c, row) for c in _split_top_level(body[1:-1])):
                fetched.append(row)
                break
    return fetched


# ---------------------------------------------------------------------------
# Exact pass
# ---------------------------------------------------------------------------

def test_exact_url_is_fetched():
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/jane-doe-12345"], CUTOFF
    )
    row = {"linkedin_url": "https://www.linkedin.com/in/jane-doe-12345",
           "original_url": None, "original_urls": []}
    assert rows_fetched_by(specs, [row]) == [row]


def test_match_via_original_url_and_original_urls_array():
    """A profile recorded under a different URL than the input still matches if
    the input URL appears in original_url or the original_urls array."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/dor-shwartz-4490a885"], CUTOFF
    )
    via_original_url = {
        "linkedin_url": "https://www.linkedin.com/in/dor-shwartz",
        "original_url": "https://www.linkedin.com/in/dor-shwartz-4490a885",
        "original_urls": [],
    }
    via_array = {
        "linkedin_url": "https://www.linkedin.com/in/someone-else",
        "original_url": None,
        "original_urls": ["https://www.linkedin.com/in/dor-shwartz-4490a885"],
    }
    assert rows_fetched_by(specs, [via_original_url]) == [via_original_url]
    assert rows_fetched_by(specs, [via_array]) == [via_array]


def test_hyphen_free_variant_in_exact_pass():
    """Input john-doe should also fetch a profile stored as johndoe."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/john-doe"], CUTOFF
    )
    row = {"linkedin_url": "https://www.linkedin.com/in/johndoe",
           "original_url": None, "original_urls": []}
    assert rows_fetched_by(specs, [row]) == [row]


def test_unrelated_profile_not_fetched():
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/john-doe"], CUTOFF
    )
    row = {"linkedin_url": "https://www.linkedin.com/in/someone-totally-different",
           "original_url": None, "original_urls": []}
    assert rows_fetched_by(specs, [row]) == []


# ---------------------------------------------------------------------------
# Suffix pass — the code-review P1: clean slug vs ID-suffixed DB row
# ---------------------------------------------------------------------------

def test_clean_slug_fetches_id_suffixed_db_row():
    """THE regression guard: input is the clean slug, the DB only has the
    ID-suffixed form. The old full scan caught this; the targeted query must
    too, via the suffix pass."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/einav-friedman"], CUTOFF
    )
    suffixed_only = {
        "linkedin_url": "https://www.linkedin.com/in/einav-friedman-9190525",
        "original_url": "https://www.linkedin.com/in/einav-friedman-9190525",
        "original_urls": ["https://www.linkedin.com/in/einav-friedman-9190525"],
    }
    assert rows_fetched_by(specs, [suffixed_only]) == [suffixed_only], (
        "clean-slug input must still fetch the ID-suffixed DB row"
    )


def test_clean_slug_fetches_hyphen_free_id_suffixed_db_row():
    """Code-review follow-up P1: the DB row may carry the ID suffix on the
    HYPHEN-FREE form (johndoe-abc123). Input john-doe must still fetch it — the
    old full scan matched it via username_no_hyphen."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/john-doe"], CUTOFF
    )
    hyphen_free_suffixed = {
        "linkedin_url": "https://www.linkedin.com/in/johndoe-abc123",
        "original_url": "https://www.linkedin.com/in/johndoe-abc123",
        "original_urls": ["https://www.linkedin.com/in/johndoe-abc123"],
    }
    assert rows_fetched_by(specs, [hyphen_free_suffixed]) == [hyphen_free_suffixed], (
        "clean-slug input must also fetch the hyphen-free ID-suffixed DB row"
    )


def test_suffix_pass_does_not_match_different_name():
    """The suffix prefix must not bleed across a name boundary:
    input john-doe must NOT fetch john-doely-... (no hyphen after 'doe')."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/john-doe"], CUTOFF
    )
    different = {"linkedin_url": "https://www.linkedin.com/in/john-doely-99999",
                 "original_url": None, "original_urls": []}
    assert rows_fetched_by(specs, [different]) == []


def test_already_suffixed_input_emits_no_suffix_pass():
    """If every input slug already carries an ID suffix, there's nothing for the
    suffix pass to do — only exact-pass specs should be emitted."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/jane-doe-6b4438122",
         "https://www.linkedin.com/in/mor-davidi-b2a61694"],
        CUTOFF,
    )
    like_specs = [s for s in specs if ".like." in s["filters"]["or"]]
    assert like_specs == []


def test_no_hyphen_slug_emits_no_suffix_pass():
    """A single-token slug (no hyphen) must not trigger a prefix pass — a bare
    'dan' prefix would over-match half the table."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/yelenacarbalido"], CUTOFF
    )
    like_specs = [s for s in specs if ".like." in s["filters"]["or"]]
    assert like_specs == []


def test_percent_encoded_slug_excluded_from_suffix_pass():
    """Percent-encoded slugs would make '%' act as a LIKE wildcard, so they are
    handled by the exact pass only — no suffix pass for them."""
    url = "https://www.linkedin.com/in/ofir-cohen-%D7%90%D7%95%D7%A4-b01a94114"
    specs = _build_candidate_query_specs([url], CUTOFF)
    like_specs = [s for s in specs if ".like." in s["filters"]["or"]]
    assert like_specs == []
    # ...but the exact pass still covers it.
    row = {"linkedin_url": url, "original_url": None, "original_urls": []}
    assert rows_fetched_by(specs, [row]) == [row]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_and_invalid_inputs_produce_no_specs():
    assert _build_candidate_query_specs([], CUTOFF) == []
    assert _build_candidate_query_specs(
        ["", None, "not a url", "https://www.linkedin.com/company/acme"], CUTOFF
    ) == []


def test_every_spec_carries_the_enriched_at_cutoff():
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/john-doe",
         "https://www.linkedin.com/in/jane-doe-12345"],
        CUTOFF,
    )
    assert specs, "expected at least one spec"
    for s in specs:
        assert s["filters"]["enriched_at"] == f"gte.{CUTOFF}"


def test_exact_specs_have_linkedin_url_only_fallback():
    """The exact pass must keep a simpler linkedin_url-only fallback for an
    un-migrated DB without the original_urls column."""
    specs = _build_candidate_query_specs(
        ["https://www.linkedin.com/in/john-doe"], CUTOFF
    )
    exact = [s for s in specs if "original_urls.ov" in s["filters"]["or"]]
    assert exact, "expected an exact-pass spec"
    for s in exact:
        assert "fallback" in s
        assert s["fallback"]["linkedin_url"].startswith("in.(")
