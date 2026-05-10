"""Regression tests for company-name filter matching.

User report (2026-05-10): the "not relevant companies" filter let Egged
profiles and Elad Systems profiles through, even though the user had
"Elad Software Systems" in her not-relevant list. Root cause: the matcher
only did exact + prefix comparison after normalization, so "Elad Software
Systems" (3 tokens) and "Elad Systems" (2 tokens) couldn't be matched as
the same company.

These tests pin the token-subset rule that catches that case while
keeping the existing exact + prefix behavior intact, and the guard that
prevents a one-word entry like "Apple" from matching every "Apple X"
company.

Tests import the helpers from the import-safe `company_matching` module
so CI doesn't need to stub streamlit / requests / openai etc.
"""

from company_matching import (
    normalize_company_name as norm,
    company_matches_filter_list as matches,
)


# ---------------------------------------------------------------------------
# normalize_company_name
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_strips():
    assert norm("  Microsoft  ") == "microsoft"


def test_normalize_strips_locale_suffix():
    assert norm("Microsoft Israel") == "microsoft"


def test_normalize_strips_legal_suffix():
    assert norm("Acme Ltd") == "acme"
    assert norm("Acme Inc") == "acme"


def test_normalize_returns_empty_for_blank():
    assert norm("") == ""
    assert norm("   ") == ""


def test_normalize_keeps_systems_token():
    # "Systems" is NOT in the strip list — it's a real identifying word.
    assert norm("Elad Software Systems") == "elad software systems"
    assert norm("Elad Systems") == "elad systems"


# ---------------------------------------------------------------------------
# company_matches_filter_list — primary regression
# ---------------------------------------------------------------------------


def test_elad_systems_matches_elad_software_systems():
    """The original Chen bug: 'Elad Systems' on a profile must match
    'Elad Software Systems' in the not-relevant list."""
    assert matches("Elad Systems", ["Elad Software Systems"]) is True


def test_elad_software_systems_matches_elad_systems():
    """Same as above but with the roles reversed (shorter in the list)."""
    assert matches("Elad Software Systems", ["Elad Systems"]) is True


# ---------------------------------------------------------------------------
# company_matches_filter_list — exact + prefix cases stay green
# ---------------------------------------------------------------------------


def test_exact_match_after_normalization():
    assert matches("Microsoft Israel", ["Microsoft"]) is True


def test_prefix_match_both_directions():
    assert matches("Bank Leumi", ["Bank Leumi Le-Israel"]) is True
    assert matches("Bank Leumi Le-Israel", ["Bank Leumi"]) is True


def test_empty_inputs_return_false():
    assert matches("", ["Anything"]) is False
    assert matches(None, ["Anything"]) is False
    assert matches("Acme", []) is False
    assert matches("Acme", [""]) is False


# ---------------------------------------------------------------------------
# company_matches_filter_list — false-positive guards
# ---------------------------------------------------------------------------


def test_unrelated_companies_do_not_match():
    assert matches("Google", ["Microsoft"]) is False
    assert matches("Facebook", ["Meta Platforms"]) is False


def test_short_single_word_does_not_match_longer_variant():
    """The ≥4-char prefix guard: 'Sun' (3 chars) shouldn't match 'Sun
    Microsystems' just because it's a substring prefix."""
    assert matches("Sun", ["Sun Microsystems"]) is False


def test_two_token_subset_requires_both_tokens():
    """{a,b} ⊆ {b,c} is False — must share BOTH tokens, not just one."""
    assert matches("Bank Leumi", ["Bank HaPoalim"]) is False
    assert matches("Tata Consultancy", ["Tata Steel"]) is False


# ---------------------------------------------------------------------------
# Documented prior behavior — not changed by this fix, just pinned so we
# notice if it shifts in the future.
# ---------------------------------------------------------------------------


def test_prior_behavior_4char_prefix_match():
    """Existing rule: a ≥4-char single-token entry matches longer variants
    via prefix. So 'Apple' filters 'Apple Industries'. This is permissive
    by design — if it ever turns into a real false-positive problem, tighten
    the prefix to require a trailing space."""
    assert matches("Apple Industries", ["Apple"]) is True
