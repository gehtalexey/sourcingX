"""Regression tests for the Tab 0 / Crustdata Search title filter shape.

Background
----------
PR #39 (docs/research/crustdata-filter-semantics.md) documented that
Crustdata persondb/search evaluates AND across nested-array columns
(current_employers.*) per-array-element. The collapse:

    title=software developer + company=wiz/monday/wix/forter + region=IL
    -> total_count == 5

is caused by Crustdata requiring both the title and the company
predicate to hit the SAME current-employer entry.

The fix wraps every title term in an OR across BOTH columns:

    current_employers.title (nested per-element)
    headline                (top-level profile field, per-profile)

This is strictly more candidate-friendly than the previous shape: every
profile that used to match still matches via the title branch, plus
every profile whose headline mentions the title now also matches.

Measured lift on the PR #28 repro: 5 -> 12-149 depending on title
breadth.

These tests pin the emitted filter shape so a future refactor cannot
silently regress back to the same-element collapse.
"""

from crustdata_search import build_filters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root(filters_dict):
    """build_filters wraps its output in {"filters": ...}; unwrap it."""
    assert "filters" in filters_dict, "build_filters must wrap output in 'filters'"
    return filters_dict["filters"]


def _all_conditions(root):
    """Top-level conditions, handling the single-condition shortcut."""
    if root.get("op") == "and":
        return list(root.get("conditions", []))
    # Single condition is returned bare.
    return [root]


def _title_block(filters_dict):
    """Return the title OR-block, or None.

    The title OR-block is the only OR whose inner predicates touch
    exclusively `current_employers.title` and/or `headline`.
    """
    root = _root(filters_dict)
    for cond in _all_conditions(root):
        if cond.get("op") != "or":
            continue
        inner = cond.get("conditions", [])
        cols = {c.get("column") for c in inner}
        if cols.issubset({"current_employers.title", "headline"}) and cols:
            return cond
    return None


def _predicates_by_column(or_block, column):
    return [c for c in or_block["conditions"] if c.get("column") == column]


# ---------------------------------------------------------------------------
# Single title -> OR(current_employers.title=t, headline=t)
# ---------------------------------------------------------------------------


class TestTitleHeadlineDualColumnMatch:
    def test_single_title_emits_or_across_both_columns(self):
        filters = build_filters(title="software developer")

        block = _title_block(filters)
        assert block is not None, "title filter should emit an OR block"
        assert block["op"] == "or"

        inner = block["conditions"]
        assert len(inner) == 2, "single title should produce exactly 2 predicates"

        title_preds = _predicates_by_column(block, "current_employers.title")
        headline_preds = _predicates_by_column(block, "headline")

        assert len(title_preds) == 1
        assert len(headline_preds) == 1
        assert title_preds[0]["value"] == "software developer"
        assert headline_preds[0]["value"] == "software developer"
        assert title_preds[0]["type"] == "[.]"
        assert headline_preds[0]["type"] == "[.]"

    def test_multi_title_mirrors_each_term_onto_both_columns(self):
        filters = build_filters(title="backend engineer, software developer, sre")

        block = _title_block(filters)
        assert block is not None
        assert block["op"] == "or"

        title_preds = _predicates_by_column(block, "current_employers.title")
        headline_preds = _predicates_by_column(block, "headline")

        # Three input terms, mirrored onto two columns each = 6 predicates.
        assert len(block["conditions"]) == 6
        assert len(title_preds) == 3
        assert len(headline_preds) == 3

        title_vals = {p["value"] for p in title_preds}
        headline_vals = {p["value"] for p in headline_preds}
        expected = {"backend engineer", "software developer", "sre"}
        assert title_vals == expected
        assert headline_vals == expected

    def test_combined_with_company_top_level_and_has_two_or_branches(self):
        """The PR #28 repro: title + companies + region=Israel.

        The final filter must AND together the title OR-block (across
        both columns) with the company OR-block — they are independent
        branches, not nested into the same per-element predicate.
        """
        filters = build_filters(
            title="software developer",
            company="wiz, monday, wix, forter",
            location="Israel",
        )

        root = _root(filters)
        assert root["op"] == "and"

        or_blocks = [c for c in root["conditions"] if c.get("op") == "or"]
        # Multi-company emits one OR block; multi-column title emits another.
        assert len(or_blocks) == 2

        # Title OR-block sanity.
        title_block = _title_block(filters)
        assert title_block is not None
        assert len(title_block["conditions"]) == 2  # 1 title * 2 columns

        # Company OR-block sanity (NOT touched by this change).
        company_blocks = [
            c
            for c in or_blocks
            if c is not title_block
            and all(
                inner.get("column") == "current_employers.name"
                for inner in c["conditions"]
            )
        ]
        assert len(company_blocks) == 1
        assert len(company_blocks[0]["conditions"]) == 4

        # Region filter is unchanged and lives outside any OR-block.
        region_preds = [
            c for c in root["conditions"] if c.get("column") == "region"
        ]
        assert len(region_preds) == 1
        assert region_preds[0]["value"] == "Israel"

    def test_empty_title_emits_no_title_block(self):
        filters = build_filters(title="", company="wiz")
        assert _title_block(filters) is None

    def test_none_title_emits_no_title_block(self):
        filters = build_filters(company="wiz")
        assert _title_block(filters) is None

    def test_whitespace_only_title_emits_no_title_block(self):
        filters = build_filters(title="   ", company="wiz")
        assert _title_block(filters) is None

    def test_no_conditions_returns_empty_dict(self):
        # Sanity: with nothing at all, build_filters returns {}.
        assert build_filters() == {}


# ---------------------------------------------------------------------------
# Other fields are NOT touched by this change.
# ---------------------------------------------------------------------------


class TestOtherFieldsUntouched:
    def test_skill_groups_still_target_skills_column(self):
        filters = build_filters(title="dev", skill_groups=["python, django"])
        root = _root(filters)
        conds = _all_conditions(root)
        # Find the skills OR-block.
        skill_block = next(
            (
                c
                for c in conds
                if c.get("op") == "or"
                and all(
                    inner.get("column") == "skills"
                    for inner in c.get("conditions", [])
                )
            ),
            None,
        )
        assert skill_block is not None
        skill_vals = {p["value"] for p in skill_block["conditions"]}
        assert skill_vals == {"python", "django"}

    def test_keywords_still_span_headline_summary_skills(self):
        filters = build_filters(title="dev", keywords="kubernetes")
        root = _root(filters)
        conds = _all_conditions(root)
        # Keywords block ORs across headline + summary + skills for each term.
        kw_block = next(
            (
                c
                for c in conds
                if c.get("op") == "or"
                and {inner.get("column") for inner in c.get("conditions", [])}
                == {"headline", "summary", "skills"}
            ),
            None,
        )
        assert kw_block is not None
        assert all(p["value"] == "kubernetes" for p in kw_block["conditions"])

    def test_location_multi_value_still_emits_region_or(self):
        filters = build_filters(title="dev", location="Tel Aviv, Haifa")
        root = _root(filters)
        conds = _all_conditions(root)
        region_or = [
            c
            for c in conds
            if c.get("op") == "or"
            and all(
                inner.get("column") == "region"
                for inner in c.get("conditions", [])
            )
        ]
        assert len(region_or) == 1
        assert len(region_or[0]["conditions"]) == 2
        assert {p["value"] for p in region_or[0]["conditions"]} == {
            "Tel Aviv",
            "Haifa",
        }

    def test_single_company_is_bare_predicate_not_or(self):
        """The company filter is not part of this change — assert its
        existing shape for a single value stays a bare predicate on
        current_employers.name."""
        filters = build_filters(title="dev", company="wiz")
        root = _root(filters)
        conds = _all_conditions(root)
        company_preds = [
            c for c in conds if c.get("column") == "current_employers.name"
        ]
        assert len(company_preds) == 1
        assert company_preds[0]["value"] == "wiz"
        assert company_preds[0]["type"] == "[.]"
