"""Regression tests for Crustdata title-variant expansion in build_filters.

Background: combining title="fullstack developer" + current_company filter
returned 0 results because almost nobody at Wiz / Monday / Forter spells
the title "Fullstack" as one word — they use "Full Stack" or "Full-Stack".
The fix expands common compound role names into all three spellings so the
AND combo finds the candidates that actually exist.
"""

from crustdata_search import _expand_title_variants, build_filters


def test_no_variant_group_hit_returns_input_only():
    assert _expand_title_variants("team lead") == ["team lead"]
    assert _expand_title_variants("Software Engineer") == ["Software Engineer"]


def test_empty_input_passthrough():
    assert _expand_title_variants("") == [""]
    assert _expand_title_variants(None) == [None]


def test_fullstack_expands_to_three_spellings():
    out = _expand_title_variants("fullstack developer")
    assert out[0] == "fullstack developer"
    assert "full stack developer" in out
    assert "full-stack developer" in out
    assert len(out) == 3


def test_full_stack_with_space_expands_back():
    out = _expand_title_variants("full stack developer")
    assert out[0] == "full stack developer"
    assert "fullstack developer" in out
    assert "full-stack developer" in out


def test_full_dash_stack_expands_back():
    out = _expand_title_variants("Full-Stack Developer")
    assert out[0] == "Full-Stack Developer"
    folded = [v.casefold() for v in out]
    assert "fullstack developer" in folded
    assert "full stack developer" in folded


def test_case_preserved_outside_variant_segment():
    out = _expand_title_variants("Senior Fullstack Engineer")
    assert out[0] == "Senior Fullstack Engineer"
    # Variant segment substitutes the matched chunk; surrounding text untouched.
    assert "Senior full stack Engineer" in out
    assert "Senior full-stack Engineer" in out


def test_frontend_group():
    out = _expand_title_variants("frontend developer")
    assert "front end developer" in out
    assert "front-end developer" in out


def test_backend_group():
    out = _expand_title_variants("backend engineer")
    assert "back end engineer" in out
    assert "back-end engineer" in out


def test_devops_group():
    out = _expand_title_variants("devops")
    assert "dev ops" in out
    assert "dev-ops" in out


def test_build_filters_emits_or_branch_for_expanded_variants():
    """End-to-end pin: when title='fullstack developer' (single value, no
    comma), build_filters should emit an OR of three conditions targeting
    current_employers.title."""
    out = build_filters(title="fullstack developer")
    assert "filters" in out
    f = out["filters"]
    # Top-level should be an AND with one condition (the title OR group).
    if "op" in f and f["op"] == "and":
        assert len(f["conditions"]) == 1
        title_cond = f["conditions"][0]
    else:
        title_cond = f
    assert title_cond.get("op") == "or"
    values = sorted(c["value"] for c in title_cond["conditions"])
    assert values == sorted(["fullstack developer", "full stack developer", "full-stack developer"])
    for c in title_cond["conditions"]:
        assert c["column"] == "current_employers.title"
        assert c["type"] == "[.]"


def test_build_filters_combined_title_and_company_produces_and_of_or():
    """Reproduces the exact Chen scenario: title + company list. The expanded
    title variants must be OR-grouped, then AND-combined with the company OR
    group."""
    out = build_filters(
        title="fullstack developer",
        company="monday.com, forter, wiz, outbrain",
    )
    assert "filters" in out
    root = out["filters"]
    assert root.get("op") == "and"
    assert len(root["conditions"]) == 2
    title_branch = next(c for c in root["conditions"]
                        if c.get("op") == "or"
                        and all(x["column"] == "current_employers.title"
                                for x in c.get("conditions", [])))
    company_branch = next(c for c in root["conditions"]
                          if c.get("op") == "or"
                          and all(x["column"] == "current_employers.name"
                                  for x in c.get("conditions", [])))
    assert {c["value"] for c in title_branch["conditions"]} == {
        "fullstack developer", "full stack developer", "full-stack developer"
    }
    assert {c["value"] for c in company_branch["conditions"]} == {
        "monday.com", "forter", "wiz", "outbrain"
    }


def test_build_filters_dedupes_when_user_supplies_multiple_variants():
    """If the user already typed two variants comma-separated, expansion
    must not duplicate them in the OR group."""
    out = build_filters(title="fullstack, full stack")
    f = out["filters"]
    # Walk down to the title OR branch
    if f.get("op") == "and":
        title_cond = f["conditions"][0]
    else:
        title_cond = f
    values = [c["value"] for c in title_cond["conditions"]]
    # Case-folded dedup means we keep one of each spelling group entry.
    folded = sorted(v.casefold() for v in values)
    assert folded == sorted(["fullstack", "full stack", "full-stack"])
    assert len(values) == len(set(v.casefold() for v in values))
