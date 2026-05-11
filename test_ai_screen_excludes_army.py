"""
Regression tests: AI Screen MUST exclude Israeli military service from
years-of-experience limits.

Chen reported (post Issue 7) that the AI Screen was still returning
IDF-veteran profiles like Ohad Kedem
(https://www.linkedin.com/in/ohad-kedem-91a806122/) as relevant for roles
with "max 5 years experience" / "min 1 year at current company" rules,
even though the project CLAUDE.md says the screening pipeline splits
TOTAL CAREER SPAN (with military) from INDUSTRY EXPERIENCE (without
military) and uses INDUSTRY for "reject >N years" rules.

A previous attempt (closed PR #17) fixed the wrong surface — the results-
table "Exp" display column — and regressed overlap-heavy careers
(Gil Katz showing 55y). The fix in THIS PR belongs in the AI Screen
prompt path, not in the display.

These tests are prompt-content / pure-Python checks. They do NOT call
an LLM, do NOT call Crustdata, do NOT need network access. They lock
in:

  1. The shared MILITARY_KEYWORDS set covers IDF / 8200 / Mamram / Talpiot
     / C4I and Hebrew variants (צה"ל / צבא / ממר"ם / תלפיות / מודיעין).
  2. is_military_position() recognizes those keywords in either the role
     title or the employer name, and ignores civilian "security" /
     "defense" companies.
  3. compute_role_durations() (dashboard.py) actually emits both
     TOTAL CAREER SPAN and INDUSTRY EXPERIENCE, EXCLUDES military
     months from INDUSTRY EXPERIENCE, and injects the explicit
     "EXPERIENCE LIMIT CHECK: compare against INDUSTRY EXPERIENCE,
     never TOTAL CAREER SPAN" instruction line.
  4. Both screening prompts — screening_policy.get_system_prompt() and
     screening_prompt_builder.STRUCTURED_SYSTEM_PROMPT — name
     INDUSTRY EXPERIENCE as the metric for tenure / experience-cap rules
     and mention the IDF / 8200 / Mamram / Talpiot / C4I keywords and
     Hebrew variants.

The intent is that future edits don't silently re-weaken military
exclusion for the AI Screen path.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from normalizers import MILITARY_KEYWORDS, is_military_position
from screening_policy import get_system_prompt
from screening_prompt_builder import STRUCTURED_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 1. Shared military keyword list — single source of truth
# ---------------------------------------------------------------------------

class TestMilitaryKeywords:
    """The MILITARY_KEYWORDS set must cover the units Crustdata typically
    surfaces for Israeli profiles, including Hebrew variants and bare
    unit numbers."""

    @pytest.mark.parametrize("keyword", [
        # English — generic
        "idf",
        "israel defense forces",
        "israeli air force",
        "israeli army",
        # Well-known IDF tech / intel units
        "unit 8200",
        "8200",  # bare — Crustdata sometimes uses just the number
        "mamram",
        "talpiot",
        "c4i",
        # Hebrew (the bug Chen flagged — previously missing)
        'צה"ל',
        "צבא",
        "מודיעין",
    ])
    def test_keyword_present(self, keyword):
        assert keyword in MILITARY_KEYWORDS, (
            f"MILITARY_KEYWORDS should include '{keyword}' so AI Screen's "
            f"EXPERIENCE SUMMARY excludes it from INDUSTRY EXPERIENCE."
        )


class TestIsMilitaryPosition:
    """Substring matching on title OR employer name. Conservative — must
    not flag civilian security / defense companies."""

    @pytest.mark.parametrize("title,company", [
        ("Software Engineer", "IDF Unit 8200"),
        ("Officer", "8200"),
        ("Captain", "Israel Defense Forces"),
        ("Developer", "Mamram"),
        ("Cadet", "Talpiot"),
        ("Intelligence Analyst", "IDF Intelligence"),
        ("Software Developer", "C4I and Cyber Defense Directorate"),
        # Hebrew employer / title
        ("מפתח", 'צה"ל'),
        ("חייל", "צבא"),
        ("Officer", "יחידה 8200"),
    ])
    def test_flags_military_roles(self, title, company):
        assert is_military_position(title, company) is True, (
            f"is_military_position({title!r}, {company!r}) must return True"
        )

    @pytest.mark.parametrize("title,company", [
        ("Software Engineer", "Wiz"),
        ("Backend Developer", "Check Point"),
        ("Security Engineer", "Palo Alto Networks"),  # civilian security
        ("Defense Counsel", "Some Law Firm"),         # civilian "defense"
        ("DevOps Engineer", "Cyberark"),
        ("Engineer", "Israel Aerospace Industries"),  # civilian aerospace
    ])
    def test_does_not_flag_civilian_roles(self, title, company):
        assert is_military_position(title, company) is False, (
            f"is_military_position({title!r}, {company!r}) must return False — "
            f"civilian role wrongly flagged as military would over-exclude yoe."
        )

    def test_handles_none_inputs(self):
        assert is_military_position(None, None) is False
        assert is_military_position("", "") is False
        assert is_military_position(None, "IDF") is True
        assert is_military_position("Captain in IDF", None) is True


# ---------------------------------------------------------------------------
# 2. compute_role_durations EXPERIENCE SUMMARY block
# ---------------------------------------------------------------------------

def _ohad_kedem_like_profile():
    """Synthetic profile that mirrors Ohad Kedem's shape: ~3 years IDF
    Unit 8200, ~4 years civilian engineering. Used to verify the
    EXPERIENCE SUMMARY block splits military vs industry years."""
    return {
        "name": "Test Ohad",
        "current_employers": [
            {
                "employee_title": "Software Engineer",
                "employer_name": "Wiz",
                "start_date": "2022-01-01",
                "end_date": None,
            }
        ],
        "past_employers": [
            {
                "employee_title": "Software Engineer",
                "employer_name": "Monday.com",
                "start_date": "2020-01-01",
                "end_date": "2022-01-01",
            },
            {
                "employee_title": "Officer",
                "employer_name": "IDF Unit 8200",
                "start_date": "2017-01-01",
                "end_date": "2020-01-01",
            },
        ],
    }


def _hebrew_idf_profile():
    """Same shape but the IDF role is labeled in Hebrew. Crustdata sometimes
    returns these. Hebrew label must still be detected as military."""
    return {
        "name": "Test Hebrew",
        "current_employers": [
            {
                "employee_title": "Backend Engineer",
                "employer_name": "Snyk",
                "start_date": "2022-01-01",
                "end_date": None,
            }
        ],
        "past_employers": [
            {
                "employee_title": "מפתח",
                "employer_name": 'צה"ל',
                "start_date": "2018-01-01",
                "end_date": "2022-01-01",
            },
        ],
    }


class TestExperienceSummaryBlock:
    """compute_role_durations must produce an EXPERIENCE SUMMARY block
    that names both metrics and tells the model to compare against
    INDUSTRY EXPERIENCE for years-of-experience rules."""

    def _compute(self, profile):
        # Import inside the test so a future refactor that moves this
        # function out of dashboard.py is caught here as well.
        from dashboard import compute_role_durations
        return compute_role_durations(profile)

    def test_block_contains_both_metrics(self):
        text = self._compute(_ohad_kedem_like_profile())
        assert "TOTAL CAREER SPAN" in text
        assert "INDUSTRY EXPERIENCE" in text
        assert "MILITARY SERVICE" in text, (
            "EXPERIENCE SUMMARY must explicitly list MILITARY SERVICE so the "
            "model knows which years were excluded."
        )

    def test_industry_experience_excludes_military_months(self):
        text = self._compute(_ohad_kedem_like_profile())
        # ~3 years military → INDUSTRY EXPERIENCE must be ~3 years shorter
        # than TOTAL CAREER SPAN.
        m_total = re.search(r"TOTAL CAREER SPAN:\s*(\d+)y\s*(\d+)m", text)
        m_industry = re.search(r"INDUSTRY EXPERIENCE.*?:\s*(\d+)y\s*(\d+)m", text)
        assert m_total, f"TOTAL CAREER SPAN missing or unparseable in:\n{text}"
        assert m_industry, f"INDUSTRY EXPERIENCE missing or unparseable in:\n{text}"
        total_months = int(m_total.group(1)) * 12 + int(m_total.group(2))
        industry_months = int(m_industry.group(1)) * 12 + int(m_industry.group(2))
        # 3 years of military (Jan 2017 → Jan 2020) ≈ 36 months excluded.
        assert total_months - industry_months >= 30, (
            f"Expected INDUSTRY EXPERIENCE to be ~3y shorter than TOTAL "
            f"CAREER SPAN, got total={total_months}m, industry={industry_months}m"
        )

    def test_block_emits_experience_limit_check(self):
        """The CLAUDE.md architecture promises an EXPERIENCE LIMIT CHECK
        verdict is injected. This pins the wording."""
        text = self._compute(_ohad_kedem_like_profile())
        assert "EXPERIENCE LIMIT CHECK" in text, (
            "compute_role_durations must emit an EXPERIENCE LIMIT CHECK "
            "line so the AI cannot trade off the rule."
        )
        # Must explicitly direct the model to INDUSTRY EXPERIENCE, not TOTAL.
        assert "INDUSTRY EXPERIENCE" in text
        assert "NEVER TOTAL CAREER SPAN" in text or "never TOTAL CAREER SPAN" in text.lower() or "never total" in text.lower(), (
            "EXPERIENCE LIMIT CHECK must explicitly forbid using TOTAL CAREER "
            "SPAN for the max-years rule. Current block:\n" + text
        )

    def test_hebrew_military_label_excluded(self):
        text = self._compute(_hebrew_idf_profile())
        assert "MILITARY SERVICE" in text, (
            "Hebrew military label (צה\"ל) must still be detected and "
            "excluded from INDUSTRY EXPERIENCE. Block was:\n" + text
        )

    def test_civilian_only_profile_unchanged(self):
        """No military positions → no MILITARY SERVICE line and
        INDUSTRY EXPERIENCE == TOTAL CAREER SPAN."""
        civilian = {
            "name": "Civilian Only",
            "current_employers": [
                {
                    "employee_title": "Software Engineer",
                    "employer_name": "Wiz",
                    "start_date": "2020-01-01",
                    "end_date": None,
                }
            ],
            "past_employers": [
                {
                    "employee_title": "Software Engineer",
                    "employer_name": "Monday.com",
                    "start_date": "2016-01-01",
                    "end_date": "2020-01-01",
                },
            ],
        }
        text = self._compute(civilian)
        assert "MILITARY SERVICE" not in text, (
            "Civilian-only profile should not produce a MILITARY SERVICE line."
        )
        # Both metrics should still be emitted with equal values.
        m_total = re.search(r"TOTAL CAREER SPAN:\s*(\d+)y\s*(\d+)m", text)
        m_industry = re.search(r"INDUSTRY EXPERIENCE:\s*(\d+)y\s*(\d+)m", text)
        assert m_total and m_industry
        assert (m_total.group(1), m_total.group(2)) == (
            m_industry.group(1), m_industry.group(2)
        ), "Civilian-only profile: TOTAL and INDUSTRY must match exactly."


# ---------------------------------------------------------------------------
# 3. screening_policy.py unified system prompt
# ---------------------------------------------------------------------------

class TestUnifiedPolicyExperienceRule:
    """The unified screening policy must name INDUSTRY EXPERIENCE as the
    metric used for years-of-experience rules and list the military
    signals so the model trusts the pre-computed flag."""

    def test_policy_names_industry_experience_for_caps(self):
        prompt = get_system_prompt()
        assert "INDUSTRY EXPERIENCE" in prompt
        # Must explicitly tie INDUSTRY EXPERIENCE to "max N years" rules.
        assert re.search(
            r'INDUSTRY EXPERIENCE.{0,300}(max\s+N|max\s+\d|reject\s*>|min\s+N|years)',
            prompt,
            re.IGNORECASE | re.DOTALL,
        ), (
            "Unified policy must explicitly tell the model to compare "
            "'max N years' / 'reject >N years' rules against INDUSTRY "
            "EXPERIENCE."
        )

    def test_policy_forbids_total_career_span_for_caps(self):
        prompt = get_system_prompt()
        # Look for an explicit "not total / never total" instruction near
        # the experience rule, not just somewhere in the prompt.
        assert re.search(
            r'(never|not)\s+TOTAL\s+CAREER\s+SPAN',
            prompt,
            re.IGNORECASE,
        ), (
            "Unified policy must explicitly forbid TOTAL CAREER SPAN as "
            "the metric for years-of-experience caps."
        )

    @pytest.mark.parametrize("signal", [
        "IDF",
        "8200",
        "Mamram",
        "Talpiot",
        "C4I",
        # Hebrew variants — these were the gap Chen flagged
        'צה"ל',
        "צבא",
        "ממר",  # ממר"ם partial — quote-mark variants must not matter
        "תלפיות",
    ])
    def test_policy_lists_military_signals(self, signal):
        prompt = get_system_prompt()
        assert signal in prompt, (
            f"Unified policy should name '{signal}' among the military "
            f"signals the pre-computed flag is built from."
        )

    def test_policy_states_military_mandatory(self):
        prompt = get_system_prompt()
        assert "mandatory" in prompt.lower(), (
            "Unified policy must explain why military years are excluded "
            "(mandatory service)."
        )


# ---------------------------------------------------------------------------
# 4. screening_prompt_builder.py STRUCTURED_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestStructuredPromptExperienceRule:
    """The structured prompt path is used for the per-requirement gate
    checking flow. It must also enforce INDUSTRY EXPERIENCE for
    experience caps and list the military signals."""

    def test_structured_prompt_names_industry_experience_for_caps(self):
        assert "INDUSTRY EXPERIENCE" in STRUCTURED_SYSTEM_PROMPT
        assert re.search(
            r'INDUSTRY EXPERIENCE.{0,400}(max\s+N|max\s+\d|reject\s*>|years)',
            STRUCTURED_SYSTEM_PROMPT,
            re.IGNORECASE | re.DOTALL,
        ), (
            "Structured prompt must explicitly tell the model to compare "
            "experience caps against INDUSTRY EXPERIENCE."
        )

    def test_structured_prompt_forbids_total_career_span_for_caps(self):
        assert re.search(
            r'(never|not)\s+TOTAL\s+CAREER\s+SPAN',
            STRUCTURED_SYSTEM_PROMPT,
            re.IGNORECASE,
        ), (
            "Structured prompt must explicitly forbid TOTAL CAREER SPAN "
            "as the metric for experience caps."
        )

    @pytest.mark.parametrize("signal", [
        "IDF",
        "8200",
        "Mamram",
        "Talpiot",
        "C4I",
        'צה"ל',
        "ממר",
        "תלפיות",
    ])
    def test_structured_prompt_lists_military_signals(self, signal):
        assert signal in STRUCTURED_SYSTEM_PROMPT, (
            f"Structured prompt should name '{signal}' among recognized "
            f"military signals."
        )
