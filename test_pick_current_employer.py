"""Regression tests for normalizers.pick_current_employer.

Crustdata returns current_employers as a list that can contain multiple
active roles (advisor + full-time, two current jobs after a switch, IDF
reserve + civilian role). The list is not guaranteed to be ordered by
recency, so picking [0] blindly surfaces the WRONG current employer.

These tests pin two real-world scenarios that surfaced the bug:

- Gil Gitlin: Unit 8200 (older) + Cytactic (since 2025-05). SourcingX stored
  Unit 8200 as current_company because Crustdata returned 8200 at [0].
- Barak Ben Shimon: Vesttoo (since 2022-12) + monday.com (since 2023-10).
  SourcingX stored Vesttoo for the same reason.
"""

import json

from normalizers import pick_current_employer


def test_returns_none_for_non_list_inputs():
    assert pick_current_employer(None) is None
    assert pick_current_employer({}) is None
    assert pick_current_employer("not a list") is None
    assert pick_current_employer(0) is None


def test_returns_none_for_empty_list():
    assert pick_current_employer([]) is None


def test_returns_none_when_no_dict_entries():
    assert pick_current_employer([None, "", 0]) is None


def test_returns_single_entry_unchanged():
    entry = {'employer_name': 'Acme', 'start_date': '2020-01-01'}
    assert pick_current_employer([entry]) is entry


def test_picks_most_recent_by_start_date_descending():
    older = {'employer_name': 'OldCo', 'start_date': '2018-03-01T00:00:00+00:00'}
    newer = {'employer_name': 'NewCo', 'start_date': '2023-10-01T00:00:00+00:00'}
    assert pick_current_employer([older, newer]) is newer
    # Order in the input list must not matter
    assert pick_current_employer([newer, older]) is newer


def test_missing_start_date_sorts_last():
    no_date = {'employer_name': 'NoDateCo'}
    with_date = {'employer_name': 'DatedCo', 'start_date': '2020-01-01'}
    assert pick_current_employer([no_date, with_date]) is with_date
    assert pick_current_employer([with_date, no_date]) is with_date


def test_gil_gitlin_scenario():
    """Gil's Crustdata profile lists Unit 8200 (older) + Cytactic (since
    2025-05). The fix should surface Cytactic, not 8200, as the current job."""
    unit_8200 = {
        'employer_name': 'Unit 8200 - Israeli Intelligence Corps',
        'employee_title': 'Full-stack Developer',
        'start_date': '2019-01-01T00:00:00+00:00',
    }
    cytactic = {
        'employer_name': 'Cytactic',
        'employee_title': 'Senior Software Engineer',
        'start_date': '2025-05-01T00:00:00+00:00',
    }
    assert pick_current_employer([unit_8200, cytactic])['employer_name'] == 'Cytactic'
    # Order independence: Crustdata sometimes returns reserve first.
    assert pick_current_employer([cytactic, unit_8200])['employer_name'] == 'Cytactic'


def test_barak_ben_shimon_scenario():
    """Barak's Crustdata profile lists Vesttoo (since 2022-12) + monday.com
    (since 2023-10). The fix should surface monday.com."""
    vesttoo = {
        'employer_name': 'Vesttoo',
        'employee_title': 'Full Stack Developer',
        'start_date': '2022-12-01T00:00:00+00:00',
    }
    monday = {
        'employer_name': 'monday.com',
        'employee_title': 'Senior Software Engineer',
        'start_date': '2023-10-01T00:00:00+00:00',
    }
    assert pick_current_employer([vesttoo, monday])['employer_name'] == 'monday.com'
    assert pick_current_employer([monday, vesttoo])['employer_name'] == 'monday.com'


def test_filters_non_dict_entries():
    valid = {'employer_name': 'Acme', 'start_date': '2020-01-01'}
    assert pick_current_employer([None, valid, "garbage", 42]) is valid


def test_unparseable_string_start_date_sorts_last():
    """Non-ISO start_date values ("Present", "May 2025") must sort LAST so
    they never win against a real ISO date. Lexicographic string sort would
    fail here because "P"/"M" > "2" in ASCII."""
    present = {'employer_name': 'PresentCo', 'start_date': 'Present'}
    real_iso = {'employer_name': 'RealCo', 'start_date': '2020-01-01'}
    assert pick_current_employer([present, real_iso])['employer_name'] == 'RealCo'
    assert pick_current_employer([real_iso, present])['employer_name'] == 'RealCo'

    localized = {'employer_name': 'LocCo', 'start_date': 'May 2025'}
    iso_2024 = {'employer_name': 'IsoCo', 'start_date': '2024-06-01'}
    assert pick_current_employer([localized, iso_2024])['employer_name'] == 'IsoCo'
    assert pick_current_employer([iso_2024, localized])['employer_name'] == 'IsoCo'


def test_parses_year_month_short_format():
    """Crustdata short forms like "2025-07" must parse and compare correctly
    against full ISO timestamps."""
    short = {'employer_name': 'ShortCo', 'start_date': '2025-07'}
    older_full = {'employer_name': 'OlderCo', 'start_date': '2020-01-01T00:00:00+00:00'}
    assert pick_current_employer([short, older_full])['employer_name'] == 'ShortCo'
    assert pick_current_employer([older_full, short])['employer_name'] == 'ShortCo'


def test_parses_year_only_format():
    """Year-only ("2024") must parse and compare correctly."""
    year_only = {'employer_name': 'YearCo', 'start_date': '2024'}
    older = {'employer_name': 'OlderCo', 'start_date': '2023-12-31'}
    assert pick_current_employer([year_only, older])['employer_name'] == 'YearCo'


def test_parses_iso_with_z_suffix():
    """ISO with 'Z' (UTC) and ISO with explicit '+00:00' must both parse."""
    with_z = {'employer_name': 'ZCo', 'start_date': '2025-05-01T00:00:00Z'}
    with_offset = {'employer_name': 'OffsetCo', 'start_date': '2020-05-01T00:00:00+00:00'}
    assert pick_current_employer([with_z, with_offset])['employer_name'] == 'ZCo'


def test_naive_and_aware_datetimes_compare_without_typeerror():
    """Mixed tz-aware ISO and naive year-month strings must not raise
    TypeError when sorted. The parser strips tzinfo to keep all datetimes
    naive."""
    aware = {'employer_name': 'AwareCo', 'start_date': '2025-05-01T00:00:00+00:00'}
    naive = {'employer_name': 'NaiveCo', 'start_date': '2024-01'}
    result = pick_current_employer([aware, naive])
    assert result['employer_name'] == 'AwareCo'


def test_tie_break_is_deterministic_on_equal_dates():
    """When two entries have the same start_date, stable sort preserves input
    order. The first one in the input wins."""
    first = {'employer_name': 'FirstIn', 'start_date': '2024-01-01'}
    second = {'employer_name': 'SecondIn', 'start_date': '2024-01-01'}
    assert pick_current_employer([first, second])['employer_name'] == 'FirstIn'
    assert pick_current_employer([second, first])['employer_name'] == 'SecondIn'


def test_two_unparseable_dates_tie_break_is_stable():
    """When both entries are unparseable, stable sort preserves input order."""
    a = {'employer_name': 'A', 'start_date': 'Present'}
    b = {'employer_name': 'B', 'start_date': 'who knows'}
    assert pick_current_employer([a, b])['employer_name'] == 'A'
    assert pick_current_employer([b, a])['employer_name'] == 'B'


def test_normalize_crustdata_profile_uses_most_recent():
    """End-to-end pin: normalize_crustdata_profile should surface the most
    recent current employer in current_company / current_title."""
    from normalizers import normalize_crustdata_profile

    raw = {
        'linkedin_profile_url': 'https://www.linkedin.com/in/gil-gitlin-87b720200',
        'first_name': 'Gil',
        'last_name': 'Gitlin',
        'current_employers': [
            {
                'employer_name': 'Unit 8200 - Israeli Intelligence Corps',
                'employee_title': 'Full-stack Developer',
                'start_date': '2019-01-01T00:00:00+00:00',
            },
            {
                'employer_name': 'Cytactic',
                'employee_title': 'Senior Software Engineer',
                'start_date': '2025-05-01T00:00:00+00:00',
            },
        ],
    }
    result = normalize_crustdata_profile(raw)
    assert result is not None
    assert result['current_company'] == 'Cytactic'
    assert result['current_title'] == 'Senior Software Engineer'


# ---------------------------------------------------------------------------
# email_generator.trim_profile_for_email — new Crustdata profile shape
# ---------------------------------------------------------------------------
#
# Stored Crustdata profiles come in two job-entry shapes. OLD (legacy
# /screener/* endpoints): employee_title, employer_name, employee_description,
# employer_linkedin_description. NEW (~88% of stored profiles and ALL results
# of the app's new search): title, name (company), description; location at
# top level in `region` (not `location`).
#
# email_generator.py's trim_profile_for_email() used to read only the old
# keys, so every new-format job was silently dropped, and
# `trimmed.get('current_employers', [{}])[0]` raised IndexError once
# current_employers existed but ended up empty after filtering. These tests
# pin the fix.

from email_generator import trim_profile_for_email


def test_trim_profile_for_email_reads_new_format_jobs_and_location():
    """A new-shape profile keeps its current and past jobs plus location,
    and does not raise."""
    raw = {
        'name': 'Dana Cohen',
        'region': 'Tel Aviv, Israel',
        'current_employers': [
            {
                'title': 'Senior Backend Engineer',
                'name': 'Wiz',
                'description': 'Building cloud security infrastructure.',
                'start_date': '2022-01-01T00:00:00',
            }
        ],
        'past_employers': [
            {
                'title': 'Backend Engineer',
                'name': 'Monday.com',
                'description': 'Worked on the automations platform.',
                'start_date': '2020-01-01T00:00:00',
                'end_date': '2021-12-01T00:00:00',
            }
        ],
    }

    trimmed = trim_profile_for_email(raw)

    assert trimmed['location'] == 'Tel Aviv, Israel'

    assert len(trimmed['current_employers']) == 1
    current = trimmed['current_employers'][0]
    assert current['title'] == 'Senior Backend Engineer'
    assert current['company'] == 'Wiz'
    assert current['role_description'] == 'Building cloud security infrastructure.'

    assert len(trimmed['past_employers']) == 1
    past = trimmed['past_employers'][0]
    assert past['title'] == 'Backend Engineer'
    assert past['company'] == 'Monday'  # normalize_company_name strips ".com"
    assert past['role_description'] == 'Worked on the automations platform.'


def test_trim_profile_for_email_old_format_location_still_wins():
    """Old-shape location field takes priority over region if both exist
    (region should never override an explicit location)."""
    raw = {
        'name': 'Old Format',
        'location': 'Haifa, Israel',
        'region': 'Should not be used',
    }
    trimmed = trim_profile_for_email(raw)
    assert trimmed['location'] == 'Haifa, Israel'


def test_trim_profile_for_email_title_less_current_employers_does_not_raise():
    """A profile whose current_employers entries are all title-less (neither
    old nor new title key present) must not raise IndexError when the
    career-pattern detection reads trimmed['current_employers'][0]."""
    raw = {
        'name': 'No Title',
        'current_employers': [
            {'name': 'Ghost Co', 'start_date': '2023-01-01'},
        ],
        'past_employers': [
            {'title': 'Engineer', 'name': 'Real Co', 'start_date': '2019-01-01', 'end_date': '2022-01-01'},
        ],
    }

    # Must not raise.
    trimmed = trim_profile_for_email(raw)

    assert trimmed['current_employers'] == []


def test_trim_profile_for_email_old_format_unchanged():
    """Old-shape output is unchanged by the new-shape fallback logic."""
    raw = {
        'name': 'Old Format Person',
        'location': 'Herzliya, Israel',
        'current_employers': [
            {
                'employee_title': 'VP Engineering',
                'employer_name': 'Check Point Software Technologies',
                'employee_description': 'Leads the platform org.',
                'employer_linkedin_description': 'Cyber security vendor. Founded 1993.',
                'start_date': '2019-01-01T00:00:00',
            }
        ],
        'past_employers': [
            {
                'employee_title': 'Director',
                'employer_name': 'CyberArk',
                'start_date': '2015-01-01T00:00:00',
                'end_date': '2021-12-01T00:00:00',
            }
        ],
    }

    trimmed = trim_profile_for_email(raw)

    assert trimmed['location'] == 'Herzliya, Israel'
    current = trimmed['current_employers'][0]
    assert current['title'] == 'VP Engineering'
    assert current['company'] == 'Check Point'
    assert current['role_description'] == 'Leads the platform org.'
    assert current['company_description'] == 'Cyber security vendor.'

    past = trimmed['past_employers'][0]
    assert past['title'] == 'Director'
    assert past['company'] == 'CyberArk'


# ===== Opener concrete-detail fixes (build_email_prompt / _opener_violations / retry) =====

from email_generator import (
    build_email_prompt,
    _opener_violations,
    _split_position_company,
    _extract_company_from_context,
    generate_email_for_profile,
)


def test_extract_company_from_context_em_dash_separator():
    assert _extract_company_from_context(
        'Orca Security — agentless cloud security platform, competitor to Wiz'
    ) == 'Orca Security'


def test_extract_company_from_context_hyphen_separator():
    assert _extract_company_from_context(
        'Wiz - cloud security platform'
    ) == 'Wiz'


def test_extract_company_from_context_colon_separator():
    assert _extract_company_from_context('Monday.com: work OS platform') == 'Monday.com'


def test_extract_company_from_context_plain_company_name_only():
    assert _extract_company_from_context('Wiz') == 'Wiz'


def test_extract_company_from_context_full_sentence_no_separator_returns_none():
    long_sentence = 'A cloud security platform that helps enterprises find and fix risks fast'
    assert len(long_sentence) > 40
    assert _extract_company_from_context(long_sentence) is None


def test_extract_company_from_context_empty_or_none():
    assert _extract_company_from_context('') is None
    assert _extract_company_from_context(None) is None
    assert _extract_company_from_context('   ') is None


def test_extract_company_from_context_reaches_prompt_via_generate_emails_batch(monkeypatch):
    """End-to-end check for point 1: the extracted company reaches the
    opener prompt the same way dashboard.py wires it up (extract, then pass
    as `company=` to generate_emails_batch)."""
    import email_generator

    captured = {}

    class _Capturing(_CapturingOpenAIClient):
        def _create(self, **kwargs):
            for msg in kwargs.get('messages', []):
                if msg.get('role') == 'system':
                    captured['system_prompt'] = msg.get('content')
            return super()._create(**kwargs)

    monkeypatch.setattr(email_generator, 'OpenAI', lambda api_key=None: _Capturing([CLEAN_OPENER]))

    company_context = 'Orca Security — agentless cloud security platform, competitor to Wiz'
    extracted_company = _extract_company_from_context(company_context)

    email_generator.generate_emails_batch(
        [THIN_PROFILE],
        api_key='test-key',
        generate_type='opener_only',
        ai_provider='openai',
        company=extracted_company
    )

    assert 'at Orca Security' in captured['system_prompt']


def test_position_with_at_company_splits_into_role_and_company():
    prompt = build_email_prompt('recruiter', 'professional', 'medium', position='Applied AI Engineer at Dwelly')
    assert 'at Dwelly' in prompt
    assert '**Applied AI Engineer**' in prompt
    assert 'at a tech company' not in prompt


def test_position_without_at_stays_neutral():
    prompt = build_email_prompt('recruiter', 'professional', 'medium', position='DevOps Engineer')
    assert 'at a tech company' in prompt
    assert '**DevOps Engineer**' in prompt


def test_explicit_company_overrides_position_text():
    prompt = build_email_prompt(
        'recruiter', 'professional', 'medium',
        position='Applied AI Engineer at Dwelly', company='Wiz'
    )
    assert 'at Wiz' in prompt
    assert 'a tech company' not in prompt
    # Position text is passed through unsplit since an explicit company won.
    assert '**Applied AI Engineer at Dwelly**' in prompt


def test_split_position_company_handles_trailing_parenthetical():
    role, company = _split_position_company('Backend Engineer at Monday.com (remote)')
    assert role == 'Backend Engineer'
    assert company == 'Monday.com'


def test_split_position_company_no_at_returns_unchanged():
    assert _split_position_company('DevOps Engineer') == ('DevOps Engineer', None)


def test_split_position_company_none_returns_none():
    assert _split_position_company(None) == (None, None)


def test_prompt_has_no_israeli_tech_company_wording():
    prompt = build_email_prompt('recruiter', 'professional', 'medium', company='Wiz')
    assert 'Israeli tech company' not in prompt


def test_prompt_includes_passed_company_and_sender():
    prompt = build_email_prompt('recruiter', 'professional', 'medium', company='Wiz')
    assert 'Wiz' in prompt
    assert 'recruiter' in prompt


def test_prompt_neutral_wording_when_company_none():
    prompt = build_email_prompt('recruiter', 'professional', 'medium', company=None)
    assert 'at a tech company' in prompt
    assert 'Israeli tech company' not in prompt


def test_prompt_contains_concrete_detail_instruction_and_bans():
    prompt = build_email_prompt('recruiter', 'professional', 'medium')
    assert 'CONCRETE DETAIL' in prompt
    assert 'NEVER start opener with "Your"' in prompt
    assert 'aligns' in prompt
    assert 'mission' in prompt


def test_opener_violations_flags_your_start():
    assert _opener_violations('Your work at CyberArk stands out.') == ['starts with "Your"']


def test_opener_violations_flags_aligns_well():
    violations = _opener_violations('This role aligns well with your background.')
    assert any('aligns' in v for v in violations)


def test_opener_violations_flags_the_mission():
    violations = _opener_violations('Excited about the mission you are building toward.')
    assert any('mission' in v for v in violations)


def test_opener_violations_clean_opener_returns_empty():
    assert _opener_violations(
        "Building the migration tooling that moved CyberArk's data pipeline to Kubernetes is the kind of hands-on work we need."
    ) == []


class _StubOpenAIResponse:
    def __init__(self, content, prompt_tokens=10, completion_tokens=5):
        message = type('Msg', (), {'content': content})()
        choice = type('Choice', (), {'message': message})()
        usage = type('Usage', (), {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})()
        self.choices = [choice]
        self.usage = usage


class _StubOpenAIClient:
    """Stub OpenAI-shaped client that returns queued responses in order."""

    def __init__(self, contents):
        self._queue = list(contents)
        self.call_count = 0
        chat = type('Chat', (), {})()
        completions = type('Completions', (), {})()
        completions.create = self._create
        chat.completions = completions
        self.chat = chat

    def _create(self, **kwargs):
        self.call_count += 1
        content = self._queue.pop(0)
        return _StubOpenAIResponse(content)


def _profile_with_data():
    return {
        'raw_data': {
            'name': 'Test Candidate',
            'current_employers': [
                {'title': 'Backend Engineer', 'employer_name': 'CyberArk', 'start_date': '2020-01-01'}
            ],
            'skills': ['Python'],
        }
    }


VIOLATING_OPENER = json.dumps({
    "email_opener": "Your work at CyberArk aligns well with the mission here.",
    "opener_angle": "career"
})

CLEAN_OPENER = json.dumps({
    "email_opener": "Building the migration tooling that moved CyberArk's pipeline to Kubernetes shows real depth.",
    "opener_angle": "career"
})


def test_retry_returns_clean_opener_after_one_violation_exactly_two_calls():
    client = _StubOpenAIClient([VIOLATING_OPENER, CLEAN_OPENER])
    result = generate_email_for_profile(
        _profile_with_data(), client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert client.call_count == 2
    assert result['email_opener'] == json.loads(CLEAN_OPENER)['email_opener']


def test_retry_still_violating_returns_empty_opener_with_opener_error():
    client = _StubOpenAIClient([VIOLATING_OPENER, VIOLATING_OPENER])
    result = generate_email_for_profile(
        _profile_with_data(), client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert client.call_count == 2
    assert result['email_opener'] == ''
    assert 'Opener broke the writing rules twice' in result['opener_error']
    assert 'error' not in result


class _StubOpenAIClientRetryRaises:
    """Stub client whose FIRST call returns a queued response and whose
    SECOND call (the corrective retry) raises, to simulate the retry
    itself failing after a bad first opener."""

    def __init__(self, first_content):
        self._first_content = first_content
        self.call_count = 0
        chat = type('Chat', (), {})()
        completions = type('Completions', (), {})()
        completions.create = self._create
        chat.completions = completions
        self.chat = chat

    def _create(self, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return _StubOpenAIResponse(self._first_content)
        raise RuntimeError('simulated API failure on retry')


def test_retry_exception_with_bad_first_returns_empty_opener_with_error():
    client = _StubOpenAIClientRetryRaises(VIOLATING_OPENER)
    result = generate_email_for_profile(
        _profile_with_data(), client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert client.call_count == 2
    assert result['email_opener'] == ''
    # The retry CALL failed (simulated API error) - the second opener was
    # never actually checked, so this must NOT say "broke the rules twice".
    assert result['opener_error'].startswith('Retry failed:')
    assert 'simulated API failure' in result['opener_error']
    assert 'error' not in result


class _FakeTracker:
    """Records log_openai calls so tests can assert error logging happened."""

    def __init__(self):
        self.calls = []

    def log_openai(self, **kwargs):
        self.calls.append(kwargs)


def test_retry_exception_logs_through_usage_tracker():
    client = _StubOpenAIClientRetryRaises(VIOLATING_OPENER)
    tracker = _FakeTracker()
    generate_email_for_profile(
        _profile_with_data(), client,
        generate_type='opener_only', ai_provider='openai', tracker=tracker
    )
    error_calls = [c for c in tracker.calls if c.get('status') == 'error']
    assert len(error_calls) == 1
    assert 'simulated API failure' in error_calls[0]['error_message']


VIOLATING_BOTH = json.dumps({
    "subject_line": "Kubernetes at CyberArk?",
    "subject_angle": "company",
    "email_opener": "Your work at CyberArk aligns well with the mission here.",
    "opener_angle": "career"
})


def test_retry_still_violating_keeps_subject_line_when_generated():
    client = _StubOpenAIClient([VIOLATING_BOTH, VIOLATING_BOTH])
    result = generate_email_for_profile(
        _profile_with_data(), client,
        generate_type='both', ai_provider='openai'
    )
    assert client.call_count == 2
    assert result['email_opener'] == ''
    assert 'Opener broke the writing rules twice' in result['opener_error']
    assert result['subject_line'] == 'Kubernetes at CyberArk?'


def test_clean_opener_first_try_makes_only_one_call():
    client = _StubOpenAIClient([CLEAN_OPENER])
    result = generate_email_for_profile(
        _profile_with_data(), client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert client.call_count == 1
    assert result['email_opener'] == json.loads(CLEAN_OPENER)['email_opener']


# ===== Codex round 1 fixes: fuller violation coverage + fabrication guard =====

from email_generator import _has_descriptive_text


def test_opener_violations_flags_leading_start():
    violations = _opener_violations('Leading the platform rewrite at CyberArk stands out.')
    assert any('Leading' in v for v in violations)


def test_opener_violations_flags_handling_start():
    violations = _opener_violations('Handling the migration to Kubernetes at CyberArk is notable.')
    assert any('Handling' in v for v in violations)


def test_opener_violations_flags_exciting():
    violations = _opener_violations('The exciting work at CyberArk stands out.')
    assert any('exciting' in v for v in violations)


def test_opener_violations_flags_exclamation_mark():
    violations = _opener_violations('The migration work at CyberArk is great!')
    assert any('exclamation' in v for v in violations)


def test_opener_violations_no_false_hit_on_submission_or_permission():
    # "mission" is banned, but must not match inside "submission"/"permission".
    violations = _opener_violations(
        'The pull request submission process and permission model they built at CyberArk stand out.'
    )
    assert violations == []


def test_opener_violations_no_false_hit_on_dynamically():
    # "dynamic" is banned, but must not match inside "dynamically".
    violations = _opener_violations('They configured the pipeline dynamically at CyberArk.')
    assert violations == []


def test_prompt_never_use_lines_render_from_shared_constant():
    prompt = build_email_prompt('recruiter', 'professional', 'medium')
    from email_generator import OPENER_NEVER_USE_PHRASES, OPENER_FORBIDDEN_STARTS
    for phrase in OPENER_NEVER_USE_PHRASES:
        assert phrase in prompt
    for start_word in OPENER_FORBIDDEN_STARTS:
        assert start_word in prompt


def test_prompt_contains_no_invention_rule():
    prompt = build_email_prompt('recruiter', 'professional', 'medium')
    assert 'do NOT invent' in prompt or 'NEVER invent' in prompt


def test_has_descriptive_text_true_with_summary():
    assert _has_descriptive_text({'summary': 'Built the payments platform.'}) is True


def test_has_descriptive_text_true_with_role_description():
    assert _has_descriptive_text({
        'current_employers': [{'title': 'Engineer', 'role_description': 'Led the migration.'}]
    }) is True


def test_has_descriptive_text_false_when_only_titles_and_skills():
    assert _has_descriptive_text({
        'current_employers': [{'title': 'Backend Engineer', 'company': 'CyberArk'}],
        'skills': ['Python'],
    }) is False


def test_has_descriptive_text_false_with_only_company_description():
    """Codex round 5: company_description is the EMPLOYER's own blurb, not
    anything the candidate did, so it must NOT count as descriptive text."""
    assert _has_descriptive_text({
        'current_employers': [{
            'title': 'Backend Engineer',
            'company': 'CyberArk',
            'company_description': 'Cyber security vendor.',
        }],
    }) is False


def test_has_descriptive_text_false_with_only_company_description_on_past_role():
    assert _has_descriptive_text({
        'current_employers': [{'title': 'Backend Engineer', 'company': 'CyberArk'}],
        'past_employers': [{
            'title': 'Engineer',
            'company': 'OldCo',
            'company_description': 'Enterprise software company.',
        }],
    }) is False


def test_profile_with_only_company_description_gets_thin_profile_note():
    client = _CapturingOpenAIClient([CLEAN_OPENER])
    profile = {
        'raw_data': {
            'name': 'Company Blurb Candidate',
            'current_employers': [
                {
                    'employee_title': 'Backend Engineer',
                    'employer_name': 'CyberArk',
                    'employer_linkedin_description': 'Cyber security vendor.',
                    'start_date': '2023-01-01',
                }
            ],
            'skills': ['Python'],
        }
    }
    generate_email_for_profile(
        profile, client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert 'do not claim anything they built' in client.last_user_prompt


def test_has_descriptive_text_true_with_past_role_description_only():
    assert _has_descriptive_text({
        'current_employers': [{'title': 'Backend Engineer', 'company': 'CyberArk'}],
        'past_employers': [{'title': 'Engineer', 'company': 'OldCo', 'role_description': 'Led the migration.'}],
    }) is True


def test_trim_profile_for_email_keeps_past_role_description_new_shape():
    """Codex round 3: a profile whose ONLY description lives on a past role
    (new shape: 'description' field) must not be flagged as thin, and the
    description must survive trimming."""
    raw = {
        'name': 'New Shape Candidate',
        'current_employers': [
            {'title': 'Backend Engineer', 'name': 'CyberArk', 'start_date': '2023-01-01'}
        ],
        'past_employers': [
            {
                'title': 'Software Engineer',
                'name': 'OldCo',
                'description': 'Migrated the billing pipeline to Kubernetes.',
                'start_date': '2021-01-01',
                'end_date': '2022-12-01',
            }
        ],
    }

    trimmed = trim_profile_for_email(raw)

    past = trimmed['past_employers'][0]
    assert past['role_description'] == 'Migrated the billing pipeline to Kubernetes.'
    assert _has_descriptive_text(trimmed) is True


def test_trim_profile_for_email_keeps_past_role_description_old_shape():
    """Same as above but for the old shape ('employee_description' field
    on 'employer_name'-keyed entries)."""
    raw = {
        'name': 'Old Shape Candidate',
        'current_employers': [
            {'employee_title': 'Backend Engineer', 'employer_name': 'CyberArk', 'start_date': '2023-01-01'}
        ],
        'past_employers': [
            {
                'employee_title': 'Software Engineer',
                'employer_name': 'OldCo',
                'employee_description': 'Migrated the billing pipeline to Kubernetes.',
                'start_date': '2021-01-01',
                'end_date': '2022-12-01',
            }
        ],
    }

    trimmed = trim_profile_for_email(raw)

    past = trimmed['past_employers'][0]
    assert past['role_description'] == 'Migrated the billing pipeline to Kubernetes.'
    assert _has_descriptive_text(trimmed) is True


def test_trim_profile_for_email_past_role_before_2021_stays_excluded():
    """The existing 'no companies before 2021' rule must still drop past
    roles that ended before 2021, description or not."""
    raw = {
        'name': 'Old Job Candidate',
        'current_employers': [
            {'title': 'Backend Engineer', 'name': 'CyberArk', 'start_date': '2023-01-01'}
        ],
        'past_employers': [
            {
                'title': 'Junior Engineer',
                'name': 'AncientCo',
                'description': 'Built the original monolith.',
                'start_date': '2015-01-01',
                'end_date': '2018-01-01',
            }
        ],
    }

    trimmed = trim_profile_for_email(raw)

    assert trimmed['past_employers'] == []
    assert _has_descriptive_text(trimmed) is False


THIN_PROFILE = {
    'raw_data': {
        'name': 'Thin Candidate',
        'current_employers': [
            {'title': 'Backend Engineer', 'employer_name': 'CyberArk', 'start_date': '2020-01-01'}
        ],
        'skills': ['Python'],
    }
}


class _CapturingOpenAIClient:
    """Stub client that records the user prompt it was called with."""

    def __init__(self, contents):
        self._queue = list(contents)
        self.call_count = 0
        self.last_user_prompt = None
        chat = type('Chat', (), {})()
        completions = type('Completions', (), {})()
        completions.create = self._create
        chat.completions = completions
        self.chat = chat

    def _create(self, **kwargs):
        self.call_count += 1
        for msg in kwargs.get('messages', []):
            if msg.get('role') == 'user':
                self.last_user_prompt = msg.get('content')
        content = self._queue.pop(0)
        return _StubOpenAIResponse(content)


def test_thin_profile_gets_no_invention_note_in_user_prompt():
    client = _CapturingOpenAIClient([CLEAN_OPENER])
    generate_email_for_profile(
        THIN_PROFILE, client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert 'do not claim anything they built' in client.last_user_prompt


def test_generate_emails_batch_passes_company_through_to_prompt(monkeypatch):
    import email_generator

    captured = {}

    class _CapturingClientForBatch(_CapturingOpenAIClient):
        def _create(self, **kwargs):
            for msg in kwargs.get('messages', []):
                if msg.get('role') == 'system':
                    captured['system_prompt'] = msg.get('content')
            return super()._create(**kwargs)

    def _fake_openai(api_key=None):
        return _CapturingClientForBatch([CLEAN_OPENER])

    monkeypatch.setattr(email_generator, 'OpenAI', _fake_openai)

    results = email_generator.generate_emails_batch(
        [THIN_PROFILE],
        api_key='test-key',
        generate_type='opener_only',
        ai_provider='openai',
        company='Acme'
    )

    assert len(results) == 1
    assert 'at Acme' in captured['system_prompt']


def test_profile_with_only_past_role_description_gets_no_thin_profile_note():
    client = _CapturingOpenAIClient([CLEAN_OPENER])
    profile = {
        'raw_data': {
            'name': 'Past Description Candidate',
            'current_employers': [
                {'title': 'Backend Engineer', 'employer_name': 'CyberArk', 'start_date': '2023-01-01'}
            ],
            'past_employers': [
                {
                    'title': 'Software Engineer',
                    'employer_name': 'OldCo',
                    'employee_description': 'Migrated the billing pipeline to Kubernetes.',
                    'start_date': '2021-01-01',
                    'end_date': '2022-12-01',
                }
            ],
            'skills': ['Python'],
        }
    }
    generate_email_for_profile(
        profile, client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert 'do not claim anything they built' not in client.last_user_prompt


def test_profile_with_descriptions_gets_no_thin_profile_note():
    client = _CapturingOpenAIClient([CLEAN_OPENER])
    profile = {
        'raw_data': {
            'name': 'Rich Candidate',
            'current_employers': [
                {'title': 'Backend Engineer', 'employer_name': 'CyberArk', 'start_date': '2020-01-01',
                 'description': 'Led the migration of the payments pipeline to Kubernetes.'}
            ],
            'skills': ['Python'],
        }
    }
    generate_email_for_profile(
        profile, client,
        generate_type='opener_only', ai_provider='openai'
    )
    assert 'do not claim anything they built' not in client.last_user_prompt
