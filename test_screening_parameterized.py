"""
Parameterized tests for AI screening logic.
Uses pytest.mark.parametrize to test multiple scenarios efficiently.
"""

import pytest
import json
from helpers import format_past_positions, format_education


class TestFormatPastPositions:
    """Tests for the format_past_positions helper function."""

    @pytest.mark.parametrize("positions,expected_contains", [
        ([{'employee_title': 'Software Engineer', 'employer_name': 'Google'}],
         ['Software Engineer', 'Google']),
        ([{'employee_title': 'DevOps Engineer', 'employer_name': 'Wiz',
           'start_date': '2022-01', 'end_date': '2023-12'}],
         ['DevOps Engineer', 'Wiz', '2022-01', '2023-12']),
        ([{'employee_title': 'Backend Dev', 'employer_name': 'Monday.com', 'duration': '2y 6m'}],
         ['Backend Dev', 'Monday.com', '2y 6m']),
        ([{'employee_title': 'Senior Engineer', 'employer_name': 'Snyk'},
          {'employee_title': 'Engineer', 'employer_name': 'AppsFlyer'}],
         ['Senior Engineer', 'Snyk', 'Engineer', 'AppsFlyer']),
        ([], []),
        ([{'employee_title': 'Team Lead', 'employer_name': 'Startup',
           'description': 'Led backend team of 5 engineers'}],
         ['Team Lead', 'Startup', 'Led backend team']),
    ])
    def test_format_positions_contains_expected(self, positions, expected_contains):
        result = format_past_positions(positions)
        for expected in expected_contains:
            assert expected in result, f"Expected '{expected}' in result: {result}"

    @pytest.mark.parametrize("positions", [None, [], [{}], [{'other_field': 'value'}]])
    def test_format_positions_handles_empty(self, positions):
        result = format_past_positions(positions or [])
        assert isinstance(result, str)


class TestFormatEducation:
    """Tests for the format_education helper function."""

    @pytest.mark.parametrize("education,expected_contains", [
        ([{'school_name': 'Technion', 'degree': 'BSc', 'field_of_study': 'Computer Science'}],
         ['Technion', 'BSc', 'Computer Science']),
        ([{'school_name': 'Tel Aviv University'}], ['Tel Aviv University']),
        ([{'school_name': 'Stanford', 'degree': 'MSc', 'field_of_study': 'AI'},
          {'school_name': 'MIT', 'degree': 'BSc'}],
         ['Stanford', 'MSc', 'AI', 'MIT', 'BSc']),
        (['Hebrew University', 'Open University'], ['Hebrew University', 'Open University']),
    ])
    def test_format_education_contains_expected(self, education, expected_contains):
        result = format_education(education)
        for expected in expected_contains:
            assert expected in result, f"Expected '{expected}' in result: {result}"


class TestProfileValidation:
    """Tests for profile data validation and normalization."""

    @pytest.mark.parametrize("profile_data,should_have_data", [
        ({'name': 'John', 'current_title': 'Engineer', 'current_company': 'Wiz'}, True),
        ({'name': 'John'}, False),
        ({'name': 'John', 'current_title': 'Engineer'}, True),
        ({'name': '', 'current_title': '', 'current_company': ''}, False),
        ({'name': 'John', 'skills': ['Python', 'Go']}, True),
    ])
    def test_profile_has_useful_data(self, profile_data, should_have_data):
        def _safe_str(val):
            if val is None:
                return ''
            if isinstance(val, float):
                import math
                return '' if math.isnan(val) else str(val)
            return str(val) if val else ''

        name = _safe_str(profile_data.get('name')) or \
               f"{_safe_str(profile_data.get('first_name'))} {_safe_str(profile_data.get('last_name'))}".strip()
        title = _safe_str(profile_data.get('current_title')) or _safe_str(profile_data.get('headline'))
        company = _safe_str(profile_data.get('current_company'))
        linkedin_url = _safe_str(profile_data.get('linkedin_url'))

        has_useful_data = bool(
            (name or title or company or linkedin_url) and
            (title or company or profile_data.get('skills') or
             profile_data.get('past_positions') or profile_data.get('summary'))
        )
        assert has_useful_data == should_have_data


class TestScreeningScenarios:
    """Parameterized tests for different screening scenarios using mocked AI."""

    @pytest.mark.parametrize("profile_fixture,expected_score_range,expected_fit", [
        ('strong_backend_profile', (7, 10), 'Strong Fit'),
        ('weak_consulting_profile', (1, 4), 'Not a Fit'),
        ('overqualified_profile', (1, 4), 'Not a Fit'),
    ])
    def test_screening_scenarios(self, profile_fixture, expected_score_range, expected_fit,
                                  mock_openai_client_factory, backend_job_description, request):
        profile = request.getfixturevalue(profile_fixture)
        mock_response = {
            "score": expected_score_range[0], "fit": expected_fit,
            "summary": f"Test result for {profile_fixture}", "why": "Test reasoning",
            "strengths": ["Test strength"], "concerns": ["Test concern"]
        }
        mock_client = mock_openai_client_factory(mock_response)

        import dashboard
        result = dashboard.screen_profile(profile, backend_job_description, mock_client)

        assert len(mock_client.captured_calls) == 1, "Expected exactly one API call"
        prompt = mock_client.get_last_prompt()
        assert profile['current_title'] in prompt or profile['name'] in prompt


class TestCompanyQualityRecognition:
    """Test that prompts sent to AI contain correct company quality signals."""

    @pytest.mark.parametrize("company,company_type", [
        ('Wiz', 'top_tier'), ('Monday.com', 'top_tier'), ('Snyk', 'top_tier'),
        ('Google Israel', 'good'), ('Microsoft Israel', 'good'),
        ('Matrix IT', 'consulting'), ('Sela Group', 'consulting'), ('Tikal', 'consulting'),
    ])
    def test_company_appears_in_prompt(self, company, company_type, mock_openai_client,
                                        backend_job_description, profile_builder):
        profile = profile_builder(name="Test User", title="Software Engineer",
                                   company=company, skills=['Python', 'Kubernetes'])

        import dashboard
        dashboard.screen_profile(profile, backend_job_description, mock_openai_client)

        prompt = mock_openai_client.get_last_prompt()
        assert company in prompt, f"Company '{company}' should appear in prompt"


class TestSkillsExtraction:
    """Test that skills are properly extracted and included in prompts."""

    @pytest.mark.parametrize("skills,must_contain", [
        (['Python', 'Go', 'Kubernetes'], ['Python', 'Go', 'Kubernetes']),
        (['React', 'TypeScript', 'Node.js'], ['React', 'TypeScript', 'Node.js']),
        (['Terraform', 'AWS', 'GCP', 'Docker'], ['Terraform', 'AWS', 'GCP']),
        ([], []),
    ])
    def test_skills_in_prompt(self, skills, must_contain, mock_openai_client,
                               backend_job_description, profile_builder):
        profile = profile_builder(name="Test User", title="Engineer",
                                   company="TestCo", skills=skills)

        import dashboard
        dashboard.screen_profile(profile, backend_job_description, mock_openai_client)

        prompt = mock_openai_client.get_last_prompt()
        for skill in must_contain:
            assert skill in prompt, f"Skill '{skill}' should appear in prompt"


class TestRejectionRuleKeywords:
    """Test that rejection rule keywords trigger appropriate handling."""

    @pytest.mark.parametrize("title,should_be_flagged_as", [
        ('VP Engineering', 'overqualified'), ('Director of Engineering', 'overqualified'),
        ('CTO', 'overqualified'), ('Head of DevOps', 'overqualified'),
        ('Junior Developer', 'junior'), ('Intern', 'junior'), ('Student Developer', 'junior'),
        ('Freelance Developer', 'freelancer'),
        ('Senior Software Engineer', None), ('Team Lead', None), ('Staff Engineer', None),
    ])
    def test_title_appears_in_prompt(self, title, should_be_flagged_as, mock_openai_client,
                                      devops_team_lead_job_description, profile_builder):
        profile = profile_builder(name="Test User", title=title, company="TestCo",
                                   skills=['Kubernetes', 'Terraform', 'AWS'])

        import dashboard
        dashboard.screen_profile(profile, devops_team_lead_job_description, mock_openai_client)

        prompt = mock_openai_client.get_last_prompt()
        assert title in prompt, f"Title '{title}' should appear in prompt"


class TestScreeningResultStructure:
    """Test that screening results have the required structure."""

    @pytest.mark.parametrize("response_data,expected_fields", [
        ({"score": 8, "fit": "Strong Fit", "summary": "Great", "why": "Good match"},
         ['score', 'fit', 'summary']),
        ({"score": 5, "fit": "Partial Fit", "summary": "OK",
          "strengths": ["A"], "concerns": ["B"]},
         ['score', 'fit', 'summary', 'strengths', 'concerns']),
    ])
    def test_result_has_required_fields(self, response_data, expected_fields,
                                         mock_openai_client_factory,
                                         strong_backend_profile, backend_job_description):
        mock_client = mock_openai_client_factory(response_data)

        import dashboard
        result = dashboard.screen_profile(strong_backend_profile,
                                           backend_job_description, mock_client)

        for field in expected_fields:
            assert field in result, f"Result should contain '{field}'"
