"""
Pytest configuration and shared fixtures for SourcingX tests.

This module provides:
- Mock clients for OpenAI API calls
- Sample profile data fixtures
- Job description fixtures
- Helper utilities for testing screening logic
"""

import json
import pytest
from unittest.mock import MagicMock, patch


class MockOpenAIResponse:
    """Mock response object matching OpenAI's response structure."""

    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content
        self.usage = MagicMock()
        self.usage.prompt_tokens = prompt_tokens
        self.usage.completion_tokens = completion_tokens


class MockOpenAIClient:
    """Mock OpenAI client for testing without API calls."""

    def __init__(self, response_data: dict = None):
        self.captured_calls = []
        self.response_data = response_data or {
            "score": 7, "fit": "Good Fit",
            "summary": "Strong candidate with relevant experience",
            "why": "Meets core requirements",
            "strengths": ["Good company background", "Relevant skills"],
            "concerns": ["Could use more leadership experience"]
        }
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create_completion

    def _create_completion(self, **kwargs):
        self.captured_calls.append({
            'model': kwargs.get('model'),
            'messages': kwargs.get('messages'),
            'response_format': kwargs.get('response_format'),
            'temperature': kwargs.get('temperature'),
            'max_tokens': kwargs.get('max_tokens')
        })
        return MockOpenAIResponse(json.dumps(self.response_data))

    def get_last_prompt(self) -> str:
        if self.captured_calls:
            messages = self.captured_calls[-1].get('messages', [])
            for msg in messages:
                if msg.get('role') == 'user':
                    return msg.get('content', '')
        return ''

    def get_system_prompt(self) -> str:
        if self.captured_calls:
            messages = self.captured_calls[-1].get('messages', [])
            for msg in messages:
                if msg.get('role') == 'system':
                    return msg.get('content', '')
        return ''


@pytest.fixture
def mock_openai_client():
    return MockOpenAIClient()


@pytest.fixture
def mock_openai_client_factory():
    def _create(response_data: dict):
        return MockOpenAIClient(response_data)
    return _create


@pytest.fixture
def strong_backend_profile():
    return {
        'first_name': 'David', 'last_name': 'Cohen', 'name': 'David Cohen',
        'current_title': 'Senior Backend Engineer', 'current_company': 'Wiz',
        'headline': 'Senior Backend Engineer at Wiz', 'location': 'Tel Aviv, Israel',
        'summary': 'Experienced backend engineer specializing in distributed systems.',
        'skills': ['Python', 'Go', 'Kubernetes', 'AWS', 'PostgreSQL', 'Redis', 'Docker'],
        'education': 'Technion', 'all_employers': ['Wiz', 'Monday.com', 'Google Israel'],
        'all_titles': ['Senior Backend Engineer', 'Backend Engineer', 'Software Engineer'],
        'raw_crustdata': {
            'past_employers': [
                {'employee_title': 'Backend Engineer', 'employer_name': 'Monday.com',
                 'start_date': '2019-06', 'end_date': '2022-03', 'duration': '2y 9m'},
                {'employee_title': 'Software Engineer', 'employer_name': 'Google Israel',
                 'start_date': '2016-09', 'end_date': '2019-05', 'duration': '2y 8m'}
            ],
            'current_employers': [
                {'employee_title': 'Senior Backend Engineer', 'employer_name': 'Wiz',
                 'start_date': '2022-04'}
            ],
            'education_background': [
                {'school_name': 'Technion', 'degree': 'BSc', 'field_of_study': 'Computer Science'}
            ],
            'all_employers': ['Wiz', 'Monday.com', 'Google Israel'],
            'all_titles': ['Senior Backend Engineer', 'Backend Engineer', 'Software Engineer']
        }
    }


@pytest.fixture
def weak_consulting_profile():
    return {
        'first_name': 'John', 'last_name': 'Doe', 'name': 'John Doe',
        'current_title': 'Software Developer', 'current_company': 'Matrix IT',
        'headline': 'Software Developer at Matrix IT', 'location': 'Tel Aviv, Israel',
        'summary': 'Developer working on various client projects.',
        'skills': ['Java', '.NET', 'SQL Server'], 'education': 'College of Management',
        'all_employers': ['Matrix IT', 'Sela Group', 'Ness Technologies'],
        'all_titles': ['Software Developer', 'Junior Developer'],
        'raw_crustdata': {
            'past_employers': [
                {'employee_title': 'Junior Developer', 'employer_name': 'Sela Group',
                 'start_date': '2020-01', 'end_date': '2021-06', 'duration': '1y 5m'},
                {'employee_title': 'Junior Developer', 'employer_name': 'Ness Technologies',
                 'start_date': '2018-06', 'end_date': '2019-12', 'duration': '1y 6m'}
            ],
            'current_employers': [
                {'employee_title': 'Software Developer', 'employer_name': 'Matrix IT',
                 'start_date': '2021-07'}
            ],
            'education_background': [{'school_name': 'College of Management'}],
            'all_employers': ['Matrix IT', 'Sela Group', 'Ness Technologies'],
            'all_titles': ['Software Developer', 'Junior Developer']
        }
    }


@pytest.fixture
def overqualified_profile():
    return {
        'first_name': 'Sarah', 'last_name': 'Levi', 'name': 'Sarah Levi',
        'current_title': 'VP Engineering', 'current_company': 'TechCorp',
        'headline': 'VP Engineering at TechCorp', 'location': 'Tel Aviv, Israel',
        'summary': 'Engineering leader with 15+ years of experience.',
        'skills': ['Leadership', 'Strategy', 'Architecture', 'Team Building'],
        'education': 'Tel Aviv University',
        'all_employers': ['TechCorp', 'BigCompany', 'StartupX'],
        'all_titles': ['VP Engineering', 'Director of Engineering', 'Engineering Manager'],
        'raw_crustdata': {
            'past_employers': [
                {'employee_title': 'Director of Engineering', 'employer_name': 'BigCompany',
                 'start_date': '2015-01', 'end_date': '2020-12', 'duration': '5y 11m'}
            ],
            'current_employers': [
                {'employee_title': 'VP Engineering', 'employer_name': 'TechCorp',
                 'start_date': '2021-01'}
            ],
            'education_background': [
                {'school_name': 'Tel Aviv University', 'degree': 'BSc',
                 'field_of_study': 'Computer Science'}
            ],
            'all_employers': ['TechCorp', 'BigCompany', 'StartupX'],
            'all_titles': ['VP Engineering', 'Director of Engineering', 'Engineering Manager']
        }
    }


@pytest.fixture
def empty_profile():
    return {
        'first_name': '', 'last_name': '', 'name': '',
        'current_title': '', 'current_company': '', 'headline': '',
        'skills': [], 'summary': ''
    }


@pytest.fixture
def devops_team_lead_profile():
    return {
        'first_name': 'Alex', 'last_name': 'Stern', 'name': 'Alex Stern',
        'current_title': 'DevOps Team Lead', 'current_company': 'Snyk',
        'headline': 'DevOps Team Lead at Snyk', 'location': 'Tel Aviv, Israel',
        'summary': 'Leading DevOps team with focus on Kubernetes.',
        'skills': ['Kubernetes', 'Terraform', 'AWS', 'GCP', 'Docker', 'CI/CD', 'Python'],
        'education': 'Hebrew University', 'all_employers': ['Snyk', 'AppsFlyer', 'Check Point'],
        'all_titles': ['DevOps Team Lead', 'Senior DevOps Engineer', 'DevOps Engineer'],
        'raw_crustdata': {
            'past_employers': [
                {'employee_title': 'Senior DevOps Engineer', 'employer_name': 'AppsFlyer',
                 'start_date': '2019-03', 'end_date': '2022-06', 'duration': '3y 3m'},
                {'employee_title': 'DevOps Engineer', 'employer_name': 'Check Point',
                 'start_date': '2016-01', 'end_date': '2019-02', 'duration': '3y 1m'}
            ],
            'current_employers': [
                {'employee_title': 'DevOps Team Lead', 'employer_name': 'Snyk',
                 'start_date': '2022-07'}
            ],
            'education_background': [
                {'school_name': 'Hebrew University', 'degree': 'BSc',
                 'field_of_study': 'Computer Science'}
            ],
            'all_employers': ['Snyk', 'AppsFlyer', 'Check Point'],
            'all_titles': ['DevOps Team Lead', 'Senior DevOps Engineer', 'DevOps Engineer']
        }
    }


@pytest.fixture
def backend_job_description():
    return """
    Senior Backend Engineer - Tel Aviv, Israel
    Requirements:
    - 5+ years of backend development experience
    - Strong experience with Python, Go, or Node.js
    - Experience with cloud platforms (AWS, GCP, or Azure)
    - Experience with distributed systems and microservices
    """


@pytest.fixture
def devops_team_lead_job_description():
    return """
    DevOps Team Lead - Tel Aviv, Israel
    Requirements:
    - 6+ years in DevOps, including 2+ years in a leadership role
    - Deep expertise in Kubernetes and cloud-native infrastructure
    - Expertise in Infrastructure as Code (IaC), particularly Terraform
    Reject junior, students, freelancers.
    Reject overqualified like VP, Director, CTO, Head of etc.
    Kubernetes is a must.
    """


@pytest.fixture
def fullstack_job_description():
    return """
    Full Stack Engineer - Tel Aviv
    Requirements:
    - 4+ years of full stack development
    - React/TypeScript frontend experience
    - Node.js or Python backend experience
    """


@pytest.fixture
def mock_supabase_client():
    client = MagicMock()
    table_mock = MagicMock()
    client.table.return_value = table_mock
    select_mock = MagicMock()
    table_mock.select.return_value = select_mock
    select_mock.eq.return_value = select_mock
    select_mock.execute.return_value = MagicMock(data=[])
    table_mock.insert.return_value = MagicMock()
    table_mock.update.return_value = MagicMock()
    table_mock.upsert.return_value = MagicMock()
    return client


@pytest.fixture
def screening_result_validator():
    def _validate(result: dict):
        required_fields = ['score', 'fit', 'summary']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        assert isinstance(result['score'], (int, float)), "Score must be numeric"
        assert 0 <= result['score'] <= 10, "Score must be between 0 and 10"
        valid_fits = ['Strong Fit', 'Good Fit', 'Partial Fit', 'Not a Fit',
                      'Skipped', 'Error', 'Missing Data']
        assert result['fit'] in valid_fits, f"Invalid fit level: {result['fit']}"
        return True
    return _validate


@pytest.fixture
def profile_builder():
    def _build(name="Test User", title="Software Engineer", company="TestCo",
               skills=None, employers=None, include_raw=True):
        profile = {
            'first_name': name.split()[0] if name else '',
            'last_name': name.split()[-1] if name and len(name.split()) > 1 else '',
            'name': name, 'current_title': title, 'current_company': company,
            'headline': f"{title} at {company}", 'location': 'Tel Aviv, Israel',
            'skills': skills or ['Python', 'JavaScript'],
            'all_employers': employers or [company], 'all_titles': [title]
        }
        if include_raw:
            profile['raw_crustdata'] = {
                'current_employers': [
                    {'employee_title': title, 'employer_name': company, 'start_date': '2022-01'}
                ],
                'past_employers': [], 'education_background': [],
                'all_employers': employers or [company], 'all_titles': [title]
            }
        return profile
    return _build
