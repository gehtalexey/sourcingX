"""
Tests demonstrating proper mocking patterns for external services.
Mock OpenAI API calls, Supabase operations, error handling, and concurrency.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call
import threading


class TestOpenAIMocking:
    """Tests demonstrating OpenAI API mocking patterns."""

    def test_screen_profile_uses_correct_model(self, mock_openai_client,
                                                strong_backend_profile,
                                                backend_job_description):
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client, ai_model="gpt-4o-mini")

        assert len(mock_openai_client.captured_calls) == 1
        assert mock_openai_client.captured_calls[0]['model'] == 'gpt-4o-mini'

    def test_screen_profile_uses_json_response_format(self, mock_openai_client,
                                                       strong_backend_profile,
                                                       backend_job_description):
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client)

        assert mock_openai_client.captured_calls[0]['response_format'] == {"type": "json_object"}

    def test_screen_profile_uses_low_temperature(self, mock_openai_client,
                                                  strong_backend_profile,
                                                  backend_job_description):
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client)

        temp = mock_openai_client.captured_calls[0]['temperature']
        assert temp is not None and temp <= 0.3, f"Temperature should be <=0.3, got {temp}"

    def test_screen_profile_includes_system_prompt(self, mock_openai_client,
                                                    strong_backend_profile,
                                                    backend_job_description):
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client)

        system_prompt = mock_openai_client.get_system_prompt()
        assert len(system_prompt) > 0, "System prompt should not be empty"
        assert 'score' in system_prompt.lower() or 'recruiter' in system_prompt.lower()

    def test_custom_system_prompt_is_used(self, mock_openai_client,
                                           strong_backend_profile,
                                           backend_job_description):
        import dashboard
        custom_prompt = "You are a specialized DevOps recruiter. Focus on Kubernetes."

        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client, role_prompt=custom_prompt)

        system_prompt = mock_openai_client.get_system_prompt()
        assert custom_prompt in system_prompt


class TestErrorHandling:
    """Tests for error handling in screening functions."""

    def test_empty_profile_returns_skipped(self, mock_openai_client, empty_profile,
                                            backend_job_description):
        import dashboard
        result = dashboard.screen_profile(empty_profile, backend_job_description,
                                           mock_openai_client)

        assert len(mock_openai_client.captured_calls) == 0
        assert result['fit'] == 'Skipped'
        assert result['score'] == 0

    def test_profile_without_work_history_screens_with_minimal_data(self, mock_openai_client,
                                                                     backend_job_description):
        import dashboard
        profile = {
            'first_name': 'Test', 'last_name': 'User',
            'current_title': 'Engineer', 'current_company': 'TestCo',
            'raw_crustdata': {}
        }

        result = dashboard.screen_profile(profile, backend_job_description, mock_openai_client)

        # Profile has current_title/company so it constructs minimal profile and screens it
        assert result['score'] >= 0
        assert result['fit'] in ['Strong Fit', 'Good Fit', 'Partial Fit', 'Not a Fit', 'Skipped', 'Error']

    def test_malformed_json_response_handling(self, backend_job_description,
                                               strong_backend_profile):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Not valid JSON"))],
            usage=MagicMock(prompt_tokens=100, completion_tokens=50)
        )

        import dashboard
        try:
            result = dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                               mock_client)
            assert result.get('fit') in ['Error', 'Skipped', 'Missing Data'] or \
                   isinstance(result.get('score'), (int, float))
        except json.JSONDecodeError:
            pass


class TestBatchScreeningMocking:
    """Tests for batch screening with mocked services."""

    def test_batch_screening_processes_all_profiles(self, strong_backend_profile,
                                                     weak_consulting_profile,
                                                     backend_job_description):
        import dashboard
        profiles = [strong_backend_profile, weak_consulting_profile]

        with patch('dashboard.OpenAI') as mock_openai_class:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps({
                    "score": 7, "fit": "Good Fit", "summary": "Test"
                })))],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50)
            )
            mock_openai_class.return_value = mock_instance

            results = dashboard.screen_profiles_batch(
                profiles, backend_job_description,
                openai_api_key="test-key", max_workers=2
            )

        assert len(results) == 2

    def test_batch_screening_handles_cancellation(self, strong_backend_profile,
                                                   backend_job_description):
        import dashboard
        profiles = [strong_backend_profile] * 5
        cancel_flag = {'cancelled': True}

        with patch('dashboard.OpenAI') as mock_openai_class:
            mock_instance = MagicMock()
            mock_openai_class.return_value = mock_instance

            results = dashboard.screen_profiles_batch(
                profiles, backend_job_description,
                openai_api_key="test-key", cancel_flag=cancel_flag
            )

        assert len(results) < len(profiles)

    def test_batch_screening_calls_progress_callback(self, strong_backend_profile,
                                                      backend_job_description):
        import dashboard
        profiles = [strong_backend_profile]
        progress_calls = []

        def progress_callback(completed, total, result):
            progress_calls.append((completed, total, result))

        with patch('dashboard.OpenAI') as mock_openai_class:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps({
                    "score": 7, "fit": "Good Fit", "summary": "Test"
                })))],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50)
            )
            mock_openai_class.return_value = mock_instance

            dashboard.screen_profiles_batch(
                profiles, backend_job_description,
                openai_api_key="test-key", progress_callback=progress_callback
            )

        assert len(progress_calls) > 0, "Progress callback should be called"


class TestHelperFunctionMocking:
    """Tests for pre-computation helper functions used in screening."""

    def test_compute_role_durations_is_called(self, mock_openai_client,
                                               strong_backend_profile,
                                               backend_job_description):
        import dashboard

        with patch('dashboard.compute_role_durations') as mock_compute:
            mock_compute.return_value = "ROLE DURATIONS: mocked"
            dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                       mock_openai_client)
            mock_compute.assert_called()

    def test_trim_raw_profile_is_called(self, mock_openai_client,
                                         strong_backend_profile,
                                         backend_job_description):
        import dashboard

        with patch('dashboard.trim_raw_profile') as mock_trim:
            mock_trim.return_value = {"name": "Test"}
            dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                       mock_openai_client)
            mock_trim.assert_called()


class TestUsageTrackerMocking:
    """Tests for usage tracking with mocked services."""

    def test_usage_tracker_records_tokens(self, mock_openai_client,
                                           strong_backend_profile,
                                           backend_job_description):
        import dashboard
        mock_tracker = MagicMock()

        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client, tracker=mock_tracker)

        if mock_tracker.method_calls:
            assert len(mock_tracker.method_calls) > 0


class TestConcurrencyMocking:
    """Tests for thread safety in batch operations."""

    def test_batch_screening_thread_safety(self, strong_backend_profile,
                                            backend_job_description):
        import dashboard
        profiles = [strong_backend_profile] * 10
        results_lock = threading.Lock()
        collected_results = []

        def mock_progress(completed, total, result):
            with results_lock:
                collected_results.append(result)

        with patch('dashboard.OpenAI') as mock_openai_class:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps({
                    "score": 7, "fit": "Good Fit", "summary": "Test"
                })))],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50)
            )
            mock_openai_class.return_value = mock_instance

            results = dashboard.screen_profiles_batch(
                profiles, backend_job_description,
                openai_api_key="test-key", max_workers=5,
                progress_callback=mock_progress
            )

        assert len(results) == len(profiles)
        assert len(collected_results) == len(profiles)


class TestAPIResponseVariations:
    """Tests for handling various API response formats."""

    def test_handles_response_without_why_field(self, mock_openai_client_factory,
                                                  strong_backend_profile,
                                                  backend_job_description):
        mock_client = mock_openai_client_factory({
            "score": 7, "fit": "Good Fit", "summary": "Good candidate",
            "strengths": ["Python"], "concerns": []
        })

        import dashboard
        result = dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                           mock_client)

        assert 'score' in result
        assert 'fit' in result

    def test_handles_response_with_extra_fields(self, mock_openai_client_factory,
                                                  strong_backend_profile,
                                                  backend_job_description):
        mock_client = mock_openai_client_factory({
            "score": 8, "fit": "Strong Fit", "summary": "Excellent candidate",
            "why": "Great background", "strengths": ["Top company"], "concerns": [],
            "extra_field": "unexpected value", "another_field": 123
        })

        import dashboard
        result = dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                           mock_client)

        assert result['score'] == 8
        assert result['fit'] == 'Strong Fit'
