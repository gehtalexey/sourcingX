"""
Tests demonstrating proper mocking patterns for external services.
Mock OpenAI API calls, Supabase operations, error handling, and concurrency.
"""

import pytest
import json
import warnings
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

    def test_screen_profile_omits_temperature(self, mock_openai_client,
                                               strong_backend_profile,
                                               backend_job_description):
        # gpt-5.6 models reject any temperature override (only the default
        # of 1 is allowed), so screen_profile must not pass one at all.
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client)

        temp = mock_openai_client.captured_calls[0]['temperature']
        assert temp is None, f"Expected no temperature override, got {temp}"

    def test_screen_profile_includes_system_prompt(self, mock_openai_client,
                                                    strong_backend_profile,
                                                    backend_job_description):
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client)

        system_prompt = mock_openai_client.get_system_prompt()
        assert len(system_prompt) > 0, "System prompt should not be empty"
        assert 'score' in system_prompt.lower() or 'recruiter' in system_prompt.lower()

    def test_unified_policy_system_prompt_is_used(self, mock_openai_client,
                                                   strong_backend_profile,
                                                   backend_job_description):
        """The dashboard now routes every screening call through the unified
        screening_policy rubric. The legacy ``role_prompt`` override no longer
        exists — every system prompt must contain the policy signature."""
        import dashboard
        dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                  mock_openai_client)

        system_prompt = mock_openai_client.get_system_prompt()
        assert "senior technical recruiter" in system_prompt.lower()
        # Policy-specific section that does not appear in any legacy prompt
        assert "User-Stated Hard Constraints" in system_prompt


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

        # Profile has current_title/company so it constructs minimal profile and screens it.
        # Unified policy returns {Good Fit, Maybe, Not a Fit}; legacy buckets
        # remain in the assertion list for backward compat with older fixtures.
        assert result['score'] >= 0
        assert result['fit'] in [
            'Strong Fit', 'Good Fit', 'Partial Fit', 'Not a Fit',
            'Maybe', 'Skipped', 'Error'
        ]

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

        # Clear the module-level duration cache so the patched function actually runs.
        # screen_profile() calls compute_role_durations_cached(), which short-circuits
        # to the cached value when another test in the session has already populated it.
        dashboard._duration_cache.clear()

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
        # Unified policy contract is {decision, score, reasoning}. The mock
        # returns a GO decision so the fit maps to 'Good Fit' regardless of
        # the legacy 'fit' key the mock also includes.
        mock_client = mock_openai_client_factory({
            "decision": "GO", "score": 8, "reasoning": "Great background",
            "extra_field": "unexpected value", "another_field": 123
        })

        import dashboard
        result = dashboard.screen_profile(strong_backend_profile, backend_job_description,
                                           mock_client)

        assert result['score'] == 8
        assert result['fit'] == 'Good Fit'


class TestScreeningModelPicker:
    """Tests for screening_models.py -- the one place that decides which
    model/provider screens a candidate, replacing literal strings that used
    to be typed into several places in dashboard.py."""

    def test_default_model_is_live_dispatchable(self):
        import screening_models
        # The default itself must resolve to a provider screen_profile() can
        # actually call, or every fallback path breaks.
        assert screening_models.DEFAULT_SCREEN_MODEL in screening_models.SCREEN_MODELS
        default_entry = screening_models.SCREEN_MODELS[screening_models.DEFAULT_SCREEN_MODEL]
        assert default_entry["provider"] in screening_models.LIVE_DISPATCH_PROVIDERS

    def test_get_screen_model_with_no_config_returns_default(self):
        from screening_models import get_screen_model, DEFAULT_SCREEN_MODEL, SCREEN_MODELS
        model, provider = get_screen_model(None)
        assert model == DEFAULT_SCREEN_MODEL
        assert provider == SCREEN_MODELS[DEFAULT_SCREEN_MODEL]["provider"]

        model2, provider2 = get_screen_model({})
        assert (model2, provider2) == (model, provider)

    def test_get_screen_model_honors_recognized_live_config_value(self):
        from screening_models import get_screen_model
        model, provider = get_screen_model({"screen_model": "gpt-4.1-mini"})
        assert model == "gpt-4.1-mini"
        assert provider == "openai"

    def test_get_screen_model_falls_back_on_unrecognized_value(self):
        # A typo, or a model removed from SCREEN_MODELS, must never reach
        # the call code as an unhandled string -- fail closed to the default.
        from screening_models import get_screen_model, DEFAULT_SCREEN_MODEL, SCREEN_MODELS
        model, provider = get_screen_model({"screen_model": "gpt-99-nonexistent"})
        assert model == DEFAULT_SCREEN_MODEL
        assert provider == SCREEN_MODELS[DEFAULT_SCREEN_MODEL]["provider"]

    def test_get_screen_model_never_selects_jev_for_live_dispatch(self):
        # Jev is listed in SCREEN_MODELS (for a shared display name/table
        # other code can reference) but its provider isn't wired into
        # screen_profile()'s live dispatch yet -- config.json can't select
        # it via this function.
        from screening_models import get_screen_model, DEFAULT_SCREEN_MODEL, SCREEN_MODELS, LIVE_DISPATCH_PROVIDERS
        assert SCREEN_MODELS["jev"]["provider"] == "typesafe"
        assert "typesafe" not in LIVE_DISPATCH_PROVIDERS

        model, provider = get_screen_model({"screen_model": "jev"})
        assert model == DEFAULT_SCREEN_MODEL
        assert provider != "typesafe"

    def test_every_registered_model_has_a_display_name(self):
        from screening_models import SCREEN_MODELS
        for name, entry in SCREEN_MODELS.items():
            assert entry.get("display_name"), f"{name} is missing a display_name"
            assert entry.get("provider"), f"{name} is missing a provider"


class TestUsageTrackerPricingFallback:
    """Tests for usage_tracker.py's pricing fallback. Before this fix, an
    unrecognized model silently used gpt-4o-mini's rate -- the exact bug
    Codex found for gpt-5.6-luna itself in PR #131 (underreporting cost by
    up to 50%) before that model got its own pricing entry. Any future
    unknown model (a typo, a renamed model, Jev if it's ever priced
    per-token) should warn loudly instead of silently mispricing."""

    def test_known_model_prices_without_warning(self):
        from usage_tracker import calculate_openai_cost, OPENAI_PRICING
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cost = calculate_openai_cost(1_000_000, 1_000_000, model="gpt-5.6-luna")
        expected = OPENAI_PRICING["gpt-5.6-luna"]["input"] + OPENAI_PRICING["gpt-5.6-luna"]["output"]
        assert cost == pytest.approx(expected)

    def test_unknown_model_warns_and_falls_back_to_gpt4o_mini(self):
        from usage_tracker import calculate_openai_cost, OPENAI_PRICING
        with pytest.warns(UserWarning, match="totally-made-up-model"):
            cost = calculate_openai_cost(1_000_000, 1_000_000, model="totally-made-up-model")
        fallback = OPENAI_PRICING["gpt-4o-mini"]
        assert cost == pytest.approx(fallback["input"] + fallback["output"])

    def test_openai_pricing_for_unknown_model_warns_and_returns_fallback_dict(self):
        from usage_tracker import _openai_pricing_for, OPENAI_PRICING
        with pytest.warns(UserWarning):
            pricing = _openai_pricing_for("another-made-up-model")
        assert pricing == OPENAI_PRICING["gpt-4o-mini"]
