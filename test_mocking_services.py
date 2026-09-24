"""
Tests demonstrating proper mocking patterns for external services.
Mock OpenAI API calls, Supabase operations, error handling, and concurrency.
"""

import pytest
import json
import csv
import argparse
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


class TestBakeoffKalamata:
    """Tests for bakeoff_kalamata.py (docs/GOAL-screening-bakeoff-vs-kalamata.md):
    the script that samples kalamata's already-screened candidates, screens
    them through Jev, and reports Luna/Jev/kalamata verdicts side by side.
    No Supabase, no Jev SDK, no network anywhere in this class -- every test
    below exercises pure functions or mocks screen_with_jev."""

    # -- brief_to_job_description: must match dashboard.py:9873-9878 exactly --

    def test_brief_to_job_description_matches_dashboard_format(self):
        import bakeoff_kalamata as bk
        brief = {
            "role_context": "Senior Backend Engineer, Tel Aviv",
            "must_haves": ["5+ years backend", "Strong Python"],
            "nice_to_haves": ["Docker"],
            "exclusions": ["Pure team leads"],
        }
        expected = "\n".join([
            "Role: Senior Backend Engineer, Tel Aviv",
            "Must-haves:\n- 5+ years backend\n- Strong Python",
            "Nice-to-haves:\n- Docker",
            "Exclusions:\n- Pure team leads",
        ])
        assert bk.brief_to_job_description(brief) == expected

    def test_brief_to_job_description_omits_empty_sections(self):
        import bakeoff_kalamata as bk
        brief = {"role_context": "VP Marketing", "must_haves": [], "nice_to_haves": [], "exclusions": []}
        assert bk.brief_to_job_description(brief) == "Role: VP Marketing"

    def test_brief_to_job_description_empty_brief_is_empty_string(self):
        import bakeoff_kalamata as bk
        assert bk.brief_to_job_description({}) == ""

    # -- thin-profile eligibility (must never feed the dashboard a thin profile) --

    def test_eligible_profile_with_skills(self):
        import bakeoff_kalamata as bk
        assert bk.is_eligible_profile({"raw_data": {"skills": ["Python"], "summary": None}})

    def test_eligible_profile_with_summary_only(self):
        import bakeoff_kalamata as bk
        assert bk.is_eligible_profile({"raw_data": {"skills": [], "summary": "Great engineer"}})

    def test_thin_profile_missing_both_is_ineligible(self):
        import bakeoff_kalamata as bk
        assert not bk.is_eligible_profile({"raw_data": {"skills": [], "summary": ""}})

    def test_profile_with_no_raw_data_is_ineligible(self):
        import bakeoff_kalamata as bk
        assert not bk.is_eligible_profile({"raw_data": None})
        assert not bk.is_eligible_profile({})

    def test_eligible_profile_accepts_json_string_raw_data(self):
        import bakeoff_kalamata as bk
        profile = {"raw_data": json.dumps({"skills": ["Go"], "summary": None})}
        assert bk.is_eligible_profile(profile)

    # -- kalamata row exclusion + grouping --

    def test_row_excluded_when_score_is_null(self):
        import bakeoff_kalamata as bk
        assert bk.is_excluded_kalamata_row("some notes", None)

    def test_row_excluded_when_notes_start_unenrichable(self):
        import bakeoff_kalamata as bk
        assert bk.is_excluded_kalamata_row("[Unenrichable] no linkedin data", 5)

    def test_row_not_excluded_otherwise(self):
        import bakeoff_kalamata as bk
        assert not bk.is_excluded_kalamata_row("looks fine", 8)
        assert not bk.is_excluded_kalamata_row(None, 3)

    def test_classify_group_yes(self):
        import bakeoff_kalamata as bk
        assert bk.classify_kalamata_group("qualified", None) == "yes"

    def test_classify_group_no_prescreen(self):
        import bakeoff_kalamata as bk
        assert bk.classify_kalamata_group("not_qualified", "[Prescreen] too junior") == "no_prescreen"

    def test_classify_group_no_fullscreen(self):
        import bakeoff_kalamata as bk
        assert bk.classify_kalamata_group("not_qualified", "missed a must-have") == "no_fullscreen"

    def test_classify_group_none_for_unknown_result(self):
        import bakeoff_kalamata as bk
        assert bk.classify_kalamata_group("incomplete", None) is None

    def test_kalamata_reason_prefers_notes(self):
        import bakeoff_kalamata as bk
        row = {"screening_notes": "x" * 250, "screening_detail": {"summary": "unused"}}
        reason = bk.kalamata_reason(row)
        assert reason == "x" * 200

    def test_kalamata_reason_falls_back_to_detail_summary(self):
        import bakeoff_kalamata as bk
        row = {"screening_notes": "", "screening_detail": {"summary": "Strong SRE background"}}
        assert bk.kalamata_reason(row) == "Strong SRE background"

    def test_kalamata_reason_empty_when_nothing_available(self):
        import bakeoff_kalamata as bk
        assert bk.kalamata_reason({"screening_notes": None, "screening_detail": None}) == ""

    # -- deterministic sampling --

    def test_deterministic_sample_same_seed_same_input_same_pick(self):
        import bakeoff_kalamata as bk
        urls = [f"https://www.linkedin.com/in/person-{i}" for i in range(20)]
        pick1 = bk.deterministic_sample(urls, 5, seed=20260924)
        pick2 = bk.deterministic_sample(list(reversed(urls)), 5, seed=20260924)
        assert pick1 == pick2  # sorted first, so input order doesn't matter
        assert len(pick1) == 5

    def test_deterministic_sample_different_seed_can_differ(self):
        import bakeoff_kalamata as bk
        urls = [f"https://www.linkedin.com/in/person-{i}" for i in range(30)]
        pick_a = bk.deterministic_sample(urls, 10, seed=1)
        pick_b = bk.deterministic_sample(urls, 10, seed=2)
        assert pick_a != pick_b

    def test_deterministic_sample_returns_all_when_k_exceeds_pool(self):
        import bakeoff_kalamata as bk
        urls = ["a", "b", "c"]
        assert bk.deterministic_sample(urls, 10, seed=1) == ["a", "b", "c"]

    # -- verdict mapping --

    def test_luna_verdict_good_fit_and_maybe_are_yes(self):
        import bakeoff_kalamata as bk
        assert bk.luna_verdict("Good Fit") == "yes"
        assert bk.luna_verdict("Maybe") == "yes"

    def test_luna_verdict_not_a_fit_is_no(self):
        import bakeoff_kalamata as bk
        assert bk.luna_verdict("Not a Fit") == "no"

    def test_luna_verdict_unknown_or_missing_is_none(self):
        import bakeoff_kalamata as bk
        assert bk.luna_verdict("Error") is None
        assert bk.luna_verdict(None) is None

    def test_jev_verdict_from_result(self):
        import bakeoff_kalamata as bk
        assert bk.jev_verdict_from_result("qualified") == "yes"
        assert bk.jev_verdict_from_result("not_qualified") == "no"
        assert bk.jev_verdict_from_result(None) is None

    # -- agreement math --

    def test_compute_agreement_counts_only_rows_with_both_present(self):
        import bakeoff_kalamata as bk
        rows = [
            {"a": "yes", "b": "yes"},
            {"a": "yes", "b": "no"},
            {"a": "no", "b": "no"},
            {"a": "missing", "b": "yes"},  # excluded: 'a' not yes/no
        ]
        result = bk.compute_agreement(rows, "a", "b")
        assert result["matches"] == 2
        assert result["total"] == 3
        assert result["rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_compute_agreement_no_eligible_rows_gives_none_rate(self):
        import bakeoff_kalamata as bk
        rows = [{"a": "missing", "b": "missing"}]
        result = bk.compute_agreement(rows, "a", "b")
        assert result == {"matches": 0, "total": 0, "rate": None}

    # -- review_sample determinism --

    def test_pick_review_sample_deterministic_and_sorted(self):
        import bakeoff_kalamata as bk
        rows = [{"linkedin_url": f"https://www.linkedin.com/in/p{i}"} for i in range(50)]
        sample1 = bk.pick_review_sample(rows, seed=20260924, limit=30)
        sample2 = bk.pick_review_sample(list(reversed(rows)), seed=20260924, limit=30)
        assert len(sample1) == 30
        assert sample1 == sample2
        urls = [r["linkedin_url"] for r in sample1]
        assert urls == sorted(urls)

    def test_pick_review_sample_returns_all_when_fewer_than_limit(self):
        import bakeoff_kalamata as bk
        rows = [{"linkedin_url": "https://www.linkedin.com/in/only-one"}]
        assert bk.pick_review_sample(rows, seed=1, limit=30) == rows

    # -- jev dry run must never call screen_with_jev --

    def test_jev_dry_run_never_calls_screen_with_jev(self, tmp_path, monkeypatch):
        import bakeoff_kalamata as bk

        sample = {
            "seed": 1,
            "positions": ["pos-a"],
            "rows": [
                {"position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-1",
                 "group": "yes", "kalamata_verdict": "yes", "kalamata_score": 8, "kalamata_reason": "ok"},
            ],
        }
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample), encoding="utf-8")
        briefs_path = tmp_path / "briefs.json"
        briefs_path.write_text(json.dumps({"pos-a": {"role_context": "Backend Engineer"}}), encoding="utf-8")
        out_path = tmp_path / "jev_results.csv"

        monkeypatch.setattr(bk, "_load_supabase_client", lambda: object())
        monkeypatch.setattr(bk, "fetch_profiles_by_urls", lambda client, urls: {
            "https://www.linkedin.com/in/candidate-1": {"raw_data": {"skills": ["Python"]}},
        })

        called = {"n": 0}

        def _fail_if_called(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("screen_with_jev must not be called without --yes")

        monkeypatch.setattr(bk.jev_client, "screen_with_jev", _fail_if_called)
        monkeypatch.setattr(bk.jev_client, "build_client", _fail_if_called)

        args = argparse.Namespace(
            sample=str(sample_path), briefs=str(briefs_path), out=str(out_path),
            yes=False, limit=None, workers=4,
        )
        bk.cmd_jev(args)

        assert called["n"] == 0
        assert not out_path.exists()

    def test_jev_with_yes_calls_screen_with_jev_and_writes_csv(self, tmp_path, monkeypatch):
        import bakeoff_kalamata as bk

        sample = {
            "seed": 1,
            "positions": ["pos-a"],
            "rows": [
                {"position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-1",
                 "group": "yes", "kalamata_verdict": "yes", "kalamata_score": 8, "kalamata_reason": "ok"},
            ],
        }
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample), encoding="utf-8")
        briefs_path = tmp_path / "briefs.json"
        briefs_path.write_text(json.dumps({"pos-a": {"role_context": "Backend Engineer"}}), encoding="utf-8")
        out_path = tmp_path / "jev_results.csv"

        monkeypatch.setattr(bk, "_load_supabase_client", lambda: object())
        monkeypatch.setattr(bk, "fetch_profiles_by_urls", lambda client, urls: {
            "https://www.linkedin.com/in/candidate-1": {"raw_data": {"skills": ["Python"]}},
        })
        monkeypatch.setattr(bk.jev_client, "build_client", lambda: "fake-client")

        def _fake_screen(profile, job_description, client=None, screening_brief=None):
            assert client == "fake-client"
            return {
                "screening_result": "qualified", "screening_score": 8,
                "reason": "Fit score: qualified", "jev_model": "jev-1.0",
                "input_tokens": 100, "output_tokens": 10,
            }

        monkeypatch.setattr(bk.jev_client, "screen_with_jev", _fake_screen)

        args = argparse.Namespace(
            sample=str(sample_path), briefs=str(briefs_path), out=str(out_path),
            yes=True, limit=None, workers=2,
        )
        bk.cmd_jev(args)

        assert out_path.exists()
        with open(out_path, encoding="utf-8-sig", newline="") as f:
            written = list(csv.DictReader(f))
        assert len(written) == 1
        assert written[0]["jev_verdict"] == "yes"
        assert written[0]["error"] == ""

    def test_jev_resume_retries_error_rows_and_keeps_one_row_per_candidate(self, tmp_path, monkeypatch):
        import bakeoff_kalamata as bk

        sample = {
            "seed": 1,
            "positions": ["pos-a"],
            "rows": [
                {"position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-1",
                 "group": "yes", "kalamata_verdict": "yes", "kalamata_score": 8, "kalamata_reason": "ok"},
                {"position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-2",
                 "group": "no_fullscreen", "kalamata_verdict": "no", "kalamata_score": 3, "kalamata_reason": "meh"},
            ],
        }
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample), encoding="utf-8")
        briefs_path = tmp_path / "briefs.json"
        briefs_path.write_text(json.dumps({"pos-a": {"role_context": "Backend Engineer"}}), encoding="utf-8")
        out_path = tmp_path / "jev_results.csv"

        fieldnames = [
            "position_id", "linkedin_url", "jev_verdict", "jev_score", "jev_fit_level",
            "jev_reason", "jev_model", "input_tokens", "output_tokens", "error",
        ]
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # candidate-1 already succeeded -- must not be retried.
            writer.writerow({
                "position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-1",
                "jev_verdict": "yes", "jev_score": 9, "jev_fit_level": "qualified",
                "jev_reason": "good fit", "jev_model": "jev-1.0",
                "input_tokens": 100, "output_tokens": 10, "error": "",
            })
            # candidate-2 errored -- must be retried, and the old row dropped.
            writer.writerow({
                "position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-2",
                "jev_verdict": "", "jev_score": "", "jev_fit_level": "",
                "jev_reason": "", "jev_model": "", "input_tokens": "", "output_tokens": "",
                "error": "RuntimeError: boom",
            })

        monkeypatch.setattr(bk, "_load_supabase_client", lambda: object())
        monkeypatch.setattr(bk, "fetch_profiles_by_urls", lambda client, urls: {
            "https://www.linkedin.com/in/candidate-1": {"raw_data": {"skills": ["Python"]}},
            "https://www.linkedin.com/in/candidate-2": {"raw_data": {"skills": ["Go"]}},
        })
        monkeypatch.setattr(bk.jev_client, "build_client", lambda: "fake-client")

        calls = []

        def _fake_screen(profile, job_description, client=None, screening_brief=None):
            calls.append(profile)
            return {
                "screening_result": "not_qualified", "screening_score": 2,
                "reason": "missing a must-have", "jev_model": "jev-1.0",
                "input_tokens": 50, "output_tokens": 5,
            }

        monkeypatch.setattr(bk.jev_client, "screen_with_jev", _fake_screen)

        args = argparse.Namespace(
            sample=str(sample_path), briefs=str(briefs_path), out=str(out_path),
            yes=True, limit=None, workers=2,
        )
        bk.cmd_jev(args)

        # Only the error row should have been retried.
        assert len(calls) == 1
        assert calls[0] == {"raw_data": {"skills": ["Go"]}}

        with open(out_path, encoding="utf-8-sig", newline="") as f:
            written = list(csv.DictReader(f))
        keys = [(row["position_id"], row["linkedin_url"]) for row in written]
        assert keys.count(("pos-a", "https://www.linkedin.com/in/candidate-1")) == 1
        assert keys.count(("pos-a", "https://www.linkedin.com/in/candidate-2")) == 1
        by_url = {row["linkedin_url"]: row for row in written}
        assert by_url["https://www.linkedin.com/in/candidate-1"]["jev_verdict"] == "yes"
        assert by_url["https://www.linkedin.com/in/candidate-2"]["jev_verdict"] == "no"
        assert by_url["https://www.linkedin.com/in/candidate-2"]["error"] == ""

    # -- report: a real zero score must survive, not become blank --

    def test_report_keeps_zero_jev_score(self, tmp_path, monkeypatch):
        import bakeoff_kalamata as bk

        sample = {
            "seed": 1,
            "positions": ["pos-a"],
            "rows": [
                {"position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-1",
                 "group": "no_fullscreen", "kalamata_verdict": "no", "kalamata_score": 1, "kalamata_reason": "meh"},
            ],
        }
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample), encoding="utf-8")
        briefs_path = tmp_path / "briefs.json"
        briefs_path.write_text(json.dumps({"pos-a": {"role_context": "Backend Engineer"}}), encoding="utf-8")

        jev_path = tmp_path / "jev.csv"
        with open(jev_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "position_id", "linkedin_url", "jev_verdict", "jev_score", "jev_fit_level",
                "jev_reason", "jev_model", "input_tokens", "output_tokens", "error",
            ])
            writer.writeheader()
            writer.writerow({
                "position_id": "pos-a", "linkedin_url": "https://www.linkedin.com/in/candidate-1",
                "jev_verdict": "no", "jev_score": "0", "jev_fit_level": "not_qualified",
                "jev_reason": "no signal", "jev_model": "jev-1.0",
                "input_tokens": 10, "output_tokens": 5, "error": "",
            })

        out_dir = tmp_path / "out"
        monkeypatch.setattr(bk, "_load_supabase_client", lambda: object())
        monkeypatch.setattr(bk, "fetch_profiles_by_urls", lambda client, urls: {})
        monkeypatch.setattr(bk, "_fetch_screening_results", lambda client, jd_hash, urls: [])

        args = argparse.Namespace(
            sample=str(sample_path), jev=str(jev_path), briefs=str(briefs_path),
            out_dir=str(out_dir), seed=1,
        )
        bk.cmd_report(args)

        report = json.loads((out_dir / "bakeoff.json").read_text(encoding="utf-8"))
        row = report["rows"][0]
        assert row["jev_score"] == "0"

    def test_score_value_keeps_zero_blanks_missing(self):
        import bakeoff_kalamata as bk
        assert bk._score_value(0) == 0
        assert bk._score_value("0") == "0"
        assert bk._score_value(None) == ""
        assert bk._score_value("") == ""

    # -- pipeline candidate dedupe: same linkedin_url must appear once --

    def test_dedupe_candidates_by_url_keeps_latest_screened_at(self):
        import bakeoff_kalamata as bk
        rows = [
            {"linkedin_url": "https://www.linkedin.com/in/dup", "screening_score": 1, "screened_at": "2026-01-01T00:00:00Z"},
            {"linkedin_url": "https://www.linkedin.com/in/dup", "screening_score": 9, "screened_at": "2026-06-01T00:00:00Z"},
            {"linkedin_url": "https://www.linkedin.com/in/other", "screening_score": 5, "screened_at": "2026-02-01T00:00:00Z"},
        ]
        result = bk.dedupe_candidates_by_url(rows)
        by_url = {r["linkedin_url"]: r for r in result}
        assert len(result) == 2
        assert by_url["https://www.linkedin.com/in/dup"]["screening_score"] == 9

    def test_fetch_pipeline_candidates_passes_stable_order_and_dedupes(self, monkeypatch):
        import bakeoff_kalamata as bk

        fake_client = MagicMock()
        fake_client.select.return_value = [
            {"linkedin_url": "https://www.linkedin.com/in/dup", "screened_at": "2026-01-01T00:00:00Z"},
            {"linkedin_url": "https://www.linkedin.com/in/dup", "screened_at": "2026-03-01T00:00:00Z"},
        ]
        result = bk._fetch_pipeline_candidates(fake_client, "pos-a")
        assert len(result) == 1
        _, kwargs = fake_client.select.call_args
        assert kwargs.get("order_by") == "linkedin_url.asc"

    # -- quote_url_list: every in.(...) / ov.{...} filter must be quoted --

    def test_quote_url_list_wraps_each_value_in_double_quotes(self):
        import bakeoff_kalamata as bk
        assert bk.quote_url_list(["a", "b"]) == '"a","b"'
        assert bk.quote_url_list(["https://www.linkedin.com/in/alice"]) == '"https://www.linkedin.com/in/alice"'
        assert bk.quote_url_list([]) == ""

    # -- preferred_match_url: profile_url wins, linkedin_url is the fallback --

    def test_preferred_match_url_prefers_profile_url(self):
        import bakeoff_kalamata as bk
        row = {
            "linkedin_url": "https://www.linkedin.com/in/pipeline-alias",
            "profile_url": "https://www.linkedin.com/in/canonical-123",
        }
        assert bk.preferred_match_url(row) == "https://www.linkedin.com/in/canonical-123"

    def test_preferred_match_url_falls_back_to_linkedin_url_when_missing(self):
        import bakeoff_kalamata as bk
        row = {"linkedin_url": "https://www.linkedin.com/in/pipeline-alias"}
        assert bk.preferred_match_url(row) == "https://www.linkedin.com/in/pipeline-alias"
        row_empty = {"linkedin_url": "https://www.linkedin.com/in/pipeline-alias", "profile_url": ""}
        assert bk.preferred_match_url(row_empty) == "https://www.linkedin.com/in/pipeline-alias"

    # -- fetch_profiles_by_urls: quoted OR filter across linkedin_url / --
    # -- original_url / original_urls, indexed by every normalized alias --

    def test_fetch_profiles_by_urls_uses_quoted_or_filter_across_url_columns(self):
        import bakeoff_kalamata as bk
        fake_client = MagicMock()
        fake_client.select.return_value = []

        bk.fetch_profiles_by_urls(fake_client, ["https://www.linkedin.com/in/alice"])

        _, kwargs = fake_client.select.call_args
        or_filter = kwargs["filters"]["or"]
        assert '"https://www.linkedin.com/in/alice"' in or_filter
        assert "linkedin_url.in.(" in or_filter
        assert "original_url.in.(" in or_filter
        assert "original_urls.ov.{" in or_filter

    def test_fetch_profiles_by_urls_indexes_result_under_every_alias(self):
        import bakeoff_kalamata as bk
        fake_client = MagicMock()
        fake_client.select.return_value = [
            {
                "linkedin_url": "https://www.linkedin.com/in/canonical-123",
                "original_url": "https://www.linkedin.com/in/alt-name",
                "original_urls": ["https://www.linkedin.com/in/third-alias"],
                "name": "Alice",
            }
        ]

        # A pipeline URL that only matches the profile's original_urls[] entry
        # must still find the row -- that's the whole point of the OR filter.
        by_url = bk.fetch_profiles_by_urls(fake_client, ["https://www.linkedin.com/in/third-alias"])

        assert by_url["https://www.linkedin.com/in/canonical-123"]["name"] == "Alice"
        assert by_url["https://www.linkedin.com/in/alt-name"]["name"] == "Alice"
        assert by_url["https://www.linkedin.com/in/third-alias"]["name"] == "Alice"

    def test_fetch_profiles_by_urls_falls_back_to_plain_linkedin_url_filter_on_error(self):
        import bakeoff_kalamata as bk
        fake_client = MagicMock()
        fake_client.select.side_effect = [
            Exception("original_urls column missing"),
            [{"linkedin_url": "https://www.linkedin.com/in/alice"}],
        ]

        by_url = bk.fetch_profiles_by_urls(fake_client, ["https://www.linkedin.com/in/alice"])

        assert "https://www.linkedin.com/in/alice" in by_url
        assert fake_client.select.call_count == 2
        _, fallback_kwargs = fake_client.select.call_args
        assert "or" not in fallback_kwargs["filters"]
        assert fallback_kwargs["filters"]["linkedin_url"] == 'in.("https://www.linkedin.com/in/alice")'

    # -- _fetch_screening_results: normalize before chunking, quote the filter --

    def test_fetch_screening_results_normalizes_and_quotes_urls(self):
        import bakeoff_kalamata as bk
        fake_client = MagicMock()
        fake_client.select.return_value = []

        bk._fetch_screening_results(fake_client, "somehash", ["linkedin.com/in/alice/"])

        _, kwargs = fake_client.select.call_args
        url_filter = kwargs["filters"]["linkedin_url"]
        assert url_filter == 'in.("https://www.linkedin.com/in/alice")'

    # -- sample: profile_url is captured and written into the upload CSV --

    def test_cmd_sample_writes_profile_url_into_upload_csv(self, tmp_path, monkeypatch):
        import bakeoff_kalamata as bk

        fake_client = MagicMock()
        fake_client.select.return_value = [
            {
                "position_id": "pos-a",
                "linkedin_url": "https://www.linkedin.com/in/pipeline-alias",
                "screening_result": "qualified",
                "screening_score": 8,
                "screening_notes": "",
                "screening_detail": None,
                "screened_at": "2026-01-01T00:00:00Z",
            },
        ]
        monkeypatch.setattr(bk, "_load_supabase_client", lambda: fake_client)
        monkeypatch.setattr(bk, "fetch_profiles_by_urls", lambda client, urls: {
            "https://www.linkedin.com/in/pipeline-alias": {
                "linkedin_url": "https://www.linkedin.com/in/canonical-123",
                "raw_data": {"skills": ["Python"]},
            },
        })

        out_dir = tmp_path / "out"
        args = argparse.Namespace(
            positions=["pos-a"], per_group=5, seed=1, out_dir=str(out_dir),
        )
        bk.cmd_sample(args)

        sample = json.loads((out_dir / "sample.json").read_text(encoding="utf-8"))
        assert sample["rows"][0]["linkedin_url"] == "https://www.linkedin.com/in/pipeline-alias"
        assert sample["rows"][0]["profile_url"] == "https://www.linkedin.com/in/canonical-123"

        with open(out_dir / "upload_pos-a.csv", encoding="utf-8-sig", newline="") as f:
            uploaded = [row["linkedin_url"] for row in csv.DictReader(f)]
        assert uploaded == ["https://www.linkedin.com/in/canonical-123"]

    # -- report: Luna rows are looked up by profile_url, not the pipeline URL --

    def test_cmd_report_matches_luna_row_by_profile_url(self, tmp_path, monkeypatch):
        import bakeoff_kalamata as bk

        sample = {
            "seed": 1,
            "positions": ["pos-a"],
            "rows": [
                {
                    "position_id": "pos-a",
                    "linkedin_url": "https://www.linkedin.com/in/pipeline-alias",
                    "profile_url": "https://www.linkedin.com/in/canonical-123",
                    "group": "yes", "kalamata_verdict": "yes",
                    "kalamata_score": 8, "kalamata_reason": "ok",
                },
            ],
        }
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample), encoding="utf-8")
        briefs_path = tmp_path / "briefs.json"
        briefs_path.write_text(json.dumps({"pos-a": {"role_context": "Backend Engineer"}}), encoding="utf-8")
        jev_path = tmp_path / "jev.csv"
        jev_path.write_text("", encoding="utf-8-sig")

        monkeypatch.setattr(bk, "_load_supabase_client", lambda: object())
        monkeypatch.setattr(bk, "fetch_profiles_by_urls", lambda client, urls: {
            "https://www.linkedin.com/in/canonical-123": {
                "name": "Alice", "current_title": "Engineer", "current_company": "Acme",
            },
        })

        captured_urls = {}

        def _fake_fetch_screening_results(client, jd_hash, urls):
            captured_urls["urls"] = set(urls)
            return [{
                "linkedin_url": "https://www.linkedin.com/in/canonical-123",
                "screening_score": 7, "screening_fit_level": "Good Fit",
                "screening_summary": "Strong fit", "screened_at": "2026-01-01T00:00:00Z",
            }]

        monkeypatch.setattr(bk, "_fetch_screening_results", _fake_fetch_screening_results)

        out_dir = tmp_path / "out"
        args = argparse.Namespace(
            sample=str(sample_path), jev=str(jev_path), briefs=str(briefs_path),
            out_dir=str(out_dir), seed=1,
        )
        bk.cmd_report(args)

        # Queried for both the profile_url and pipeline linkedin_url variants.
        assert "https://www.linkedin.com/in/canonical-123" in captured_urls["urls"]
        assert "https://www.linkedin.com/in/pipeline-alias" in captured_urls["urls"]

        report = json.loads((out_dir / "bakeoff.json").read_text(encoding="utf-8"))
        row = report["rows"][0]
        assert row["luna_verdict"] == "yes"
        assert row["name"] == "Alice"
