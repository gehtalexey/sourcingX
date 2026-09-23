"""Tests for usage_tracker.py's pricing fallback. Before this fix, an
unrecognized model silently used gpt-4o-mini's rate -- the exact bug Codex
found for gpt-5.6-luna itself in PR #131 (underreporting cost by up to 50%)
before that model got its own pricing entry. Any future unknown model
(a typo, a renamed model, Jev if it's ever priced per-token) should warn
loudly instead of silently mispricing."""

import warnings

import pytest

from usage_tracker import calculate_openai_cost, _openai_pricing_for, OPENAI_PRICING


def test_known_model_prices_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost = calculate_openai_cost(1_000_000, 1_000_000, model="gpt-5.6-luna")
    expected = OPENAI_PRICING["gpt-5.6-luna"]["input"] + OPENAI_PRICING["gpt-5.6-luna"]["output"]
    assert cost == pytest.approx(expected)


def test_unknown_model_warns_and_falls_back_to_gpt4o_mini():
    with pytest.warns(UserWarning, match="totally-made-up-model"):
        cost = calculate_openai_cost(1_000_000, 1_000_000, model="totally-made-up-model")
    fallback = OPENAI_PRICING["gpt-4o-mini"]
    assert cost == pytest.approx(fallback["input"] + fallback["output"])


def test_openai_pricing_for_unknown_model_warns_and_returns_fallback_dict():
    with pytest.warns(UserWarning):
        pricing = _openai_pricing_for("another-made-up-model")
    assert pricing == OPENAI_PRICING["gpt-4o-mini"]
