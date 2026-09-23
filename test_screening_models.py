"""Tests for screening_models.py -- the one place that decides which
model/provider screens a candidate, replacing literal strings that used to
be typed into several places in dashboard.py."""

import screening_models
from screening_models import get_screen_model, SCREEN_MODELS, DEFAULT_SCREEN_MODEL, LIVE_DISPATCH_PROVIDERS


def test_default_model_is_live_dispatchable():
    """The default itself must resolve to a provider screen_profile() can
    actually call, or every fallback path breaks."""
    assert DEFAULT_SCREEN_MODEL in SCREEN_MODELS
    assert SCREEN_MODELS[DEFAULT_SCREEN_MODEL]["provider"] in LIVE_DISPATCH_PROVIDERS


def test_get_screen_model_with_no_config_returns_default():
    model, provider = get_screen_model(None)
    assert model == DEFAULT_SCREEN_MODEL
    assert provider == SCREEN_MODELS[DEFAULT_SCREEN_MODEL]["provider"]

    model2, provider2 = get_screen_model({})
    assert (model2, provider2) == (model, provider)


def test_get_screen_model_honors_recognized_live_config_value():
    model, provider = get_screen_model({"screen_model": "gpt-4.1-mini"})
    assert model == "gpt-4.1-mini"
    assert provider == "openai"


def test_get_screen_model_falls_back_on_unrecognized_value():
    """A typo, or a model removed from SCREEN_MODELS, must never reach the
    call code as an unhandled string -- fail closed to the default."""
    model, provider = get_screen_model({"screen_model": "gpt-99-nonexistent"})
    assert model == DEFAULT_SCREEN_MODEL
    assert provider == SCREEN_MODELS[DEFAULT_SCREEN_MODEL]["provider"]


def test_get_screen_model_never_selects_jev_for_live_dispatch():
    """Jev is listed in SCREEN_MODELS (for a shared display name/table other
    code can reference) but its provider isn't wired into screen_profile()'s
    live dispatch yet -- config.json can't select it via this function."""
    assert SCREEN_MODELS["jev"]["provider"] == "typesafe"
    assert "typesafe" not in LIVE_DISPATCH_PROVIDERS

    model, provider = get_screen_model({"screen_model": "jev"})
    assert model == DEFAULT_SCREEN_MODEL
    assert provider != "typesafe"


def test_every_registered_model_has_a_display_name():
    for name, entry in SCREEN_MODELS.items():
        assert entry.get("display_name"), f"{name} is missing a display_name"
        assert entry.get("provider"), f"{name} is missing a provider"
