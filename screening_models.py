"""
Screening model registry for SourcingX.

Before this module existed, the screening model was a literal string typed
into several places in dashboard.py (`ai_model = "gpt-5.6-luna"`). Changing
it, or adding a new model (Jev), meant editing code in multiple spots and
hoping none were missed. This module is the one place that decides which
model/provider screens a candidate; dashboard.py's call sites read from it
instead of hardcoding a string.

This is deliberately narrow. It does NOT own request-parameter shaping
(max_completion_tokens vs max_tokens, temperature), retry logic, or
per-call cost logging -- those already live in dashboard.py's
`_screening_api_call()` and usage_tracker.py, fixed in PR #131 before this
module existed. Adding a model here only makes it *selectable*; a genuinely
new model family may still need its own request-shaping branch in
`_screening_api_call()` (as gpt-5.x did) or its own client module (as Jev
did, in jev_client.py).
"""

from __future__ import annotations

# Today's live default (PR #131, merged 2026-09-17). Changing this line
# changes what every recruiter's screening runs on -- don't, without
# Alexey's explicit go-ahead.
DEFAULT_SCREEN_MODEL = "gpt-5.6-luna"

# provider: "openai" -> goes through _screening_api_call() in dashboard.py
#           "anthropic" -> same function, its Anthropic branch
#           "typesafe" -> jev_client.screen_with_jev() -- NOT wired into
#             screen_profile()'s live dispatch yet (see LIVE_DISPATCH_PROVIDERS
#             below). Listed here so it has one canonical name/display_name
#             that compare_screening_modes.py (and the 200-profile bake-off)
#             can reference, without implying config.json can select it.
SCREEN_MODELS = {
    "gpt-5.6-luna": {
        "provider": "openai",
        "display_name": "GPT-5.6 Luna",
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "display_name": "GPT-4.1 mini",
    },
    "jev": {
        "provider": "typesafe",
        "display_name": "Jev (TypeSafe)",
    },
}

# Providers screen_profile()/_screening_api_call() in dashboard.py can
# actually dispatch to today. Jev is deliberately excluded: its real
# behavior on a low-confidence answer is to defer (screening_result: None,
# "read this as screening_incomplete" per jev_client.py's own docstring),
# and dashboard.py has no "incomplete" outcome for screen_profile() to
# return into yet -- that's parked work (Alexey's call, 2026-09-23), not
# built here. Wiring Jev into the live dashboard is a separate follow-up
# once that exists. Until then, Jev is only ever called directly (e.g. by
# the 200-profile bake-off script), never through get_screen_model().
LIVE_DISPATCH_PROVIDERS = {"openai", "anthropic"}


def get_screen_model(config: dict | None = None) -> tuple[str, str]:
    """Return (ai_model, ai_provider) for LIVE dashboard screening.

    Resolution order: `config['screen_model']` if set, recognized, AND its
    provider is one screen_profile() can actually dispatch to; otherwise
    DEFAULT_SCREEN_MODEL. A config value naming an unknown model, a model
    since removed from SCREEN_MODELS, or a model whose provider isn't live
    yet (Jev) all fail closed to the default -- never pass an unhandled
    provider string through to the call code.

    `config` is the dict `load_config()` in dashboard.py returns; pass None
    (or omit) to just get the default.
    """
    requested = (config or {}).get("screen_model")
    entry = SCREEN_MODELS.get(requested)
    if entry is None or entry["provider"] not in LIVE_DISPATCH_PROVIDERS:
        requested = DEFAULT_SCREEN_MODEL
    return requested, SCREEN_MODELS[requested]["provider"]
