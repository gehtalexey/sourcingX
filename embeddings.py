"""
Profile Embeddings Module

Builds the text that represents a profile semantically, calls OpenAI's
embedding API, and exposes helpers used by the backfill worker, the
enrichment pipeline, and the "find similar profiles" search.

Design notes
------------
* Model: ``text-embedding-3-small`` (1536 dims). Cheap (~$0.02 / 1M tokens),
  fast, and matches the column dimension defined in migration 019.
* Input hash: we md5 the embedding text. The DB stores it so the backfill
  worker can skip rows whose source text hasn't changed.
* This module never *reads* config.json directly — callers pass an
  ``OpenAI`` client. That keeps it easy to unit-test with a fake.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

# OpenAI client is created by the caller (mirrors how `dashboard.py` and
# `crustdata_search.py` already do it) so we only import the type.
try:
    from openai import OpenAI  # noqa: F401  — type hint only at runtime
except ImportError:  # pragma: no cover — handled by callers
    OpenAI = None  # type: ignore[assignment]


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# OpenAI accepts up to 2048 inputs per request; we stay well under that to
# keep individual requests small and recoverable on retry.
EMBEDDING_BATCH_SIZE = 96

# OpenAI's text-embedding-3-small has an 8192-token context. One token is
# ~4 characters, so 24,000 chars is a comfortable ceiling that leaves room
# for tokeniser variance and avoids 400s on unusually long profiles.
MAX_INPUT_CHARS = 24_000


# ---------------------------------------------------------------------------
# Embedding-text construction
# ---------------------------------------------------------------------------
def _safe_str(value) -> str:
    """Coerce ``value`` to a stripped string; treat None/NaN as empty."""
    if value is None:
        return ""
    try:
        # pandas NaN is a float; str(NaN) -> 'nan'. Filter it out.
        if isinstance(value, float) and value != value:
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _join_list(value, sep: str = ", ", limit: int = 30) -> str:
    """Join a list-of-strings field into one line, capped to ``limit`` items."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    try:
        items = [_safe_str(v) for v in value if _safe_str(v)]
    except TypeError:
        return ""
    if not items:
        return ""
    return sep.join(items[:limit])


def _education_summary(raw_data: dict) -> str:
    """One-line summary of education entries from Crustdata raw data."""
    if not raw_data:
        return ""
    schools = raw_data.get("education_background") or []
    parts = []
    for entry in schools[:6]:
        if not isinstance(entry, dict):
            continue
        bits = [
            _safe_str(entry.get("school_name")),
            _safe_str(entry.get("degree_name")),
            _safe_str(entry.get("field_of_study")),
        ]
        joined = " — ".join(b for b in bits if b)
        if joined:
            parts.append(joined)
    return " | ".join(parts)


def _past_roles_summary(raw_data: dict) -> str:
    """Comma-separated short summary of past employer roles."""
    if not raw_data:
        return ""
    past = raw_data.get("past_employers") or []
    parts = []
    for entry in past[:12]:
        if not isinstance(entry, dict):
            continue
        title = _safe_str(entry.get("employee_title"))
        company = _safe_str(entry.get("employer_name") or entry.get("employee_company"))
        combined = " @ ".join(b for b in (title, company) if b)
        if combined:
            parts.append(combined)
    return "; ".join(parts)


def build_embedding_text(profile: dict) -> str:
    """Build the canonical text we embed for a profile.

    Accepts a ``profiles`` row (dict). Uses the cheap indexed columns first
    and falls back into ``raw_data`` for headline / summary / education
    detail. Returns an empty string when there's nothing meaningful to
    embed — the caller should skip those rows.

    The shape of this text is part of the embedding's identity: changing it
    invalidates every stored vector. If you must change it, bump the
    ``embedding_model`` tag (e.g. add ``+v2``) so the backfill worker
    re-embeds rows automatically via the input-hash check.
    """
    raw_data = profile.get("raw_data") or {}
    if not isinstance(raw_data, dict):
        raw_data = {}

    headline = _safe_str(raw_data.get("headline"))
    summary = _safe_str(raw_data.get("summary"))
    current_title = _safe_str(profile.get("current_title"))
    current_company = _safe_str(profile.get("current_company"))
    location = _safe_str(profile.get("location"))

    all_titles = _join_list(profile.get("all_titles"))
    all_employers = _join_list(profile.get("all_employers"))
    all_schools = _join_list(profile.get("all_schools"))
    skills = _join_list(profile.get("skills"), limit=50)

    education_detail = _education_summary(raw_data)
    past_detail = _past_roles_summary(raw_data)

    # Labeled fields help the embedding model weight semantic categories;
    # plain prose collapses into a single fuzzy meaning, while labels keep
    # "title" close to "title" across profiles. This is the same approach
    # OpenAI's own retrieval cookbook recommends for structured records.
    lines = [
        f"Current role: {current_title}" if current_title else "",
        f"Current company: {current_company}" if current_company else "",
        f"Location: {location}" if location else "",
        f"Headline: {headline}" if headline else "",
        f"Summary: {summary}" if summary else "",
        f"Past titles: {all_titles}" if all_titles else "",
        f"Past employers: {all_employers}" if all_employers else "",
        f"Past roles: {past_detail}" if past_detail else "",
        f"Education: {all_schools}" if all_schools else "",
        f"Education detail: {education_detail}" if education_detail else "",
        f"Skills: {skills}" if skills else "",
    ]
    text = "\n".join(line for line in lines if line)

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    return text


def compute_input_hash(text: str) -> str:
    """Stable hash of the embedding input. Used to skip already-embedded rows."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# OpenAI calls
# ---------------------------------------------------------------------------
def embed_texts(
    client,
    texts: list[str],
    model: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Embed a batch of texts in one OpenAI call.

    Returns a list of vectors aligned with the input order. Empty/blank
    inputs are replaced by a single space so OpenAI doesn't reject the
    request; callers should filter those out beforehand if they care.
    """
    if not texts:
        return []
    cleaned = [t if t and t.strip() else " " for t in texts]
    response = client.embeddings.create(model=model, input=cleaned)
    # Response order matches input order per OpenAI spec.
    return [item.embedding for item in response.data]


def embed_text(client, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Embed a single text. Thin convenience wrapper over ``embed_texts``."""
    vectors = embed_texts(client, [text], model=model)
    return vectors[0] if vectors else []


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------
# OpenAI pricing for embedding models (per 1M tokens, input only).
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def estimate_embedding_cost(total_tokens: int, model: str = EMBEDDING_MODEL) -> float:
    """Return USD cost for ``total_tokens`` at the given model's rate."""
    rate = EMBEDDING_PRICING.get(model, EMBEDDING_PRICING[EMBEDDING_MODEL])
    return (total_tokens / 1_000_000) * rate


def estimate_token_count(text: str) -> int:
    """Cheap heuristic token estimate (~4 chars per token).

    Used for cost preview and usage-tracker logging when we don't get a
    real token count back. The OpenAI embeddings response *does* include
    ``usage.total_tokens``; prefer that when available.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------
def embed_profiles_for_save(
    openai_client,
    profiles: list[dict],
    model: str = EMBEDDING_MODEL,
) -> list[dict]:
    """Embed a list of freshly-enriched profiles.

    Takes profile dicts (must include the columns used by
    ``build_embedding_text``) and returns a list of dicts ready for
    ``db.update_profile_embedding`` — one per input profile that produced
    non-empty text. Order matches the input order; profiles that yielded
    empty text are omitted.

    Failures (network, OpenAI errors) bubble up to the caller so the
    enrichment pipeline can log + continue without rolling back saved
    rows. Callers should catch broadly.
    """
    if not profiles:
        return []

    rows = []
    texts = []
    hashes = []
    for profile in profiles:
        text = build_embedding_text(profile)
        if not text:
            continue
        rows.append(profile)
        texts.append(text)
        hashes.append(compute_input_hash(text))

    if not rows:
        return []

    # Chunk to keep individual OpenAI payloads small.
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        chunk = texts[i : i + EMBEDDING_BATCH_SIZE]
        vectors.extend(embed_texts(openai_client, chunk, model=model))

    return [
        {
            "linkedin_url": row["linkedin_url"],
            "embedding": vector,
            "model": model,
            "input_hash": h,
        }
        for row, vector, h in zip(rows, vectors, hashes)
    ]


__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_PRICING",
    "build_embedding_text",
    "compute_input_hash",
    "embed_text",
    "embed_texts",
    "embed_profiles_for_save",
    "estimate_embedding_cost",
    "estimate_token_count",
]
