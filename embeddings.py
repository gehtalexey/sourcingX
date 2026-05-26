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


# ---------------------------------------------------------------------------
# Seniority / track derivation helpers
# ---------------------------------------------------------------------------
_MANAGER_KEYWORDS = {
    "manager", "director", " vp ", "v.p.", "head of", "chief",
    "cto", "cpo", "ceo", "coo", "president", "team lead",
}
_STAFF_KEYWORDS = {"staff ", "principal ", "distinguished ", "fellow"}
_SENIOR_KEYWORDS = {"senior", "sr.", " lead"}
_JUNIOR_KEYWORDS = {"junior", "jr.", "associate ", "entry level"}


def _derive_seniority(title: str) -> str:
    t = title.lower()
    if any(k in t for k in _STAFF_KEYWORDS):
        return "Staff/Principal"
    if any(k in t for k in _MANAGER_KEYWORDS):
        return "Manager/Leadership"
    if any(k in t for k in _SENIOR_KEYWORDS):
        return "Senior"
    if any(k in t for k in _JUNIOR_KEYWORDS):
        return "Junior"
    return "Mid-level"


def _derive_track(title: str) -> str:
    t = title.lower()
    if any(k in t for k in _MANAGER_KEYWORDS):
        return "Manager"
    return "Individual Contributor"


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
    """Build canonical embedding text focused on role, skills, and career path.

    Intentionally excludes: name, company names, headline, summary, education,
    and school names. These cause false-positive similarity matches (two people
    at the same company look "similar"; LinkedIn marketing copy dominates the
    vector). What actually matters for recruiting similarity is what someone
    does, what they build with, and where their career has gone.

    Changing this function invalidates all stored vectors — run
    ``backfill_embeddings.py --re-embed-changed --page-size 50`` after
    any modification.
    """
    current_title = _safe_str(profile.get("current_title"))
    location = _safe_str(profile.get("location"))
    skills = _join_list(profile.get("skills"), limit=30)

    # Recent titles only (no company names) — gives trajectory signal
    # without anchoring similarity to specific employers.
    recent_titles = _join_list(
        (profile.get("all_titles") or [])[:5], sep=" → "
    )

    seniority = _derive_seniority(current_title) if current_title else ""
    track = _derive_track(current_title) if current_title else ""

    lines = [
        f"Current role: {current_title}" if current_title else "",
        f"Seniority: {seniority}" if seniority else "",
        f"Track: {track}" if track else "",
        f"Skills: {skills}" if skills else "",
        f"Career path: {recent_titles}" if recent_titles else "",
        f"Location: {location}" if location else "",
    ]
    text = "\n".join(line for line in lines if line)
    return text[:MAX_INPUT_CHARS]


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
