"""
"Find Similar Profiles" — semantic search powered by embeddings.

Given a LinkedIn URL, this module:
1. Looks the profile up in Supabase.
2. Gets (or generates on the fly) its embedding.
3. Calls the ``match_profiles_by_embedding`` RPC to return ranked matches.

The Streamlit UI lives in ``dashboard.py``; this module exposes the
plumbing so the UI code stays thin and the logic is testable.
"""

from __future__ import annotations

from typing import Optional

from db import (
    SupabaseClient,
    find_similar_profiles_rpc,
    get_profile,
    update_profile_embedding,
)
from embeddings import (
    EMBEDDING_MODEL,
    build_embedding_text,
    compute_input_hash,
    embed_text,
)
from normalizers import normalize_linkedin_url


class SimilarProfileError(Exception):
    """Raised when we can't find or build an embedding to search against."""


def get_or_build_query_embedding(
    db_client: SupabaseClient,
    openai_client,
    linkedin_url: str,
) -> tuple[list[float], dict, bool]:
    """Return the query embedding for ``linkedin_url``.

    Resolution order:
      1. Profile is in the DB and already has an ``embedding`` → reuse it.
      2. Profile is in the DB but missing an embedding → build the
         embedding text from the stored row, embed it now, write it back,
         and return it.
      3. Profile is not in the DB → raise ``SimilarProfileError``. The
         caller should enrich it via Crustdata first (separate UI step)
         to avoid surprise credit consumption inside this function.

    Returns:
        Tuple of ``(embedding, profile_row, was_freshly_embedded)``.
    """
    normalized = normalize_linkedin_url(linkedin_url)
    if not normalized:
        raise SimilarProfileError(f"Not a valid LinkedIn URL: {linkedin_url}")

    profile = get_profile(db_client, normalized)
    if not profile:
        raise SimilarProfileError(
            "This profile isn't in our database yet. Enrich it first "
            "(via the Load tab or Crustdata search) before searching for "
            "similar profiles."
        )

    existing = profile.get("embedding")
    if existing:
        return existing, profile, False

    # Profile exists but has no embedding — generate one on the fly.
    text = build_embedding_text(profile)
    if not text:
        raise SimilarProfileError(
            "This profile has no usable text to embed (missing title, "
            "headline, summary, employers, and education). Re-enrich it "
            "or check that raw_data is populated."
        )

    vector = embed_text(openai_client, text, model=EMBEDDING_MODEL)
    if not vector:
        raise SimilarProfileError("OpenAI returned an empty embedding.")

    update_profile_embedding(
        db_client,
        normalized,
        vector,
        EMBEDDING_MODEL,
        compute_input_hash(text),
    )
    return vector, profile, True


def search_similar(
    db_client: SupabaseClient,
    openai_client,
    linkedin_url: str,
    match_count: int = 20,
    min_similarity: float = 0.0,
    exclude_self: bool = True,
) -> dict:
    """High-level "find similar profiles" entry point.

    Returns a dict::

        {
            "query_profile": <row>,        # the profile we searched FROM
            "freshly_embedded": <bool>,    # True if we just embedded it
            "matches": [<row>, ...],       # ranked by similarity desc
        }

    ``matches`` rows always include a ``similarity`` float in [0, 1].
    """
    embedding, query_profile, freshly = get_or_build_query_embedding(
        db_client, openai_client, linkedin_url
    )

    # Ask for one extra so we can drop the self-match without coming up short.
    rpc_count = match_count + 1 if exclude_self else match_count

    raw_matches = find_similar_profiles_rpc(
        db_client,
        query_embedding=embedding,
        match_count=rpc_count,
        min_similarity=min_similarity,
    )

    if exclude_self:
        target_url = (query_profile or {}).get("linkedin_url")
        raw_matches = [
            row for row in raw_matches
            if row.get("linkedin_url") != target_url
        ][:match_count]
    else:
        raw_matches = raw_matches[:match_count]

    return {
        "query_profile": query_profile,
        "freshly_embedded": freshly,
        "matches": raw_matches,
    }
