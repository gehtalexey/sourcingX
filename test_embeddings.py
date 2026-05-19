"""
Tests for the embedding module.

Network-free: a fake OpenAI client stands in so we can verify batching,
ordering, and write-back payload shape without a real API key.
"""

from __future__ import annotations

import pytest

from embeddings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    build_embedding_text,
    compute_input_hash,
    embed_profiles_for_save,
    embed_text,
    embed_texts,
    estimate_embedding_cost,
    estimate_token_count,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_profile(**overrides) -> dict:
    base = {
        "linkedin_url": "https://www.linkedin.com/in/jane-doe/",
        "name": "Jane Doe",
        "current_title": "Senior Backend Engineer",
        "current_company": "Acme",
        "location": "Tel Aviv, Israel",
        "all_titles": ["Backend Engineer", "Junior Developer"],
        "all_employers": ["Acme", "Wayne Enterprises"],
        "all_schools": ["Technion"],
        "skills": ["Python", "PostgreSQL", "Redis"],
        "raw_data": {
            "headline": "Backend engineer working on distributed systems",
            "summary": "Ten years across fintech and adtech",
            "education_background": [
                {"school_name": "Technion", "degree_name": "BSc", "field_of_study": "CS"},
            ],
            "past_employers": [
                {"employee_title": "Junior Developer", "employer_name": "Wayne Enterprises"},
            ],
        },
    }
    base.update(overrides)
    return base


class FakeOpenAIResponse:
    """Mimics the shape of an OpenAI embeddings response."""
    def __init__(self, vectors):
        self.data = [type("E", (), {"embedding": v}) for v in vectors]


class FakeOpenAIEmbeddings:
    def __init__(self, dim: int = EMBEDDING_DIMENSIONS):
        self.dim = dim
        self.calls = []

    def create(self, model, input):  # noqa: A002 — match SDK kwarg name
        self.calls.append({"model": model, "input": list(input)})
        # Return deterministic, distinct vectors so order checks are meaningful.
        vectors = []
        for i, _ in enumerate(input):
            vectors.append([float(i) / self.dim] + [0.0] * (self.dim - 1))
        return FakeOpenAIResponse(vectors)


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeOpenAIEmbeddings()


# ---------------------------------------------------------------------------
# build_embedding_text
# ---------------------------------------------------------------------------
def test_build_embedding_text_includes_labeled_fields():
    profile = _make_profile()
    text = build_embedding_text(profile)

    assert "Name: Jane Doe" in text
    assert "Current role: Senior Backend Engineer" in text
    assert "Current company: Acme" in text
    assert "Headline: Backend engineer working on distributed systems" in text
    assert "Past employers: Acme, Wayne Enterprises" in text
    assert "Education: Technion" in text
    assert "Skills: Python, PostgreSQL, Redis" in text


def test_build_embedding_text_empty_profile_returns_empty_string():
    assert build_embedding_text({}) == ""
    assert build_embedding_text({"linkedin_url": "x"}) == ""


def test_build_embedding_text_skips_nan_and_none():
    text = build_embedding_text({
        "name": "Jane",
        "current_title": None,
        "current_company": "nan",
        "skills": [None, "Python", ""],
        "raw_data": {"headline": float("nan")},
    })
    assert "Name: Jane" in text
    assert "Current role" not in text
    assert "Current company" not in text  # 'nan' filtered
    assert "Headline" not in text
    assert "Skills: Python" in text


def test_build_embedding_text_truncates_long_input():
    long_summary = "x" * 60_000
    profile = _make_profile(raw_data={"summary": long_summary})
    text = build_embedding_text(profile)
    assert len(text) <= 24_000  # MAX_INPUT_CHARS


# ---------------------------------------------------------------------------
# compute_input_hash
# ---------------------------------------------------------------------------
def test_compute_input_hash_is_stable_and_distinct():
    a = compute_input_hash("hello world")
    b = compute_input_hash("hello world")
    c = compute_input_hash("hello world!")
    assert a == b
    assert a != c
    assert len(a) == 32  # md5 hex digest


# ---------------------------------------------------------------------------
# embed_text / embed_texts
# ---------------------------------------------------------------------------
def test_embed_texts_preserves_order_and_model():
    client = FakeOpenAIClient()
    vectors = embed_texts(client, ["one", "two", "three"], model="custom-model")
    assert len(vectors) == 3
    assert client.embeddings.calls[0]["model"] == "custom-model"
    assert client.embeddings.calls[0]["input"] == ["one", "two", "three"]


def test_embed_text_replaces_empty_with_space():
    client = FakeOpenAIClient()
    # An empty string would normally make OpenAI reject the request.
    vector = embed_text(client, "")
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert client.embeddings.calls[0]["input"] == [" "]


# ---------------------------------------------------------------------------
# embed_profiles_for_save
# ---------------------------------------------------------------------------
def test_embed_profiles_for_save_returns_update_payloads():
    client = FakeOpenAIClient()
    profiles = [_make_profile(linkedin_url="https://www.linkedin.com/in/a/"),
                _make_profile(linkedin_url="https://www.linkedin.com/in/b/")]
    payloads = embed_profiles_for_save(client, profiles)

    assert len(payloads) == 2
    for payload in payloads:
        assert set(payload.keys()) == {"linkedin_url", "embedding", "model", "input_hash"}
        assert payload["model"] == EMBEDDING_MODEL
        assert len(payload["embedding"]) == EMBEDDING_DIMENSIONS
        assert len(payload["input_hash"]) == 32


def test_embed_profiles_for_save_drops_empty_text_rows():
    client = FakeOpenAIClient()
    profiles = [
        _make_profile(linkedin_url="https://www.linkedin.com/in/a/"),
        {"linkedin_url": "https://www.linkedin.com/in/empty/"},  # no usable fields
    ]
    payloads = embed_profiles_for_save(client, profiles)
    assert len(payloads) == 1
    assert payloads[0]["linkedin_url"] == "https://www.linkedin.com/in/a/"


def test_embed_profiles_for_save_chunks_large_batches():
    client = FakeOpenAIClient()
    # Create 250 profiles → should split into 3 OpenAI calls (96 + 96 + 58).
    profiles = [
        _make_profile(linkedin_url=f"https://www.linkedin.com/in/p{i}/")
        for i in range(250)
    ]
    payloads = embed_profiles_for_save(client, profiles)
    assert len(payloads) == 250
    call_sizes = [len(c["input"]) for c in client.embeddings.calls]
    assert call_sizes == [EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_SIZE, 250 - 2 * EMBEDDING_BATCH_SIZE]


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------
def test_estimate_token_count_handles_empty():
    assert estimate_token_count("") == 0
    assert estimate_token_count(None) == 0


def test_estimate_token_count_uses_4_chars_per_token_heuristic():
    assert estimate_token_count("a" * 400) == 100


def test_estimate_embedding_cost_uses_text_embedding_3_small_rate():
    # $0.02 per 1M tokens → 1M tokens = $0.02
    assert estimate_embedding_cost(1_000_000) == pytest.approx(0.02)
    # Unknown model falls back to the default rate.
    assert estimate_embedding_cost(1_000_000, model="unknown") == pytest.approx(0.02)
