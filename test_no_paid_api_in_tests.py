"""
Regression test: the suite must never call a paid API.

conftest.py installs a session-wide guard on requests' HTTPAdapter and on
httpx's Client/AsyncClient (the Anthropic and OpenAI SDKs use httpx) that
refuses any request to anthropic.com, openai.com, crustdata.com or
salesql.com hosts before a socket is opened. Before this,
test_structured_screening.py built a real Anthropic client from config.json
and spent money on every pytest run.
"""

import asyncio

import pytest
import requests

GUARD_MSG = "tests must not call paid APIs"


@pytest.mark.parametrize("url", [
    "https://api.crustdata.com/screener/person/enrich",
    "https://api.salesql.com/persons/enrich",
    "https://api-public.salesql.com/v1/persons/enrich",
    "https://api.anthropic.com/v1/messages",
    "https://api.openai.com/v1/chat/completions",
])
def test_requests_to_paid_api_is_blocked(url):
    with pytest.raises(RuntimeError, match=GUARD_MSG):
        requests.post(url, json={}, timeout=1)


@pytest.mark.parametrize("url", [
    "https://api.anthropic.com/v1/messages",
    "https://api.openai.com/v1/chat/completions",
    "https://api.crustdata.com/person/enrich",
])
@pytest.mark.parametrize("module", ["httpx", "httpx2"])  # httpx2: openai 3.x
def test_httpx_to_paid_api_is_blocked(module, url):
    httpx = pytest.importorskip(module)
    with httpx.Client() as client:
        with pytest.raises(RuntimeError, match=GUARD_MSG):
            client.post(url, json={}, timeout=1)


def test_httpx_async_to_paid_api_is_blocked():
    httpx = pytest.importorskip("httpx")

    async def call():
        async with httpx.AsyncClient() as client:
            await client.post("https://api.anthropic.com/v1/messages", json={}, timeout=1)

    with pytest.raises(RuntimeError, match=GUARD_MSG):
        asyncio.run(call())


def _assert_guard_fired(excinfo):
    """The SDKs wrap transport errors in APIConnectionError; find our guard."""
    err = excinfo.value
    while err is not None:
        if isinstance(err, RuntimeError) and GUARD_MSG in str(err):
            return
        err = err.__cause__ or err.__context__
    pytest.fail(f"guard did not fire, got: {excinfo.value!r}")


def test_anthropic_sdk_is_blocked():
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="test-key", max_retries=0)
    with pytest.raises(Exception) as excinfo:
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    _assert_guard_fired(excinfo)


def test_openai_sdk_is_blocked():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key="test-key", max_retries=0)
    with pytest.raises(Exception) as excinfo:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    _assert_guard_fired(excinfo)


def test_non_paid_host_is_not_blocked_by_guard():
    """The guard only matches paid-API domains, not look-alikes."""
    import conftest
    assert conftest._is_paid_api_host("api.anthropic.com")
    assert conftest._is_paid_api_host("api-public.salesql.com")
    assert not conftest._is_paid_api_host("notanthropic.com")
    assert not conftest._is_paid_api_host("example.com")
