"""Tests for search_people_semantic_paged() — the description search's
auto-pagination past Crustdata's 100-results-per-request cap.

No network: every test passes a fake `fetch` that records its calls.
"""

from crustdata_search import search_people_semantic_paged


class FakeFetch:
    """Stands in for search_people_semantic. `pages` is a list of
    (profile_count, next_cursor) tuples returned in order."""

    def __init__(self, pages, total_count=1000):
        self.pages = list(pages)
        self.total_count = total_count
        self.calls = []

    def __call__(self, query, limit=20, cursor=None, api_key=None,
                 filters=None, tracker=None):
        self.calls.append({"query": query, "limit": limit, "cursor": cursor,
                           "filters": filters, "tracker": tracker})
        count, next_cursor = self.pages.pop(0)
        count = min(count, limit)
        return {
            "profiles": [{"i": len(self.calls), "n": n} for n in range(count)],
            "cursor": next_cursor,
            "total_count": self.total_count,
            "credits_used": round(count * 0.03, 2),
            "response_time_ms": 10,
        }


def test_250_target_takes_three_pages_of_100_100_50():
    fetch = FakeFetch([(100, "c1"), (100, "c2"), (100, "c3")])
    res = search_people_semantic_paged("founding engineers", 250, fetch=fetch)

    assert [c["limit"] for c in fetch.calls] == [100, 100, 50]
    assert [c["cursor"] for c in fetch.calls] == [None, "c1", "c2"]
    assert len(res["profiles"]) == 250
    assert res["cursor"] == "c3"  # last cursor, so Load More continues
    assert res["credits_used"] == 7.5
    assert res["total_count"] == 1000


def test_target_under_100_is_a_single_request():
    fetch = FakeFetch([(20, "c1")])
    res = search_people_semantic_paged("q", 20, fetch=fetch)
    assert len(fetch.calls) == 1
    assert fetch.calls[0]["limit"] == 20
    assert len(res["profiles"]) == 20
    assert res["cursor"] == "c1"


def test_stops_when_no_cursor():
    fetch = FakeFetch([(100, "c1"), (100, None)])
    res = search_people_semantic_paged("q", 500, fetch=fetch)
    assert len(fetch.calls) == 2
    assert len(res["profiles"]) == 200
    assert res["cursor"] is None


def test_stops_on_empty_page():
    fetch = FakeFetch([(100, "c1"), (0, "c2")])
    res = search_people_semantic_paged("q", 500, fetch=fetch)
    assert len(fetch.calls) == 2
    assert len(res["profiles"]) == 100
    assert res["credits_used"] == 3.0


def test_stops_at_total_count():
    fetch = FakeFetch([(100, "c1"), (100, "c2"), (100, "c3")], total_count=150)
    res = search_people_semantic_paged("q", 500, fetch=fetch)
    assert [c["limit"] for c in fetch.calls] == [100, 50]
    assert len(res["profiles"]) == 150


def test_passes_query_filters_and_tracker_to_every_page():
    tracker = object()
    filters = {"op": "and", "conditions": []}
    fetch = FakeFetch([(100, "c1"), (100, "c2")])
    search_people_semantic_paged("q", 200, filters=filters, tracker=tracker,
                                 fetch=fetch)
    assert all(c["query"] == "q" for c in fetch.calls)
    assert all(c["filters"] is filters for c in fetch.calls)
    assert all(c["tracker"] is tracker for c in fetch.calls)


def test_no_keep_means_nothing_removed():
    fetch = FakeFetch([(100, "c1"), (100, "c2")])
    res = search_people_semantic_paged("q", 200, fetch=fetch)
    assert res["removed"] == 0


def test_keep_filter_pages_further_to_fill_target():
    # 20 of the first page are excluded, so one more page of 20 fills the count.
    fetch = FakeFetch([(100, "c1"), (100, "c2"), (100, "c3")])
    res = search_people_semantic_paged(
        "q", 100, fetch=fetch,
        keep=lambda p: not (p["i"] == 1 and p["n"] < 20),
    )
    assert [c["limit"] for c in fetch.calls] == [100, 20]
    assert len(res["profiles"]) == 100
    assert res["removed"] == 20
    assert all(not (p["i"] == 1 and p["n"] < 20) for p in res["profiles"])
    assert res["cursor"] == "c2"


def test_keep_filter_stops_at_twice_the_target():
    fetch = FakeFetch([(100, "c1"), (100, "c2"), (100, "c3"), (100, "c4")])
    res = search_people_semantic_paged("q", 100, fetch=fetch,
                                       keep=lambda p: False)
    assert [c["limit"] for c in fetch.calls] == [100, 100]
    assert res["profiles"] == []
    assert res["removed"] == 200
    assert res["credits_used"] == 6.0


def test_on_page_reports_progress_before_each_follow_up():
    seen = []
    fetch = FakeFetch([(100, "c1"), (100, "c2"), (100, "c3")])
    search_people_semantic_paged("q", 300, fetch=fetch,
                                 on_page=lambda got, goal: seen.append((got, goal)))
    assert seen == [(100, 300), (200, 300)]
