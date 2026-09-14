"""WebSearcher: result parsing and its degrade-don't-raise contract. Never hits the real
network — the `ddgs.DDGS` client is monkeypatched in every case that would."""

from __future__ import annotations

import pytest

from rag.config import WebSearchCfg
from rag.web_search import WebResult, WebSearcher, build_web_searcher


class _FakeDDGS:
    """Stand-in for ddgs.DDGS: returns scripted rows, or raises if `error` is set."""

    rows: list[dict] = []
    error: Exception | None = None

    def __init__(self, *args, **kwargs):
        pass

    def text(self, query, max_results=None):
        if _FakeDDGS.error is not None:
            raise _FakeDDGS.error
        return _FakeDDGS.rows[:max_results] if max_results else _FakeDDGS.rows


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeDDGS.rows = []
    _FakeDDGS.error = None
    yield


@pytest.fixture
def patch_ddgs(monkeypatch):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", _FakeDDGS)


def test_empty_query_returns_empty_without_touching_the_network():
    # No patch_ddgs fixture: an empty query must short-circuit before importing/calling ddgs.
    assert WebSearcher().search("   ") == []


def test_parses_rows_into_web_results(patch_ddgs):
    _FakeDDGS.rows = [
        {"title": "External Reference", "href": "https://example.com/ref", "body": "snippet"},
    ]
    results = WebSearcher(k=3).search("external concept")
    assert results == [
        WebResult(title="External Reference", url="https://example.com/ref", snippet="snippet"),
    ]


def test_rows_without_a_url_are_skipped(patch_ddgs):
    _FakeDDGS.rows = [
        {"title": "no link", "href": "", "body": "x"},
        {"title": "ok", "href": "https://example.com", "body": "y"},
    ]
    results = WebSearcher().search("q")
    assert [r.url for r in results] == ["https://example.com"]


def test_non_http_and_credentialed_urls_are_skipped(patch_ddgs):
    _FakeDDGS.rows = [
        {"title": "script", "href": "javascript:alert(1)", "body": "x"},
        {"title": "data", "href": "data:text/html,bad", "body": "x"},
        {"title": "relative", "href": "/result", "body": "x"},
        {"title": "credentials", "href": "https://user:pass@example.com", "body": "x"},
        {"title": "safe", "href": "https://example.com/result", "body": "ok"},
    ]

    results = WebSearcher().search("q")

    assert [r.url for r in results] == ["https://example.com/result"]


def test_ddgs_failure_degrades_to_empty(patch_ddgs):
    _FakeDDGS.error = RuntimeError("rate limited")
    # Must not raise — a failed web search falls back to "answer from the papers".
    assert WebSearcher().search("q") == []


def test_build_web_searcher_carries_config():
    ws = build_web_searcher(WebSearchCfg(k=7, timeout=4.0))
    assert ws.k == 7
    assert ws.timeout == 4.0
