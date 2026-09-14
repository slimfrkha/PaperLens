"""Keyless web search for the chat agent's `web_search` tool.

A thin wrapper over DuckDuckGo (`ddgs`) — no API key, no external service to sign up
for. Returns a small list of `WebResult` (title, url, snippet) for the agent to cite by
URL. Deliberately **degrades to an empty list, never raises**: `ddgs` scrapes DuckDuckGo
and is rate-limited / brittle to HTML changes, and this sits on the default chat path, so
a failed search must fall back to "answer from the papers" rather than 500 the whole turn
— the same degrade-don't-fail contract the reranker has in ``search.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import WebSearchCfg


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


class WebSearcher:
    def __init__(self, k: int = 5, timeout: float = 10.0):
        self.k = k
        self.timeout = timeout

    def search(self, query: str, k: int | None = None) -> list[WebResult]:
        query = query.strip()
        if not query:
            return []
        # Imported lazily: keeps the module import cheap and means an environment without
        # `ddgs` only fails if web search is actually used, not at startup.
        try:
            from ddgs import DDGS
        except ImportError:
            return []
        limit = k or self.k
        try:
            ddgs = DDGS(timeout=self.timeout) if self.timeout > 0 else DDGS()
            rows = ddgs.text(query, max_results=limit)
        except Exception as e:
            print(f"  [warn] web search failed for {query!r}: {e}")
            return []
        results: list[WebResult] = []
        for row in rows or []:
            url = (row.get("href") or "").strip()
            if not _is_safe_http_url(url):
                continue
            results.append(
                WebResult(
                    title=(row.get("title") or url).strip(),
                    url=url,
                    snippet=(row.get("body") or "").strip(),
                )
            )
        return results


def _is_safe_http_url(url: str) -> bool:
    """Only pass absolute HTTP(S) result links to the browser.

    Search results are network-controlled data. React 18 does not block dangerous URL
    schemes such as ``javascript:`` on anchors, so the backend must reject anything that
    is not an ordinary web URL before it enters a citation or saved chat.
    """
    if not url or any(ord(char) < 32 for char in url):
        return False
    try:
        parsed = urlsplit(url)
        # Accessing hostname/port performs validation that urlsplit itself defers.
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() in {"http", "https"}
        and hostname is not None
        and parsed.username is None
        and parsed.password is None
    )


def build_web_searcher(cfg: WebSearchCfg) -> WebSearcher:
    """Instantiate the web searcher for a config. Gating on ``cfg.enabled`` is the
    caller's job (the agent offers the tool only when enabled and the turn opts in) —
    this always returns a usable searcher, mirroring ``build_reranker``/``build_embedder``."""
    return WebSearcher(k=cfg.k, timeout=cfg.timeout)
