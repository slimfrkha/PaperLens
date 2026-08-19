"""Query paraphrasing for multi-query expansion (opt-in, see ``MultiQueryCfg``).

``generate_paraphrases`` asks the chat LLM for alternative phrasings of a query so
``Searcher.search`` can retrieve against each variant and RRF-fuse the resulting
rankings — a recall boost against how a question happens to be phrased. A single-shot
completion + JSON-array parse, same recipe as ``tagger.py``'s ``generate_tags``; any
parse failure degrades to an empty list rather than raising, so a flaky LLM just means
"no paraphrases this call," not a broken search.
"""

from __future__ import annotations

import json
import re

from .llm import LLMBackend

_SYSTEM = (
    "You rewrite search queries for a retrieval system over research papers. Produce "
    "alternative phrasings that preserve the original meaning but vary wording, "
    "terminology, or specificity, to help retrieve relevant passages the exact "
    "original wording might miss."
)


def _parse(text: str, n: int) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()][:n]


def generate_paraphrases(query: str, llm: LLMBackend, n: int = 3) -> list[str]:
    """Return up to ``n`` alternative phrasings of ``query``, excluding the original.

    Empty on any LLM failure or unparsable output — callers fall back to searching
    just the original query, never raise.
    """
    prompt = (
        f"Query: {query}\n\nReturn ONLY a JSON array of {n} alternative phrasings of this query."
    )
    try:
        raw = llm.complete(system=_SYSTEM, user=prompt, max_tokens=40 * n)
    except Exception:
        return []
    paraphrases = _parse(raw, n)  # already truncated to n
    original = query.strip().casefold()
    seen = {original}
    out: list[str] = []
    for p in paraphrases:
        key = p.casefold()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
