"""Query paraphrasing for multi-query expansion (opt-in, see ``MultiQueryCfg``).

``generate_paraphrases`` asks the chat LLM for alternative phrasings of a query so
``Searcher.search`` can retrieve against each variant and RRF-fuse the resulting
rankings — a recall boost against how a question happens to be phrased. A single-shot,
validated-and-retried structured completion, same recipe as ``tagger.py``'s
``generate_tags`` (both go through ``LLMBackend.complete_structured``). Unlike
tagger.py, there's no caller-side handler to rely on here — ``Searcher.search`` and
``eval/optimizer.py`` call this with no try/except of their own — so any failure
(LLM error, or a malformed reply that exhausts instructor's retries) is caught and
logged locally, degrading to an empty list rather than raising: a flaky LLM just
means "no paraphrases this call," not a broken search.
"""

from __future__ import annotations

from pydantic import BaseModel

from .llm import LLMBackend

_SYSTEM = (
    "You rewrite search queries for a retrieval system over research papers. Produce "
    "alternative phrasings that preserve the original meaning but vary wording, "
    "terminology, or specificity, to help retrieve relevant passages the exact "
    "original wording might miss."
)


class _ParaphrasesOut(BaseModel):
    paraphrases: list[str]


def generate_paraphrases(query: str, llm: LLMBackend, n: int = 3) -> list[str]:
    """Return up to ``n`` alternative phrasings of ``query``, excluding the original.

    Empty on any LLM failure or unparsable output — callers fall back to searching
    just the original query, never raise.
    """
    prompt = f"Query: {query}\n\nGive {n} alternative phrasings of this query."
    try:
        res = llm.complete_structured(
            system=_SYSTEM, user=prompt, response_model=_ParaphrasesOut, max_tokens=40 * n
        )
    except Exception as e:
        print(f"  [warn] paraphrase generation failed: {e}")
        return []
    paraphrases = res.paraphrases[:n]
    original = query.strip().casefold()
    seen = {original}
    out: list[str] = []
    for p in paraphrases:
        key = p.casefold()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
