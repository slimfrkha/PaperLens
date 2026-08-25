"""Pluggable rerankers for the second retrieval stage.

A reranker rescores the ``(query, chunk)`` candidates the dense search returned;
``Searcher`` then sorts by that score and keeps the top ``k``. One backend per
``reranker.type`` variant (see ``RerankerCfg`` in ``config.py``), each exposing a
uniform ``score(query, docs) -> list[float]`` (one score per doc, in input order):

* ``hf``     — a local sentence-transformers cross-encoder (default
               ``BAAI/bge-reranker-v2-m3``); the original in-``Searcher`` behavior.
* ``llm``    — reuses the chat LLM to rate relevance pointwise (no new dependency;
               works offline against a local server).
* ``voyage`` — a dedicated rerank API (default ``rerank-2.5``), via LiteLLM's
               ``rerank()``; purpose-built for this, unlike ``llm``'s prompt-and-parse
               workaround.

Add a backend by registering a ``RerankerCfg`` variant in ``config.py`` and adding
a match arm to ``build_reranker`` below.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

import litellm

from .config import HFRerankerCfg, LLMRerankerCfg, RerankerCfg, VoyageRerankerCfg
from .llm import LLMBackend


class Reranker(ABC):
    """Rescores candidate passages for a query."""

    @abstractmethod
    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per doc, in the same order as ``docs``.
        Higher is more relevant; ``Searcher`` sorts descending and truncates."""


class CrossEncoderReranker(Reranker):
    """Local sentence-transformers cross-encoder (lazy: loads on first ``score``)."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        scores = self.model.predict([(query, d) for d in docs])
        return [float(s) for s in scores]


_LLM_SYSTEM = (
    "You are a search-relevance judge. Rate how well each passage answers the user's query."
)


def _parse_scores(text: str, n: int) -> list[float]:
    """Extract the first JSON array of numbers from ``text``, fit it to length ``n``.

    On any parse failure returns zeros — with a stable sort that preserves the
    dense-retrieval order, so a flaky LLM degrades to "no rerank" rather than noise.
    """
    m = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    if not m:
        return [0.0] * n
    try:
        raw = json.loads(m.group(0))
    except ValueError, TypeError:
        return [0.0] * n
    scores: list[float] = []
    for v in raw:
        try:
            scores.append(float(v))
        except ValueError, TypeError:
            scores.append(0.0)
    scores = (scores + [0.0] * n)[:n]  # pad short, truncate long
    return scores


class LLMReranker(Reranker):
    """Pointwise LLM reranker: one batched call rates every candidate 0-10."""

    def __init__(self, llm: LLMBackend, max_chars: int = 600):
        self.llm = llm
        self.max_chars = max_chars

    def _excerpt(self, doc: str) -> str:
        doc = doc.strip().replace("\n", " ")
        return doc[: self.max_chars]

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        passages = "\n\n".join(f"[{i}] {self._excerpt(d)}" for i, d in enumerate(docs))
        user = (
            f"Query: {query}\n\n"
            f"Passages:\n{passages}\n\n"
            f"Score each passage's relevance to the query from 0 (irrelevant) to 10 "
            f"(directly answers it). Return ONLY a JSON array of {len(docs)} numbers, "
            f"in passage order."
        )
        text = self.llm.complete(_LLM_SYSTEM, user, max_tokens=32 + 6 * len(docs))
        return _parse_scores(text, len(docs))


class VoyageReranker(Reranker):
    """Voyage AI's dedicated rerank API, via LiteLLM — purpose-built for scoring
    (query, passage) relevance, unlike ``LLMReranker``'s prompt-and-parse workaround.
    """

    def __init__(self, model_name: str, api_key_env: str = "VOYAGE_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env}. Export it (or add it to .env)."
            )
        self.model_name = model_name
        self._api_key = api_key

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        resp = litellm.rerank(
            model=f"voyage/{self.model_name}",
            query=query,
            documents=docs,
            top_n=len(docs),  # score every doc, not just the API's top matches
            api_key=self._api_key,
        )
        # Results come back sorted by relevance, not input order — `index` maps
        # each result back to its position in `docs` (see Reranker.score's contract).
        # A contract-violating index degrades that one doc to 0.0 (dense-retrieval
        # order) rather than raising — same failure philosophy as LLMReranker's
        # _parse_scores, just for an external API's response instead of an LLM's.
        scores = [0.0] * len(docs)
        for r in resp.results:
            if 0 <= r["index"] < len(docs):
                scores[r["index"]] = r["relevance_score"]
        return scores


def build_reranker(
    cfg: RerankerCfg, *, device: str | None = None, llm: LLMBackend | None = None
) -> Reranker:
    """Construct the reranker described by ``config.reranker`` (its variant)."""
    match cfg:
        case HFRerankerCfg():
            return CrossEncoderReranker(cfg.model, device=device, max_length=cfg.max_length)
        case LLMRerankerCfg():
            if llm is None:
                raise ValueError(
                    "The 'llm' reranker needs an LLM; pass build_reranker(..., llm=...)."
                )
            return LLMReranker(llm, max_chars=cfg.max_chars)
        case VoyageRerankerCfg():
            return VoyageReranker(cfg.model, api_key_env=cfg.api_key_env)
        case _:
            raise ValueError(f"Unknown reranker config: {type(cfg).__name__}")
