"""Pluggable rerankers for the second retrieval stage.

A reranker rescores the ``(query, chunk)`` candidates the dense search returned;
``Searcher`` then sorts by that score and keeps the top ``k``. Backends register
under a config ``type`` string via ``@register_reranker`` and expose a uniform
``score(query, docs) -> list[float]`` (one score per doc, in input order):

* ``hf``  — a local sentence-transformers cross-encoder (default
            ``BAAI/bge-reranker-v2-m3``); the original in-``Searcher`` behavior.
* ``llm`` — reuses the chat LLM to rate relevance pointwise (no new dependency;
            works offline against a local server).

Add a backend by dropping a ``@register_reranker("name")`` class here — no other
wiring needed.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .config import RerankerCfg
from .llm import LLMBackend


class Reranker(ABC):
    """Rescores candidate passages for a query."""

    @abstractmethod
    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per doc, in the same order as ``docs``.
        Higher is more relevant; ``Searcher`` sorts descending and truncates."""

    @classmethod
    @abstractmethod
    def build(cls, model: str, *, device: str | None, llm: LLMBackend | None) -> Reranker:
        """Construct from the config fields (ignoring the ones it doesn't use)."""


_RERANKERS: dict[str, type[Reranker]] = {}


def register_reranker(name: str):
    """Register a :class:`Reranker` subclass under a config ``type`` string."""

    def deco(cls: type[Reranker]) -> type[Reranker]:
        _RERANKERS[name] = cls
        return cls

    return deco


@register_reranker("hf")
class CrossEncoderReranker(Reranker):
    """Local sentence-transformers cross-encoder (lazy: loads on first ``score``)."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        scores = self.model.predict([(query, d) for d in docs])
        return [float(s) for s in scores]

    @classmethod
    def build(
        cls, model: str, *, device: str | None, llm: LLMBackend | None
    ) -> CrossEncoderReranker:
        return cls(model, device=device)


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


@register_reranker("llm")
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

    @classmethod
    def build(cls, model: str, *, device: str | None, llm: LLMBackend | None) -> LLMReranker:
        if llm is None:
            raise ValueError("The 'llm' reranker needs an LLM; pass build_reranker(..., llm=...).")
        return cls(llm)


def build_reranker(
    cfg: RerankerCfg, *, device: str | None = None, llm: LLMBackend | None = None
) -> Reranker:
    """Construct the reranker described by ``config.reranker`` via the registry."""
    try:
        cls = _RERANKERS[cfg.type]
    except KeyError:
        raise ValueError(
            f"Unknown reranker type {cfg.type!r} (expected one of {sorted(_RERANKERS)})"
        ) from None
    return cls.build(cfg.model, device=device, llm=llm)
