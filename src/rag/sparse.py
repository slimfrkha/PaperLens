"""BM25 sparse retrieval + reciprocal rank fusion, for hybrid dense+sparse search.

A leaf module (no upstream ``rag`` deps), like ``chunking.py``/``embedders.py``: ``search.py``
imports it without creating a cycle.

Unlike the reranker (a stateless model — scores whatever ``(query, doc)`` pairs it's handed, no
corpus dependency), a :class:`BM25Index` is a **snapshot** of the corpus at build time (BM25's
IDF needs global corpus stats). ``built_at_count`` lets a caller detect a stale snapshot after
the collection has grown (see ``Searcher.sparse`` in ``search.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """A BM25 index over a fixed snapshot of chunk ids/text/paper ids.

    ``built_at_count`` records the collection size at build time, so a caller can detect
    staleness (``collection.count() != built_at_count``) after new chunks are upserted.
    """

    ids: list[str]
    paper_ids: list[str]
    built_at_count: int
    bm25: BM25Okapi = field(repr=False)

    def search(self, query: str, n: int, allowed_ids: set[str] | None = None) -> list[str]:
        """Top-``n`` chunk ids for ``query``, optionally restricted to chunks whose
        ``paper_id`` is in ``allowed_ids`` — every real caller (``Searcher.search``) builds
        this from a resolved ``paper``/``paper_ids`` filter, so it's paper ids, not chunk ids;
        that's what ``paper_ids`` (parallel to ``ids``) exists for.

        ``BM25Okapi.get_scores`` always scores the full corpus it was built from — there is no
        cheaper "restrict-then-score" API — so filtering by ``allowed_ids`` necessarily happens
        on the output, not the input. That's a deliberate constraint of the library, not an
        oversight here.

        Zero-score docs (no query term appears anywhere in them) are dropped before the
        top-``n`` cut: without this, a query with fewer true lexical matches than ``n`` pads
        the result with docs the query shares no vocabulary with, injecting corpus-order noise
        into RRF fusion instead of leaving that slot to dense recall.
        """
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self.ids, self.paper_ids, scores, strict=True), key=lambda t: t[2], reverse=True
        )
        ranked = [(cid, pid, s) for cid, pid, s in ranked if s > 0]
        if allowed_ids is not None:
            ranked = [(cid, pid, s) for cid, pid, s in ranked if pid in allowed_ids]
        return [cid for cid, _, _ in ranked[:n]]


def build_sparse_index(collection, k1: float = 1.5, b: float = 0.75) -> BM25Index:
    """Build a :class:`BM25Index` from every chunk currently in ``collection``."""
    got = collection.get(include=["documents", "metadatas"])
    ids: list[str] = got["ids"]
    docs: list[str] = got["documents"]
    metas: list[dict] = got["metadatas"]
    tokenized = [_tokenize(doc) for doc in docs]
    return BM25Index(
        ids=ids,
        paper_ids=[m["paper_id"] for m in metas],
        built_at_count=collection.count(),
        bm25=BM25Okapi(tokenized, k1=k1, b=b),
    )


def rrf_scores(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """RRF score per id: ``score(d) = Σ w_i / (k + rank_i(d))``, ``rank_i`` 1-indexed within
    ``rankings[i]``; ids absent from a ranking don't contribute a term for it.

    ``weights`` defaults to uniform (``1.0`` per ranking) and is **reserved, not wired** by any
    caller in this codebase yet — a future dense/sparse weight is a one-argument change at each
    call site instead of a signature change plus call-site changes.

    The single source of truth for the RRF formula — both :func:`reciprocal_rank_fusion` and
    ``Searcher.search`` (which needs the actual score values, not just fused order, to populate
    ``Result.score``) build on this rather than each re-deriving the ``k + rank`` sum, so a
    future ``weights`` wiring can't update the fused order without also updating the displayed
    score.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (k + rank)
    return scores


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[str]:
    """Fuse ranked id lists via RRF, returning ids sorted by fused score descending."""
    scores = rrf_scores(rankings, k, weights)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)
