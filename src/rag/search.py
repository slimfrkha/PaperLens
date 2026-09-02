"""Retrieval over the Chroma RAG DB, with optional cross-encoder reranking and hybrid BM25 fusion.

Three-stage retrieval:
  1. Dense vector search in Chroma returns the top ``candidates`` chunks. When hybrid sparse
     retrieval is enabled, a BM25 lexical search runs alongside it and the two rankings are
     fused via reciprocal rank fusion (RRF) before reranking. When multi-query expansion is
     also enabled, the chat LLM paraphrases the query and every variant's dense (+ sparse)
     ranking is fused into that same flat RRF pass — a recall boost against how a question
     happens to be phrased, opt-in like hybrid (see ``MultiQueryCfg`` in ``config.py``).
  2. A cross-encoder reranker (default BAAI/bge-reranker-v2-m3, the sibling of
     the bge-m3 embedder) rescores each (query, chunk) pair. Reranking reorders
     keyword-y matches below the truly relevant ones; disable it with ``--no-rerank``
     for pure vector results.
  3. An elbow cutoff (``find_cutoff``) picks how many of the reranked results to keep:
     the first real drop-off in score, bounded to ``[min_k, max_k]`` — not a fixed count.
     Only runs when stage 2 actually reranked; ``--no-rerank`` or a reranker failure both
     fall back to plain ``max_k`` truncation.

CLI:
    python -m rag.search "how does MLA reduce the KV cache?"
    python -m rag.search "FP8 quantization" --min-k 2 --max-k 8 --candidates 30 --paper <paper_id>
    python -m rag.search "long context" --no-rerank
    python -m rag.search "SwiGLU activation" --sparse
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Literal, cast

from .config import AnthropicSpec
from .embedders import HFEmbedder
from .llm import LLMBackend, build_llm
from .query_expansion import generate_paraphrases
from .reranker import CrossEncoderReranker, Reranker
from .sparse import BM25Index, build_sparse_index, reciprocal_rank_fusion, rrf_scores

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass
class Result:
    score: float  # rerank score if reranked; RRF score if hybrid+no-rerank; else cosine similarity
    paper_id: str
    breadcrumb: str
    section_title: str
    section_number: str
    text: str  # breadcrumb + body (what was embedded)
    body: str  # body only
    source: str = "dense"  # "dense" | "sparse" | "both" — which retrieval pool(s) surfaced this


CutoffReason = Literal["elbow", "pool_exhausted", "no_elbow", "no_rerank", "disabled"]


@dataclass
class SearchOutcome:
    """``Searcher.search``'s return value: the results plus *why* that many came back —
    needed because the count is no longer a fixed ``k`` a caller chose, it's whatever the
    elbow cutoff (or a fallback) decided. ``cutoff_reason``:
      - ``"elbow"``          — a real score cliff was found; cut there.
      - ``"no_elbow"``       — reranked fine, but nothing looked like a cliff; returned
                                ``max_k`` (or the whole pool, if smaller).
      - ``"pool_exhausted"`` — fewer than ``min_k`` candidates existed at all; returned
                                every one of them.
      - ``"no_rerank"``      — ``rerank=False`` or the reranker failed; plain ``max_k``
                                truncation of the pre-rerank order, elbow never attempted.
      - ``"disabled"``       — ``elbow_enabled=False``; plain ``max_k`` truncation of the
                                reranked order, elbow never attempted.
    """

    results: list[Result]
    cutoff_reason: CutoffReason


@dataclass(frozen=True)
class RecallSelection:
    """One materialized candidate-id ordering from a raw recall snapshot."""

    candidate_ids: tuple[str, ...]
    scores: dict[str, float]
    dense_ids: frozenset[str]
    sparse_ids: frozenset[str]


@dataclass(frozen=True)
class RecallPoolSnapshot:
    """Raw ranked source lists captured once and materialized many times.

    This is intentionally the seam the eval cache crosses: callers can select a candidate
    depth without knowing RRF, over-fetching, or which variant rankings need fusing.
    It contains ids only; passage hydration and per-materialization provenance remain in
    ``Searcher`` so a hybrid arm cannot leak ``Result.source`` into a dense-only arm.
    """

    dense_ids: tuple[str, ...]
    sparse_ids: tuple[str, ...] = ()
    variant_rankings: tuple[tuple[str, ...], ...] = ()
    variant_sources: tuple[str, ...] = ()

    def materialize(
        self,
        candidates: int,
        *,
        sparse: bool = False,
        multi_query: bool = False,
        rrf_k: int = 60,
        fetch_multiplier: int = 3,
        multi_query_fetch_multiplier: int = 3,
    ) -> RecallSelection:
        """Select one candidate pool from this snapshot without another retrieval."""
        if multi_query:
            fetch_n = candidates * multi_query_fetch_multiplier
            rankings = [list(self.dense_ids[:fetch_n])] + [
                list(ids[:fetch_n]) for ids in self.variant_rankings
            ]
            candidate_ids = reciprocal_rank_fusion(rankings, k=rrf_k)[:candidates]
            scores = rrf_scores(rankings, k=rrf_k)
            dense_ids = set(self.dense_ids[:fetch_n])
            sparse_ids: set[str] = set(self.sparse_ids[:fetch_n])
            if self.sparse_ids:
                rankings.insert(1, list(self.sparse_ids[:fetch_n]))
                candidate_ids = reciprocal_rank_fusion(rankings, k=rrf_k)[:candidates]
                scores = rrf_scores(rankings, k=rrf_k)
            for ids, source in zip(self.variant_rankings, self.variant_sources, strict=True):
                if source == "dense":
                    dense_ids.update(ids[:fetch_n])
                else:
                    sparse_ids.update(ids[:fetch_n])
            return RecallSelection(
                tuple(candidate_ids), scores, frozenset(dense_ids), frozenset(sparse_ids)
            )

        if sparse:
            fetch_n = candidates * fetch_multiplier
            dense_ids = list(self.dense_ids[:fetch_n])
            sparse_ids = list(self.sparse_ids[:fetch_n])
            rankings = [dense_ids, sparse_ids]
            candidate_ids = reciprocal_rank_fusion(rankings, k=rrf_k)[:candidates]
            return RecallSelection(
                tuple(candidate_ids),
                rrf_scores(rankings, k=rrf_k),
                frozenset(dense_ids),
                frozenset(sparse_ids),
            )

        candidate_ids = tuple(self.dense_ids[:candidates])
        return RecallSelection(candidate_ids, {}, frozenset(candidate_ids), frozenset())

    def to_record(self) -> dict[str, list]:
        return {
            "dense_ids": list(self.dense_ids),
            "sparse_ids": list(self.sparse_ids),
            "variant_rankings": [list(ids) for ids in self.variant_rankings],
            "variant_sources": list(self.variant_sources),
        }

    @classmethod
    def from_record(cls, record: dict[str, list]) -> RecallPoolSnapshot:
        return cls(
            dense_ids=tuple(record["dense_ids"]),
            sparse_ids=tuple(record.get("sparse_ids", [])),
            variant_rankings=tuple(tuple(ids) for ids in record.get("variant_rankings", [])),
            variant_sources=tuple(record.get("variant_sources", [])),
        )


@dataclass(frozen=True)
class RecallEntry:
    """One candidate's stable chunk id and fully hydrated passage."""

    chunk_id: str
    result: Result


@dataclass(frozen=True)
class RecallPool:
    """The fully hydrated, pre-rerank candidate pool returned to product and eval callers."""

    entries: tuple[RecallEntry, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(entry.chunk_id for entry in self.entries)

    @property
    def results(self) -> tuple[Result, ...]:
        return tuple(entry.result for entry in self.entries)


@dataclass
class CapturedRecall:
    """A cache-ready raw pool plus the hydrated passages needed to score it once."""

    snapshot: RecallPoolSnapshot
    results_by_id: dict[str, Result]


def find_cutoff(
    scores: list[float],
    min_k: int,
    max_k: int,
    mad_multiplier: float,
    prominence: float,
) -> tuple[int, CutoffReason]:
    """Where to cut a reranked, score-**descending** list: the first real drop-off (an
    "elbow") within ``[min_k, max_k]``, or ``max_k`` if nothing looks like one.

    A gap counts as a real cliff only if it clears two tests against the *other* gaps in
    the window — both robust to the candidate cliff itself, so it can't inflate its own
    baseline:
      - a MAD-based outlier test (``median + mad_multiplier * MAD`` of the other gaps),
        with MAD floored at a small fraction of the window's score range — several
        equal/near-equal low-ranked gaps (a reranker clustering similar-relevance chunks)
        would otherwise collapse MAD toward 0 and make any nonzero gap read as a
        statistically significant cliff, exactly the smooth-decay case this test exists
        to reject;
      - a prominence floor relative to the window's own score range, so a technically
        "significant" but tiny wobble on an already-flat curve doesn't count.

    Returns ``(cutoff_index, reason)`` — see ``SearchOutcome.cutoff_reason`` for what each
    reason means. Never returns fewer than ``min(min_k, len(scores))`` or more than
    ``min(max_k, len(scores))``.
    """
    if len(scores) <= min_k:
        return len(scores), "pool_exhausted"

    window = scores[:max_k]
    if len(window) <= 1:
        # Nothing to find a gap between (max_k == 1, or a single-candidate pool already
        # past the min_k check above) — no elbow is even a meaningful question here.
        return len(window), "no_elbow"

    gaps = [window[i] - window[i + 1] for i in range(len(window) - 1)]
    score_range = window[0] - window[-1]

    max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    max_gap = gaps[max_gap_idx]
    other_gaps = gaps[:max_gap_idx] + gaps[max_gap_idx + 1 :]

    # No other gaps to build a baseline from (max_k == 2), or a flat window (score_range
    # == 0) — nothing to robustly test against, so no elbow rather than a noisy guess.
    if other_gaps and score_range > 0:
        baseline = median(other_gaps)
        mad = median([abs(g - baseline) for g in other_gaps])
        mad = max(mad, 1e-3 * score_range)  # floor: guards the near-zero-MAD false trigger
        is_real_cliff = max_gap > baseline + mad_multiplier * mad
        clears_prominence = max_gap > prominence * score_range
        if is_real_cliff and clears_prominence:
            cut = max_gap_idx + 1
            return max(min_k, min(cut, max_k)), "elbow"

    return min(max_k, len(scores)), "no_elbow"


def _pick_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


class Searcher:
    def __init__(
        self,
        db_dir: str = "rag_db",
        collection: str = "arxiv_papers",
        embedder_model: str = DEFAULT_EMBEDDER,
        reranker_model: str = DEFAULT_RERANKER,
        device: str | None = None,
        embedder=None,
        query_prefix: str = "",
        reranker: Reranker | None = None,
        sparse_enabled: bool = False,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        fetch_multiplier: int = 3,
        sparse: BM25Index | None = None,
        multi_query_enabled: bool = False,
        multi_query_n: int = 3,
        multi_query_fetch_multiplier: int = 3,
        llm: LLMBackend | None = None,
    ):
        import chromadb

        # Multi-query is opt-in with no sane default LLM (unlike reranker/sparse, which
        # always have a default backend) — require one explicitly rather than lazily
        # building one, matching build_reranker's check that the `llm` reranker needs an
        # LLM passed.
        if multi_query_enabled and llm is None:
            raise ValueError("multi_query_enabled requires an LLM; pass Searcher(..., llm=...).")

        self.device = _pick_device(device)
        # Query embedder: inject one to avoid loading the local model (tests); the
        # default builds the same HF model used at indexing time. query_prefix only
        # applies to this default-build path — an injected embedder already carries
        # its own prefixes (see build_embedder).
        self.embedder = embedder or HFEmbedder(
            embedder_model, device=self.device, query_prefix=query_prefix
        )
        self.collection = chromadb.PersistentClient(path=db_dir).get_collection(collection)
        self._reranker_model = reranker_model
        # Inject a Reranker (e.g. the config-built llm reranker) or leave None to
        # lazily build the default hf cross-encoder on first use.
        self._reranker = reranker
        self.sparse_enabled = sparse_enabled
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._rrf_k = rrf_k
        self._fetch_multiplier = fetch_multiplier
        # Inject a BM25Index for tests, or leave None to lazily build/rebuild on first use.
        self._sparse = sparse
        self.multi_query_enabled = multi_query_enabled
        self.multi_query_n = multi_query_n
        self.multi_query_fetch_multiplier = multi_query_fetch_multiplier
        self.llm = llm

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self._reranker_model, device=self.device)
        return self._reranker

    @property
    def sparse(self) -> BM25Index:
        """The BM25 index, rebuilt whenever the collection has grown since it was built.

        A BM25 index is a snapshot (its IDF needs global corpus stats), unlike the stateless
        reranker — so this checks staleness on every access rather than caching forever. That's
        what makes a live `/api/admin/rescan` (server/main.py) safe: dense search already
        queries Chroma live, and this keeps the sparse side from silently falling behind it. The
        extra cost when sparse is on is one cheap `collection.count()` call per search — a
        metadata lookup, not a corpus scan — with a full rebuild only right after a rescan
        changed the count.
        """
        if self._sparse is None or self._sparse.built_at_count != self.collection.count():
            self._sparse = build_sparse_index(self.collection, k1=self._bm25_k1, b=self._bm25_b)
        return self._sparse

    def _embed_query(self, query: str) -> list[list[float]]:
        """Embeds one query string for the recall pool's precomputed variant vectors.

        Asymmetric embedders (e.g. Gemini) embed queries differently from docs; symmetric
        ones fall through to __call__.
        """
        embed_query = getattr(self.embedder, "embed_query", None)
        return embed_query([query]) if embed_query else self.embedder([query])

    def _dense_recall(
        self,
        query: str,
        fetch_n: int,
        where: dict[str, Any] | None,
        qvec: list[list[float]] | None = None,
    ) -> tuple[list[str], dict[str, Result]]:
        """One query's dense ranking: chunk ids in Chroma's distance order, plus their Results.

        An implementation detail of ``capture_recall_snapshot``. It returns ids and hydrated
        passages so the enclosing recall module can fuse and materialize the pool.

        ``qvec`` lets a caller pass an already-computed embedding for ``query`` — ``search``
        precomputes one per variant and reuses it across every scope (e.g. every paper under
        per-paper retrieval) instead of re-embedding identical query text per scope. Omit it
        (the default) to embed ``query`` here, as before.
        """
        qvec = qvec if qvec is not None else self._embed_query(query)
        # Chroma's type stubs narrow query_embeddings/results more tightly than
        # runtime accepts; the include= keys are always present and non-None here.
        res = cast(
            dict[str, Any],
            self.collection.query(
                query_embeddings=qvec,  # ty: ignore[invalid-argument-type]  # list invariance vs Chroma's Sequence param
                n_results=fetch_n,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )
        dense_ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        by_id = {
            cid: Result(
                score=1 - d,  # cosine distance -> similarity
                paper_id=m["paper_id"],
                breadcrumb=m["breadcrumb"],
                section_title=m["section_title"],
                section_number=m["section_number"],
                text=doc,
                body=m["body"],
            )
            for cid, doc, m, d in zip(dense_ids, docs, metas, dists, strict=True)
        }
        return dense_ids, by_id

    def _backfill_missing(self, by_id: dict[str, Result], candidate_ids: list[str]) -> None:
        """Fetch any candidate id ``dense_recall`` didn't already return — reached via BM25
        (hybrid retrieval) and add it with a placeholder score. The enclosing recall module
        replaces that score with an RRF score whenever the candidate pool is materialized; only
        paper_id/breadcrumb/section/text/body need to be real during capture.
        """
        missing = [cid for cid in candidate_ids if cid not in by_id]
        if not missing:
            return
        got = cast(
            dict[str, Any],
            self.collection.get(ids=missing, include=["documents", "metadatas"]),
        )
        for cid, doc, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True):
            by_id[cid] = Result(
                score=0.0,
                paper_id=m["paper_id"],
                breadcrumb=m["breadcrumb"],
                section_title=m["section_title"],
                section_number=m["section_number"],
                text=doc,
                body=m["body"],
            )

    def capture_recall_snapshot(
        self,
        query: str,
        max_candidates: int,
        *,
        where: dict[str, Any] | None = None,
        allowed_paper_ids: set[str] | None = None,
        sparse: BM25Index | None = None,
        fetch_multiplier: int = 3,
        variants: list[str] | None = None,
        multi_query_fetch_multiplier: int = 3,
        base_fetch_multiplier: int | None = None,
        qvecs: dict[str, list[list[float]]] | None = None,
    ) -> CapturedRecall:
        """Capture raw rankings once for the eval cache or one product recall scope.

        Callers only choose the corpus scope and which already-created query variants to
        capture. Dense lookup, sparse lookup, hydration, and the shape of the raw rankings
        stay behind this seam. ``RecallPoolSnapshot.materialize`` owns all later fusion.
        """
        variants = variants or []
        multi_query = bool(variants)
        sparse_enabled = sparse is not None
        fetch_n = max_candidates * (
            base_fetch_multiplier
            if base_fetch_multiplier is not None
            else multi_query_fetch_multiplier
            if multi_query
            else fetch_multiplier
            if sparse_enabled
            else 1
        )
        variant_fetch_n = max_candidates * multi_query_fetch_multiplier
        qvec = qvecs.get(query) if qvecs is not None else None
        dense_ids, by_id = self._dense_recall(query, fetch_n, where, qvec=qvec)
        sparse_ids = (
            sparse.search(query, n=fetch_n, allowed_ids=allowed_paper_ids)
            if sparse is not None
            else []
        )
        self._backfill_missing(by_id, sparse_ids)

        variant_rankings: list[tuple[str, ...]] = []
        variant_sources: list[str] = []
        for variant in variants:
            qvec = qvecs.get(variant) if qvecs is not None else None
            variant_dense_ids, variant_by_id = self._dense_recall(
                variant, variant_fetch_n, where, qvec=qvec
            )
            for cid, result in variant_by_id.items():
                by_id.setdefault(cid, result)
            variant_rankings.append(tuple(variant_dense_ids))
            variant_sources.append("dense")
            if sparse is not None:
                variant_sparse_ids = sparse.search(
                    variant, n=variant_fetch_n, allowed_ids=allowed_paper_ids
                )
                self._backfill_missing(by_id, variant_sparse_ids)
                variant_rankings.append(tuple(variant_sparse_ids))
                variant_sources.append("sparse")

        return CapturedRecall(
            RecallPoolSnapshot(
                dense_ids=tuple(dense_ids),
                sparse_ids=tuple(sparse_ids),
                variant_rankings=tuple(variant_rankings),
                variant_sources=tuple(variant_sources),
            ),
            by_id,
        )

    def recall(
        self,
        query: str,
        *,
        candidates: int,
        max_k: int,
        paper: str | None = None,
        paper_ids: list[str] | None = None,
        per_paper: bool = False,
        per_paper_min_candidates: int | None = None,
    ) -> RecallPool:
        """Build the fully hydrated pre-rerank pool for product and evaluation callers."""
        ids = paper_ids
        if paper is not None:
            ids = [paper] if ids is None else [pid for pid in ids if pid == paper]
        if ids is not None and not ids:
            return RecallPool(())
        if per_paper and ids is None:
            raise ValueError(
                "per_paper=True requires a resolved paper/paper_ids filter — Searcher has no "
                "manifest to fall back to every paper (ChatAgent.run does that fallback "
                "for you)."
            )

        paraphrases: list[str] = []
        if self.multi_query_enabled:
            assert self.llm is not None  # enforced at construction
            paraphrases = generate_paraphrases(query, self.llm, n=self.multi_query_n)
        variants = [query, *paraphrases]
        qvecs = {variant: self._embed_query(variant) for variant in variants}

        where = {"paper_id": {"$in": ids}} if ids is not None else None
        allowed = set(ids) if ids is not None else None
        if per_paper:
            assert ids is not None
            minimum = max_k if per_paper_min_candidates is None else per_paper_min_candidates
            scopes = [
                (
                    {"paper_id": {"$in": [pid]}},
                    {pid},
                    max(minimum, min(candidates, candidates // len(ids))),
                )
                for pid in ids
            ]
        else:
            scopes = [(where, allowed, candidates)]

        entries: list[RecallEntry] = []
        for scope_where, scope_allowed, scope_candidates in scopes:
            captured = self.capture_recall_snapshot(
                query,
                scope_candidates,
                where=scope_where,
                allowed_paper_ids=scope_allowed,
                sparse=self.sparse if self.sparse_enabled else None,
                fetch_multiplier=self._fetch_multiplier,
                variants=paraphrases,
                multi_query_fetch_multiplier=self.multi_query_fetch_multiplier,
                qvecs=qvecs,
            )
            selection = captured.snapshot.materialize(
                scope_candidates,
                sparse=self.sparse_enabled,
                multi_query=bool(paraphrases),
                rrf_k=self._rrf_k,
                fetch_multiplier=self._fetch_multiplier,
                multi_query_fetch_multiplier=self.multi_query_fetch_multiplier,
            )
            for cid in selection.candidate_ids:
                result = captured.results_by_id[cid]
                in_dense, in_sparse = cid in selection.dense_ids, cid in selection.sparse_ids
                source = "both" if in_dense and in_sparse else "sparse" if in_sparse else "dense"
                entries.append(
                    RecallEntry(
                        cid,
                        replace(
                            result, score=selection.scores.get(cid, result.score), source=source
                        ),
                    )
                )

        if per_paper:
            entries.sort(key=lambda entry: entry.result.score, reverse=True)
        return RecallPool(tuple(entries))

    def search(
        self,
        query: str,
        min_k: int = 2,
        max_k: int = 10,
        candidates: int = 20,
        paper: str | None = None,
        paper_ids: list[str] | None = None,
        rerank: bool = True,
        per_paper: bool = False,
        elbow_enabled: bool = True,
        elbow_mad_multiplier: float = 3.0,
        elbow_prominence: float = 0.15,
    ) -> SearchOutcome:
        if min_k > max_k:
            # RetrievalCfg.__post_init__ already enforces this for the agent path, but the
            # CLI (--min-k/--max-k) and any direct library caller build no Config at all —
            # without this, find_cutoff's max(min_k, min(cut, max_k)) bounding silently
            # returns more than max_k instead of failing loudly.
            raise ValueError(f"min_k ({min_k}) must be <= max_k ({max_k})")

        pool = self.recall(
            query,
            candidates=candidates,
            max_k=max_k,
            paper=paper,
            paper_ids=paper_ids,
            per_paper=per_paper,
        )
        results = [entry.result for entry in pool.entries]
        if not results:
            return SearchOutcome([], "pool_exhausted")

        reranked_ok = False
        if rerank:
            try:
                scores = self.reranker.score(query, [r.text for r in results])
                if len(scores) != len(results):
                    raise ValueError(
                        f"reranker returned {len(scores)} scores for {len(results)} docs"
                    )
                # Lengths are already checked equal above; strict=False since a mismatch is
                # handled there, not here.
                results = [
                    replace(result, score=score)
                    for result, score in zip(results, scores, strict=False)
                ]
                results.sort(key=lambda result: result.score, reverse=True)
                reranked_ok = True
            except Exception as e:
                # Reranker failure (model load, inference error, network/rate-limit/auth from an
                # LLM reranker, or a malformed score list) — degrade to the pre-rerank (dense/RRF)
                # order rather than failing the whole search.
                print(
                    f"  [warn] reranking failed for query {query!r},"
                    f" falling back to pre-rerank order: {e}"
                )

        # Elbow cutoff only runs on a genuinely reranked, score-comparable ordering — a
        # skipped or failed rerank leaves cosine/RRF scores, which don't have the same
        # "confident cliff" shape a cross-encoder produces, so both fall back to plain
        # max_k truncation instead (see SearchOutcome.cutoff_reason docstring).
        if not reranked_ok:
            return SearchOutcome(results[:max_k], "no_rerank")
        if not elbow_enabled:
            return SearchOutcome(results[:max_k], "disabled")

        cutoff, reason = find_cutoff(
            [r.score for r in results], min_k, max_k, elbow_mad_multiplier, elbow_prominence
        )
        return SearchOutcome(results[:cutoff], reason)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("query")
    p.add_argument("--db-dir", default="rag_db")
    p.add_argument("--collection", default="arxiv_papers")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--query-prefix", default="", help="prepended to the query before embedding")
    p.add_argument("--reranker", default=DEFAULT_RERANKER)
    p.add_argument("--min-k", type=int, default=2, help="never return fewer than this")
    p.add_argument("--max-k", type=int, default=10, help="never return more than this")
    p.add_argument("--candidates", type=int, default=20, help="vector hits to rerank")
    p.add_argument("--paper", default=None, help="restrict to one paper_id")
    p.add_argument("--no-rerank", action="store_true", help="skip cross-encoder rerank")
    p.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fuse in BM25 lexical search (hybrid); --no-sparse is dense-only (default)",
    )
    p.add_argument(
        "--multi-query",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="paraphrase the query with the chat LLM and RRF-fuse each variant's results",
    )
    p.add_argument(
        "--multi-query-n", type=int, default=3, help="paraphrases to generate (--multi-query only)"
    )
    args = p.parse_args()

    searcher = Searcher(
        db_dir=args.db_dir,
        collection=args.collection,
        embedder_model=args.embedder,
        query_prefix=args.query_prefix,
        reranker_model=args.reranker,
        sparse_enabled=args.sparse,
        multi_query_enabled=args.multi_query,
        multi_query_n=args.multi_query_n,
        llm=build_llm(AnthropicSpec()) if args.multi_query else None,
    )
    outcome = searcher.search(
        args.query,
        min_k=args.min_k,
        max_k=args.max_k,
        candidates=args.candidates,
        paper=args.paper,
        rerank=not args.no_rerank,
    )

    mode = "hybrid" if searcher.sparse_enabled else "vector"
    mode = f"{mode}+multi-query" if searcher.multi_query_enabled else mode
    tag = mode if args.no_rerank else f"{mode}+rerank"
    print(
        f"\nQ: {args.query}   [{tag}, {outcome.cutoff_reason}, "
        f"{len(outcome.results)} of {args.min_k}-{args.max_k}, {args.candidates} candidates]"
    )
    for r in outcome.results:
        crumb = r.breadcrumb.split(" > ", 1)[-1]
        snippet = textwrap.shorten(r.body.replace("\n", " "), width=200)
        print(f"\n  [{r.score:.3f}] {r.paper_id}  ::  {crumb}")
        print(f"          {snippet}")


if __name__ == "__main__":
    main()
