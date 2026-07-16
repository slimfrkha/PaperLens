"""Single-config run: drive an eval split through ``Searcher`` and report the two metrics.

This is flow step 1 — "run the default config, see what you get" — with no sweep. It
composes ``rag`` (``Searcher`` + its embedder/collection/reranker) rather than forking it,
but reaches past ``Searcher.search`` (which hides chunk ids and the pre-rerank pool) for the
per-stage ids the metrics need. That thin retrieval seam is also what Tier-A caching builds
on later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from rag.config import Config, LLMRerankerCfg
from rag.llm import build_llm
from rag.reranker import build_reranker
from rag.search import Searcher

from .metrics import (
    QueryScore,
    mrr_at_k,
    n_conditioned,
    n_ungoldable,
    relevant_ids,
)
from .queryset import QAItem
from .stats import cluster_bootstrap, mdd, resolution_warning, success_samples


@dataclass
class RunReport:
    n_queries: int
    candidates: int
    k: int
    rerank: bool
    success_at_candidates: float  # stage-1 ceiling: gold section reached the dense pool
    mrr_at_k: float  # stage-2 quality, conditioned on gold-in-pool
    n_conditioned: int  # queries the MRR@k average is over
    n_ungoldable: int  # queries whose gold section is absent from the index (excluded)
    # Resolution of the loaded pool (paper-clustered bootstrap on the ceiling). n_clusters is
    # the effective sample; mdd_upfront is the conservative single-arm MDD — a real paired
    # sweep resolves finer, since paired arms are correlated. Leads the report.
    n_clusters: int
    ceiling_ci: tuple[float, float]
    mdd_upfront: float


def build_searcher(cfg: Config) -> Searcher:
    """A ``Searcher`` over the config's on-disk collection (loads the real embedder)."""
    llm = build_llm(cfg.llm.chat) if isinstance(cfg.reranker, LLMRerankerCfg) else None
    reranker = build_reranker(cfg.reranker, llm=llm) if cfg.reranker.enabled else None
    return Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder_model=cfg.embedding.model,
        reranker=reranker,
    )


def _retrieve(
    searcher: Searcher, query: str, *, candidates: int, k: int, rerank: bool
) -> tuple[list[str], list[tuple[str, float]]]:
    """Return ``(candidate_ids, ranked_top_k)`` for one query.

    ``candidate_ids`` is the full pre-rerank dense pool (for stage-1 recall); ``ranked`` is
    the reranked top-k as ``(chunk_id, score)`` (for stage-2 nDCG). Mirrors ``Searcher``'s
    own embed → query → rerank path but keeps the chunk ids ``search`` throws away.
    """
    embed_query = getattr(searcher.embedder, "embed_query", None)
    qvec = embed_query([query]) if embed_query else searcher.embedder([query])
    # Chroma's stubs narrow query_embeddings/results more tightly than runtime accepts;
    # the requested include= keys are always present and non-None here (mirrors search.py).
    res = cast(
        dict[str, Any],
        searcher.collection.query(
            query_embeddings=qvec,  # ty: ignore[invalid-argument-type]  # list invariance vs Chroma's Sequence param
            n_results=candidates,
            include=["documents", "distances"],
        ),
    )
    ids: list[str] = res["ids"][0]
    if not ids:
        return [], []
    if not rerank:
        dists = res["distances"][0]
        ranked = [(i, 1 - d) for i, d in zip(ids, dists, strict=True)][:k]
        return ids, ranked
    docs = res["documents"][0]
    scores = searcher.reranker.score(query, docs)
    ranked = sorted(zip(ids, scores, strict=True), key=lambda t: t[1], reverse=True)[:k]
    return ids, ranked


def score_items(
    searcher: Searcher, items: list[QAItem], *, candidates: int, k: int, rerank: bool
) -> list[QueryScore]:
    """Retrieve for every item and pair it with its gold-section relevant set."""
    scores: list[QueryScore] = []
    for i, it in enumerate(items):
        cand_ids, ranked = _retrieve(searcher, it.query, candidates=candidates, k=k, rerank=rerank)
        rel = relevant_ids(searcher.collection, it.paper_id, it.section_number, it.section_title)
        scores.append(
            QueryScore(
                qid=str(i),
                candidate_ids=cand_ids,
                ranked=ranked,
                relevant_ids=rel,
                paper_id=it.paper_id,
            )
        )
    return scores


def run(cfg: Config, items: list[QAItem], *, searcher: Searcher | None = None) -> RunReport:
    """Score ``items`` at ``cfg``'s retrieval settings and return the report.

    ``searcher`` is injectable for offline tests; production builds one from ``cfg``.
    """
    searcher = searcher or build_searcher(cfg)
    candidates = cfg.retrieval.candidates
    k = cfg.retrieval.k
    rerank = cfg.reranker.enabled
    scores = score_items(searcher, items, candidates=candidates, k=k, rerank=rerank)
    boot = cluster_bootstrap(success_samples(scores))
    # boot.point is the same quantity as success_at_candidates(scores) — read it off the
    # bootstrap rather than recomputing, so the point and its CI can never disagree.
    return RunReport(
        n_queries=len(items),
        candidates=candidates,
        k=k,
        rerank=rerank,
        success_at_candidates=boot.point,
        mrr_at_k=mrr_at_k(scores, k),
        n_conditioned=n_conditioned(scores),
        n_ungoldable=n_ungoldable(scores),
        n_clusters=boot.n_clusters,
        ceiling_ci=(boot.ci_lo, boot.ci_hi),
        mdd_upfront=mdd(boot.se),
    )


def format_report(report: RunReport) -> str:
    """Human-readable summary. Leads with resolution — ceiling+CI, MDD, n_clusters — before
    any metric, per the plan: a number is only meaningful once you know what this pool can
    resolve.
    """
    rr = "on" if report.rerank else "off"
    n_goldable = report.n_queries - report.n_ungoldable
    lo, hi = report.ceiling_ci
    lines = [
        f"n_queries={report.n_queries}  goldable={n_goldable}  "
        f"ungoldable={report.n_ungoldable}  candidates={report.candidates}  k={report.k}  "
        f"rerank={rr}",
        f"  resolution: n_clusters={report.n_clusters} papers  "
        f"MDD≈{report.mdd_upfront * 100:.1f} pts on the ceiling "
        f"@95%/80%-power (up-front, single-arm)",
    ]
    warning = resolution_warning(report.n_clusters)
    if warning:
        lines.append(f"  ⚠ {warning}")
    lines += [
        f"  success@candidates = {report.success_at_candidates:.3f}  "
        f"[{lo:.3f}, {hi:.3f}]   "
        f"(stage-1 ceiling: gold section reached the dense pool, over {n_goldable} goldable)",
        f"  MRR@{report.k}             = {report.mrr_at_k:.3f}   "
        f"(stage-2, over {report.n_conditioned}/{n_goldable} gold-in-pool queries)",
    ]
    return "\n".join(lines)
