"""Single-config run: drive an eval split through ``Searcher`` and report the two metrics.

This is flow step 1 — "run the default config, see what you get" — with no sweep. It
composes ``rag`` (``Searcher`` + its embedder/collection/reranker) rather than forking it,
but reaches past ``Searcher.search`` (which hides chunk ids and the pre-rerank pool) for the
per-stage ids the metrics need. That thin retrieval seam is also what the retrieval screen's
caching builds on later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from rag.config import BM25Cfg, Config, LLMRerankerCfg
from rag.llm import build_llm
from rag.reranker import build_reranker
from rag.search import Searcher
from rag.sparse import reciprocal_rank_fusion, rrf_scores

from .checkpoint import CheckpointWriter, resume_units
from .metrics import (
    QueryScore,
    mrr_at_k,
    n_conditioned,
    n_ungoldable,
    relevant_ids,
)
from .queryset import QAItem
from .stats import (
    ceiling_saturation_note,
    cluster_bootstrap,
    mdd,
    resolution_warning,
    success_samples,
)


def _query_score_to_record(score: QueryScore) -> dict[str, Any]:
    return {
        "candidate_ids": score.candidate_ids,
        "ranked": [list(pair) for pair in score.ranked],
        "relevant_ids": sorted(score.relevant_ids),
        "paper_id": score.paper_id,
    }


def _query_score_from_record(qid: str, record: dict[str, Any]) -> QueryScore:
    return QueryScore(
        qid=qid,
        candidate_ids=record["candidate_ids"],
        ranked=[(cid, s) for cid, s in record["ranked"]],
        relevant_ids=set(record["relevant_ids"]),
        paper_id=record["paper_id"],
    )


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
        sparse_enabled=cfg.sparse.enabled,
        bm25_k1=cfg.sparse.k1 if isinstance(cfg.sparse, BM25Cfg) else 1.5,
        bm25_b=cfg.sparse.b if isinstance(cfg.sparse, BM25Cfg) else 0.75,
        rrf_k=cfg.sparse.rrf_k,
        fetch_multiplier=cfg.sparse.fetch_multiplier,
    )


def _retrieve(
    searcher: Searcher, query: str, *, candidates: int, k: int, rerank: bool
) -> tuple[list[str], list[tuple[str, float]]]:
    """Return ``(candidate_ids, ranked_top_k)`` for one query.

    ``candidate_ids`` is the full pre-rerank pool (for stage-1 recall) — dense alone, or
    RRF-fused with BM25 when ``searcher.sparse_enabled`` (``build_searcher`` sets this from
    ``cfg.sparse.enabled``, so ``run``/``confirm`` genuinely reflect a config that has hybrid
    turned on, not just the dense-only reading the pre-hybrid version of this function gave).
    ``ranked`` is the reranked top-k as ``(chunk_id, score)`` (for stage-2 nDCG). Mirrors
    ``Searcher``'s own embed → query → (fuse) → rerank path but keeps the chunk ids ``search``
    throws away.
    """
    embed_query = getattr(searcher.embedder, "embed_query", None)
    qvec = embed_query([query]) if embed_query else searcher.embedder([query])
    fetch_n = candidates * searcher._fetch_multiplier if searcher.sparse_enabled else candidates
    # Chroma's stubs narrow query_embeddings/results more tightly than runtime accepts;
    # the requested include= keys are always present and non-None here (mirrors search.py).
    res = cast(
        dict[str, Any],
        searcher.collection.query(
            query_embeddings=qvec,  # ty: ignore[invalid-argument-type]  # list invariance vs Chroma's Sequence param
            n_results=fetch_n,
            include=["documents", "distances"],
        ),
    )
    dense_ids: list[str] = res["ids"][0]
    if not dense_ids:
        return [], []
    docs_by_id = dict(zip(dense_ids, res["documents"][0], strict=True))

    if searcher.sparse_enabled:
        sparse_ids = searcher.sparse.search(query, n=fetch_n)
        ids = reciprocal_rank_fusion([dense_ids, sparse_ids], k=searcher._rrf_k)[:candidates]
        missing = [cid for cid in ids if cid not in docs_by_id]
        if missing:
            got = cast(dict[str, Any], searcher.collection.get(ids=missing, include=["documents"]))
            docs_by_id.update(zip(got["ids"], got["documents"], strict=True))
    else:
        ids = dense_ids

    if not rerank:
        if searcher.sparse_enabled:
            scores = rrf_scores([dense_ids, sparse_ids], k=searcher._rrf_k)
            ranked = [(i, scores[i]) for i in ids][:k]
        else:
            dists = res["distances"][0]
            ranked = [(i, 1 - d) for i, d in zip(ids, dists, strict=True)][:k]
        return ids, ranked
    docs = [docs_by_id[i] for i in ids]
    scores = searcher.reranker.score(query, docs)
    ranked = sorted(zip(ids, scores, strict=True), key=lambda t: t[1], reverse=True)[:k]
    return ids, ranked


def score_items(
    searcher: Searcher,
    items: list[QAItem],
    *,
    candidates: int,
    k: int,
    rerank: bool,
    desc: str | None = None,
    checkpoint_path: Path | None = None,
) -> list[QueryScore]:
    """Retrieve for every item and pair it with its gold-section relevant set.

    ``desc`` labels a progress bar over ``items`` (one dense query + optional rerank pass
    each — the dominant per-arm cost in a screen/sweep); ``None`` (offline tests) runs silent.

    ``checkpoint_path``, when given, makes the loop resumable: each ``QueryScore`` is
    appended (flushed) to the checkpoint the moment it's computed, so a second call with
    the same path/params skips straight to it instead of re-retrieving. The header folds
    in ``searcher.collection.count()`` as a trip-wire — this reads the persistent index
    directly (unlike an isolated chunking-cell searcher), so a resumed run against an
    index whose chunk count changed since the interrupted run is treated as a different
    run, not silently reused (see docs/harness.md).
    """
    done: dict[str, dict[str, Any]] = {}
    writer: CheckpointWriter | None = None
    if checkpoint_path is not None:
        header = {
            "candidates": candidates,
            "k": k,
            "rerank": rerank,
            "index_count": searcher.collection.count(),
        }
        done = resume_units(checkpoint_path, header)
        writer = CheckpointWriter(checkpoint_path, header)
        if done:
            print(f"  resuming: {len(done)}/{len(items)} queries already scored")

    scores: list[QueryScore] = []
    for i, it in enumerate(tqdm(items, desc=desc, disable=desc is None, leave=False)):
        qid = str(i)
        if qid in done:
            scores.append(_query_score_from_record(qid, done[qid]))
            continue
        cand_ids, ranked = _retrieve(searcher, it.query, candidates=candidates, k=k, rerank=rerank)
        rel = relevant_ids(searcher.collection, it.paper_id, it.section_number, it.section_title)
        score = QueryScore(
            qid=qid, candidate_ids=cand_ids, ranked=ranked, relevant_ids=rel, paper_id=it.paper_id
        )
        scores.append(score)
        if writer is not None:
            writer.append(qid, _query_score_to_record(score))
    if writer is not None:
        writer.finish()
    return scores


def run(
    cfg: Config,
    items: list[QAItem],
    *,
    searcher: Searcher | None = None,
    candidates: int | None = None,
    k: int | None = None,
    rerank: bool | None = None,
    desc: str | None = None,
    checkpoint_path: Path | None = None,
) -> RunReport:
    """Score ``items`` at ``cfg``'s retrieval settings and return the report.

    ``searcher`` is injectable for offline tests; production builds one from ``cfg``.
    ``candidates``/``k``/``rerank`` override ``cfg``'s own values when given — this is what
    lets ``confirm`` score a different config than ``cfg`` carries (e.g. against an isolated,
    re-chunked searcher) while reusing this function's stats/report plumbing. ``desc`` labels
    a progress bar over ``items``; ``None`` (offline tests) runs silent. ``checkpoint_path``
    is threaded straight through to :func:`score_items` — see its docstring.
    """
    searcher = searcher or build_searcher(cfg)
    candidates = cfg.retrieval.candidates if candidates is None else candidates
    k = cfg.retrieval.k if k is None else k
    rerank = cfg.reranker.enabled if rerank is None else rerank
    scores = score_items(
        searcher,
        items,
        candidates=candidates,
        k=k,
        rerank=rerank,
        desc=desc,
        checkpoint_path=checkpoint_path,
    )
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
    saturated = ceiling_saturation_note(report.success_at_candidates)
    if saturated:
        lines.append(f"  ⚠ {saturated}")
    lines += [
        f"  success@candidates = {report.success_at_candidates:.3f}  "
        f"[{lo:.3f}, {hi:.3f}]   "
        f"(stage-1 ceiling: gold section reached the dense pool, over {n_goldable} goldable)",
        f"  MRR@{report.k}             = {report.mrr_at_k:.3f}   "
        f"(stage-2, over {report.n_conditioned}/{n_goldable} gold-in-pool queries)",
    ]
    return "\n".join(lines)
