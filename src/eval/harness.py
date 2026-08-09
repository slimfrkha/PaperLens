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
from typing import Any

from tqdm import tqdm

from rag.config import BM25Cfg, Config, HFEmbeddingCfg, LLMRerankerCfg
from rag.llm import build_llm
from rag.reranker import build_reranker
from rag.search import Searcher
from rag.sparse import reciprocal_rank_fusion, rrf_scores

from .checkpoint import CheckpointWriter, resume_units
from .metrics import (
    QueryScore,
    elbow_cutoffs,
    mean_returned_at_elbow,
    mrr_at_k,
    n_conditioned,
    n_ungoldable,
    precision_at_elbow,
    recall_at_elbow,
    relevant_ids,
)
from .queryset import QAItem
from .stats import (
    BootResult,
    DeltaResult,
    ceiling_saturation_note,
    cluster_bootstrap,
    elbow_recall_samples,
    mdd,
    paired_delta,
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
    # Additive elbow-cutoff metrics (see eval.metrics) — None when rerank=False, since
    # elbow never runs on a non-reranked ordering. Never disturb the metrics above.
    mean_returned_at_elbow: float | None = None
    precision_at_elbow: float | None = None
    recall_at_elbow: float | None = None


def build_searcher(cfg: Config) -> Searcher:
    """A ``Searcher`` over the config's on-disk collection (loads the real embedder)."""
    llm = build_llm(cfg.llm.chat) if isinstance(cfg.reranker, LLMRerankerCfg) else None
    reranker = build_reranker(cfg.reranker, llm=llm) if cfg.reranker.enabled else None
    return Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder_model=cfg.embedding.model,
        query_prefix=cfg.embedding.query_prefix
        if isinstance(cfg.embedding, HFEmbeddingCfg)
        else "",
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
    ``ranked`` is the reranked top-k as ``(chunk_id, score)`` (for stage-2 nDCG). Composes
    ``Searcher.dense_recall``/``backfill_missing`` directly — the same primitive ``search``
    uses internally — for the chunk ids ``search``'s own return value throws away.
    """
    fetch_n = candidates * searcher._fetch_multiplier if searcher.sparse_enabled else candidates
    dense_ids, by_id = searcher.dense_recall(query, fetch_n, where=None)
    if not dense_ids:
        return [], []

    if searcher.sparse_enabled:
        sparse_ids = searcher.sparse.search(query, n=fetch_n)
        ids = reciprocal_rank_fusion([dense_ids, sparse_ids], k=searcher._rrf_k)[:candidates]
        searcher.backfill_missing(by_id, ids)
    else:
        ids = dense_ids

    if not rerank:
        if searcher.sparse_enabled:
            scores = rrf_scores([dense_ids, sparse_ids], k=searcher._rrf_k)
            ranked = [(i, scores[i]) for i in ids][:k]
        else:
            ranked = [(i, by_id[i].score) for i in ids][:k]
        return ids, ranked
    docs = [by_id[i].text for i in ids]
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
    k = cfg.retrieval.max_k if k is None else k
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
    elbow_mean = elbow_precision = elbow_recall = None
    if rerank:
        cutoffs = elbow_cutoffs(
            scores,
            cfg.retrieval.min_k,
            cfg.retrieval.max_k,
            cfg.retrieval.elbow_mad_multiplier,
            cfg.retrieval.elbow_prominence,
        )
        elbow_mean = mean_returned_at_elbow(cutoffs)
        elbow_precision = precision_at_elbow(cutoffs)
        elbow_recall = recall_at_elbow(cutoffs)
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
        mean_returned_at_elbow=elbow_mean,
        precision_at_elbow=elbow_precision,
        recall_at_elbow=elbow_recall,
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
    if report.mean_returned_at_elbow is not None:
        lines.append(
            f"  elbow: mean_returned={report.mean_returned_at_elbow:.1f}  "
            f"precision@elbow={report.precision_at_elbow:.3f}  "
            f"recall@elbow={report.recall_at_elbow:.3f}   "
            f"(additive — never affects the metrics above)"
        )
    return "\n".join(lines)


# --- Elbow-knob screen: min_k/max_k stay out of this (docs/harness.md — "how much context
# the agent gets" is a product decision, not chosen by the tool, same as retrieval.k always
# was); mad_multiplier/prominence are pure retrieval-quality knobs, screened the same way
# reranker.enabled/candidates already are.

DEFAULT_MAD_GRID = [1.5, 2.0, 3.0, 4.5]
DEFAULT_PROMINENCE_GRID = [0.05, 0.1, 0.15, 0.25]


@dataclass
class ElbowArmResult:
    label: str
    mad_multiplier: float
    prominence: float
    mean_returned: float
    precision: float
    recall: BootResult
    recall_delta: DeltaResult | None  # paired vs the default arm; None for the default itself


@dataclass
class ElbowScreenReport:
    min_k: int
    max_k: int
    default_mad_multiplier: float
    default_prominence: float
    n_queries: int
    n_clusters: int
    results: list[ElbowArmResult]


def screen_elbow(
    cfg: Config,
    items: list[QAItem],
    *,
    mad_grid: list[float] | None = None,
    prominence_grid: list[float] | None = None,
    searcher: Searcher | None = None,
    show_progress: bool = False,
) -> ElbowScreenReport:
    """One-factor-at-a-time screen of ``elbow_mad_multiplier``/``elbow_prominence``, paired
    vs the config's own defaults.

    Unlike ``screen_retrieval``, this needs no cache or checkpoint: every arm here is a pure
    postprocessing pass (``find_cutoff``, via :func:`~eval.metrics.elbow_cutoffs`) over one
    already-reranked ``score_items`` call at ``cfg``'s own ``candidates``/``max_k`` — the
    only real cost (one dense query + one rerank pass per item), shared by every arm.
    Re-cutting the same cached scores at a different mad_multiplier/prominence is close to
    free at typical pool/grid sizes, so there's nothing here worth resuming across runs.
    """
    searcher = searcher or build_searcher(cfg)
    min_k, max_k = cfg.retrieval.min_k, cfg.retrieval.max_k
    mad0, prom0 = cfg.retrieval.elbow_mad_multiplier, cfg.retrieval.elbow_prominence
    scores = score_items(
        searcher,
        items,
        candidates=cfg.retrieval.candidates,
        k=max_k,
        rerank=True,
        desc="scoring (shared by every elbow arm)" if show_progress else None,
    )

    grid_arms: list[tuple[str, float, float]] = [
        (f"mad_multiplier={m}", m, prom0) for m in (mad_grid or DEFAULT_MAD_GRID) if m != mad0
    ] + [
        (f"prominence={p}", mad0, p)
        for p in (prominence_grid or DEFAULT_PROMINENCE_GRID)
        if p != prom0
    ]

    default_cutoffs = elbow_cutoffs(scores, min_k, max_k, mad0, prom0)
    default_samples = elbow_recall_samples(default_cutoffs)
    default_boot = cluster_bootstrap(default_samples)

    results = [
        ElbowArmResult(
            label="default",
            mad_multiplier=mad0,
            prominence=prom0,
            mean_returned=mean_returned_at_elbow(default_cutoffs),
            precision=precision_at_elbow(default_cutoffs),
            recall=default_boot,
            recall_delta=None,
        )
    ]
    for label, mad, prom in grid_arms:
        cutoffs = elbow_cutoffs(scores, min_k, max_k, mad, prom)
        samples = elbow_recall_samples(cutoffs)
        results.append(
            ElbowArmResult(
                label=label,
                mad_multiplier=mad,
                prominence=prom,
                mean_returned=mean_returned_at_elbow(cutoffs),
                precision=precision_at_elbow(cutoffs),
                recall=cluster_bootstrap(samples),
                recall_delta=paired_delta(samples, default_samples),
            )
        )
    return ElbowScreenReport(
        min_k=min_k,
        max_k=max_k,
        default_mad_multiplier=mad0,
        default_prominence=prom0,
        n_queries=len(items),
        n_clusters=default_boot.n_clusters,
        results=results,
    )


def format_elbow_screen_report(report: ElbowScreenReport) -> str:
    """Human-readable summary — same "resolution before ranking" rule as ``format_report``:
    a knob whose paired-delta CI straddles zero is worth reporting as *not* worth tuning for
    this pool, not hidden."""
    warning = resolution_warning(report.n_clusters)
    lines = [
        f"n_queries={report.n_queries}  n_clusters={report.n_clusters} papers  "
        f"min_k={report.min_k}  max_k={report.max_k}  "
        f"default: mad_multiplier={report.default_mad_multiplier} "
        f"prominence={report.default_prominence}",
    ]
    if warning:
        lines.append(f"  ⚠ {warning}")
    for r in report.results:
        lo, hi = r.recall.ci_lo, r.recall.ci_hi
        line = (
            f"  {r.label:<20} mean_returned={r.mean_returned:.1f}  "
            f"precision={r.precision:.3f}  recall={r.recall.point:.3f} [{lo:.3f}, {hi:.3f}]"
        )
        if r.recall_delta is not None:
            d = r.recall_delta
            straddles = d.ci_lo <= 0.0 <= d.ci_hi
            flag = "  (straddles zero — not worth tuning here)" if straddles else "  ***"
            line += f"   Δrecall={d.delta:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]{flag}"
        lines.append(line)
    return "\n".join(lines)
