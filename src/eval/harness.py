"""Single-config run: drive an eval split through ``Searcher`` and report the two metrics.

This is flow step 1 — "run the default config, see what you get" — with no sweep. It
composes ``rag`` (``Searcher`` + its embedder/collection/reranker) rather than forking it,
but reaches past ``Searcher.search`` (which hides chunk ids and the pre-rerank pool) for the
per-stage ids the metrics need. That thin retrieval seam is also what the retrieval screen's
caching builds on later.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rag.config import BM25Cfg, Config, HFEmbeddingCfg, LLMRerankerCfg
from rag.llm import build_llm
from rag.reranker import build_reranker
from rag.search import Searcher

from .checkpoint import CheckpointWriter, resume_units
from .comparative_metrics import (
    ComparativeQueryScore,
    comparative_mrr_samples,
    comparative_success_samples,
)
from .comparative_queryset import ComparativeQAItem
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
    mrr_samples,
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
    llm = (
        build_llm(cfg.llm.chat)
        if isinstance(cfg.reranker, LLMRerankerCfg) or cfg.multi_query.enabled
        else None
    )
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
        multi_query_enabled=cfg.multi_query.enabled,
        multi_query_n=cfg.multi_query.n_paraphrases,
        multi_query_fetch_multiplier=cfg.multi_query.fetch_multiplier,
        llm=llm,
    )


def _retrieve(
    searcher: Searcher, query: str, *, candidates: int, k: int, rerank: bool
) -> tuple[list[str], list[tuple[str, float]]]:
    """Return ``(candidate_ids, ranked_top_k)`` for one query.

    ``candidate_ids`` is the full pre-rerank pool (for stage-1 recall) — dense alone, or
    RRF-fused with BM25 when ``searcher.sparse_enabled`` (``build_searcher`` sets this from
    ``cfg.sparse.enabled``, so ``run``/``confirm`` genuinely reflect a config that has hybrid
    turned on, not just the dense-only reading the pre-hybrid version of this function gave).
    ``ranked`` is the reranked top-k as ``(chunk_id, score)`` (for stage-2 nDCG). It crosses
    ``Searcher.recall`` — the same pre-rerank pool seam product search uses.
    """
    pool = searcher.recall(query, candidates=candidates, max_k=k)
    if not pool.entries:
        return [], []
    ids = [entry.chunk_id for entry in pool.entries]
    if not rerank:
        ranked = [(entry.chunk_id, entry.result.score) for entry in pool.entries][:k]
        return ids, ranked
    docs = [entry.result.text for entry in pool.entries]
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


# --- per-paper: measures Searcher.search(per_paper=True) vs False on a randomly-sampled
# multi-paper scope, swept across a candidates grid. Standalone (paperlens-eval per-paper
# sweep/confirm), not a screen tier — per_paper has no config field to feed a confirm-style
# config.yaml block. See per-paper-eval-spec.md for the full design.

# Duplicated from optimizer.DEFAULT_CANDIDATE_GRID's value (not imported: optimizer.py
# already imports from this module, so importing back would cycle) — same default grid,
# reused for consistency with the rest of the harness.
DEFAULT_PER_PAPER_N = 4
DEFAULT_PER_PAPER_CANDIDATE_GRID = [10, 20, 30, 50]


def build_per_paper_scopes(
    items: list[QAItem], pool: dict[str, str], n_papers: int, seed: int
) -> dict[str, list[str]]:
    """One scope per item: its own gold paper plus ``n_papers - 1`` other papers sampled
    uniformly at random from the rest of the pool, seeded for reproducibility. Degrades to
    every paper in the pool when the pool has fewer than ``n_papers`` papers total.
    """
    all_papers = sorted(pool)
    rng = random.Random(seed)
    scopes: dict[str, list[str]] = {}
    for i, item in enumerate(items):
        qid = str(i)
        others = [p for p in all_papers if p != item.paper_id]
        n_others = min(n_papers - 1, len(others))
        chosen = rng.sample(others, n_others) if n_others > 0 else []
        scopes[qid] = [item.paper_id, *chosen]
    return scopes


def _retrieve_scoped(
    searcher: Searcher,
    query: str,
    *,
    paper_ids: list[str],
    per_paper: bool,
    candidates: int,
    k: int,
    rerank: bool,
    budget_matched: bool = False,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Return ``(candidate_ids, ranked_top_k)`` for one query, scoped to ``paper_ids`` —
    the ``per_paper``-aware counterpart to :func:`_retrieve`. ``len(candidate_ids)`` is the
    actual total pool size fetched (mean_pool_size derives directly from it — no separate
    tracking needed).

    ``per_paper=False``: one fused (dense[+sparse]) pool over the whole scope — today's
    default ``search()`` behavior if ``paper_ids`` were the turn's resolved filter.
    ``per_paper=True``: one fused pool *per paper* in ``paper_ids``, concatenated; each
    paper's own budget is either ``Searcher.search``'s real formula
    (``max(k, min(candidates, candidates // n_papers))``, ``budget_matched=False`` — the
    production variant) or a budget-matched cap with no ``k`` floor
    (``max(1, candidates // n_papers)``, ``budget_matched=True``) — see
    per-paper-eval-spec.md's "Retrieval primitive" for why both exist.
    """
    pool = searcher.recall(
        query,
        candidates=candidates,
        max_k=k,
        paper_ids=paper_ids,
        per_paper=per_paper,
        per_paper_min_candidates=1 if budget_matched else None,
    )
    if not pool.entries:
        return [], []
    all_ids = [entry.chunk_id for entry in pool.entries]
    if not rerank:
        ranked = [(entry.chunk_id, entry.result.score) for entry in pool.entries][:k]
        return all_ids, ranked
    docs = [entry.result.text for entry in pool.entries]
    scores = searcher.reranker.score(query, docs)
    ranked = sorted(zip(all_ids, scores, strict=True), key=lambda t: t[1], reverse=True)[:k]
    return all_ids, ranked


def score_items_scoped(
    searcher: Searcher,
    items: list[QAItem],
    scopes: dict[str, list[str]],
    *,
    per_paper: bool,
    candidates: int,
    k: int,
    rerank: bool,
    budget_matched: bool = False,
    desc: str | None = None,
) -> list[QueryScore]:
    """:func:`score_items`'s scoped counterpart: retrieve every item within its own
    ``scopes[qid]`` instead of whole-corpus, pair with its gold-section relevant set.
    """
    scores: list[QueryScore] = []
    for i, it in enumerate(tqdm(items, desc=desc, disable=desc is None, leave=False)):
        qid = str(i)
        cand_ids, ranked = _retrieve_scoped(
            searcher,
            it.query,
            paper_ids=scopes[qid],
            per_paper=per_paper,
            candidates=candidates,
            k=k,
            rerank=rerank,
            budget_matched=budget_matched,
        )
        rel = relevant_ids(searcher.collection, it.paper_id, it.section_number, it.section_title)
        scores.append(
            QueryScore(
                qid=qid,
                candidate_ids=cand_ids,
                ranked=ranked,
                relevant_ids=rel,
                paper_id=it.paper_id,
            )
        )
    return scores


def _mean_pool_size(scores: list[QueryScore]) -> float:
    return sum(len(s.candidate_ids) for s in scores) / len(scores) if scores else 0.0


@dataclass
class PerPaperArmResult:
    label: str  # "off" | "on (production)" | "on (budget-matched)"
    candidates: int  # nominal grid point
    mean_pool_size: float  # actual mean total candidates fetched — exposes the budget
    # confound if present, distinct from the nominal candidates value above
    success: BootResult  # stage-1 ceiling for this arm
    mrr: BootResult  # stage-2, at k
    success_delta: DeltaResult | None  # paired vs the off-arm at the SAME candidates value
    mrr_delta: DeltaResult | None


@dataclass
class PerPaperReport:
    n_papers: int
    candidates_grid: list[int]
    k: int
    seed: int  # the scope-assignment seed this run used
    n_queries: int
    n_clusters: int
    results: list[PerPaperArmResult]  # [off, on-prod, on-matched?] per grid point


def per_paper_sweep(
    cfg: Config,
    items: list[QAItem],
    pool: dict[str, str],
    *,
    n_papers: int = DEFAULT_PER_PAPER_N,
    candidates_grid: list[int] | None = None,
    seed: int = 0,
    searcher: Searcher | None = None,
    show_progress: bool = False,
    checkpoint_path: Path | None = None,
) -> PerPaperReport:
    """Exploratory paired comparison of ``per_paper=True`` vs ``False``, at each
    ``candidates`` grid point, as both the production variant (``Searcher.search``'s real
    formula) and — where it would actually differ — the budget-matched variant (no ``k``
    floor, total pool approximately matching the off-arm's — off by up to ``n_papers - 1``
    candidates when ``candidates`` doesn't divide ``n_papers`` evenly, integer division).
    Reports candidate points, not a recommendation — see :func:`per_paper_confirm` for the
    one-shot, fresh-seed check that alone is allowed to recommend. Checkpointed at
    ``(candidates, variant)`` granularity: real cost here, `(n_papers + 1)` dense(+sparse)
    queries per item per grid point per variant — not a free cache re-slice like
    ``screen --tier elbow``. Reranks iff ``cfg.reranker.enabled``, mirroring ``run``/
    ``screen_retrieval`` — this measures your config, not a fixed assumption.
    """
    searcher = searcher or build_searcher(cfg)
    k = cfg.retrieval.max_k
    rerank = cfg.reranker.enabled
    grid = sorted(candidates_grid or DEFAULT_PER_PAPER_CANDIDATE_GRID)
    scopes = build_per_paper_scopes(items, pool, n_papers, seed)

    done: dict[str, dict[str, Any]] = {}
    writer: CheckpointWriter | None = None
    if checkpoint_path is not None:
        header = {
            "n_papers": n_papers,
            "candidates_grid": grid,
            "seed": seed,
            "k": k,
            "rerank": rerank,
            "index_count": searcher.collection.count(),
        }
        done = resume_units(checkpoint_path, header)
        writer = CheckpointWriter(checkpoint_path, header)
        if done:
            print(f"  resuming: {len(done)} cell(s) already scored")

    def scored_arm(
        unit_id: str, *, per_paper: bool, candidates: int, budget_matched: bool
    ) -> list[QueryScore]:
        if unit_id in done:
            records = done[unit_id]["scores"]
            return [_query_score_from_record(str(i), rec) for i, rec in enumerate(records)]
        scores = score_items_scoped(
            searcher,
            items,
            scopes,
            per_paper=per_paper,
            candidates=candidates,
            k=k,
            rerank=rerank,
            budget_matched=budget_matched,
            desc=f"per-paper sweep {unit_id}" if show_progress else None,
        )
        if writer is not None:
            writer.append(unit_id, {"scores": [_query_score_to_record(s) for s in scores]})
        return scores

    results: list[PerPaperArmResult] = []
    n_clusters = 0
    for candidates in grid:
        off_scores = scored_arm(
            f"{candidates}:off", per_paper=False, candidates=candidates, budget_matched=False
        )
        off_success = success_samples(off_scores)
        off_mrr = mrr_samples(off_scores, k)
        off_boot_success = cluster_bootstrap(off_success)
        n_clusters = max(n_clusters, off_boot_success.n_clusters)
        results.append(
            PerPaperArmResult(
                label="off",
                candidates=candidates,
                mean_pool_size=_mean_pool_size(off_scores),
                success=off_boot_success,
                mrr=cluster_bootstrap(off_mrr),
                success_delta=None,
                mrr_delta=None,
            )
        )

        prod_scores = scored_arm(
            f"{candidates}:production", per_paper=True, candidates=candidates, budget_matched=False
        )
        prod_success = success_samples(prod_scores)
        prod_mrr = mrr_samples(prod_scores, k)
        results.append(
            PerPaperArmResult(
                label="on (production)",
                candidates=candidates,
                mean_pool_size=_mean_pool_size(prod_scores),
                success=cluster_bootstrap(prod_success),
                mrr=cluster_bootstrap(prod_mrr),
                success_delta=paired_delta(prod_success, off_success),
                mrr_delta=paired_delta(prod_mrr, off_mrr),
            )
        )

        if candidates // n_papers < k:
            matched_scores = scored_arm(
                f"{candidates}:budget_matched",
                per_paper=True,
                candidates=candidates,
                budget_matched=True,
            )
            matched_success = success_samples(matched_scores)
            matched_mrr = mrr_samples(matched_scores, k)
            results.append(
                PerPaperArmResult(
                    label="on (budget-matched)",
                    candidates=candidates,
                    mean_pool_size=_mean_pool_size(matched_scores),
                    success=cluster_bootstrap(matched_success),
                    mrr=cluster_bootstrap(matched_mrr),
                    success_delta=paired_delta(matched_success, off_success),
                    mrr_delta=paired_delta(matched_mrr, off_mrr),
                )
            )

    if writer is not None:
        writer.finish()

    return PerPaperReport(
        n_papers=n_papers,
        candidates_grid=grid,
        k=k,
        seed=seed,
        n_queries=len(items),
        n_clusters=n_clusters,
        results=results,
    )


def format_per_paper_sweep(report: PerPaperReport) -> str:
    """Exploratory report: resolution first, then every arm grouped by grid point, ending
    in a plain list of candidate points (never a recommendation — that's `confirm`'s job).
    """
    warning = resolution_warning(report.n_clusters)
    lines = [
        f"n_queries={report.n_queries}  n_papers_per_scope={report.n_papers}  "
        f"k={report.k}  seed={report.seed}",
    ]
    by_candidates: dict[int, list[PerPaperArmResult]] = {}
    for r in report.results:
        by_candidates.setdefault(r.candidates, []).append(r)

    candidate_points: list[str] = []
    saw_budget_matched = False
    for candidates in report.candidates_grid:
        arms = by_candidates.get(candidates, [])
        if not arms:
            continue
        off = arms[0]
        lines.append(f"candidates={candidates}   n_clusters={off.success.n_clusters} papers")
        if warning:
            lines.append(f"  ⚠ {warning}")
        for arm in arms:
            lo, hi = arm.success.ci_lo, arm.success.ci_hi
            mlo, mhi = arm.mrr.ci_lo, arm.mrr.ci_hi
            lines.append(
                f"  {arm.label:<20} pool={arm.mean_pool_size:.1f}  "
                f"success={arm.success.point:.3f} [{lo:.3f}, {hi:.3f}]  "
                f"MRR@{report.k}={arm.mrr.point:.3f} [{mlo:.3f}, {mhi:.3f}]"
            )
            if arm.success_delta is not None:
                d = arm.success_delta
                straddles = d.ci_lo <= 0.0 <= d.ci_hi
                flag = "  (straddles zero — not worth tuning here)" if straddles else "  ***"
                lines.append(f"    Δsuccess={d.delta:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]{flag}")
                if arm.label == "on (production)" and off.mean_pool_size > 0:
                    ratio = arm.mean_pool_size / off.mean_pool_size
                    if not straddles and ratio >= 1.5:
                        lines.append(
                            f"    ⚠ pool {ratio:.1f}x the off-arm's — may reflect fetch "
                            f"volume, not allocation; see the budget-matched row before "
                            f"attributing this to crowding"
                        )
                if not straddles:
                    variant = "production" if arm.label == "on (production)" else "budget-matched"
                    candidate_points.append(f"candidates={candidates} --variant {variant}")
            if arm.label == "on (budget-matched)":
                saw_budget_matched = True
        if len(arms) == 2:
            lines.append(
                "    (budget-matched not computed — production pool already ≈ "
                "off-arm's at this grid point)"
            )

    n_variants = 3 if saw_budget_matched else 2
    lines.append("")
    lines.append(
        f"Exploratory only — grid searched over {len(report.candidates_grid)} candidates "
        f"values x up to {n_variants} variants x 2 metrics."
    )
    if candidate_points:
        lines.append("Candidate points to confirm (fresh seed, one at a time):")
        for cp in candidate_points:
            lines.append(f"  per-paper confirm {cp} --seed <fresh>")
    else:
        lines.append(
            "Candidate points to confirm: none — no arm's success delta cleared its CI "
            "at any grid point in this run."
        )
    return "\n".join(lines)


def per_paper_confirm(
    cfg: Config,
    items: list[QAItem],
    pool: dict[str, str],
    *,
    n_papers: int,
    candidates: int,
    variant: str,
    seed: int,
    searcher: Searcher | None = None,
) -> PerPaperArmResult:
    """One candidates value, one variant (``"production"`` or ``"budget-matched"``), a
    fresh scope-seed (deliberately never the sweep's own — that would just re-read the
    same draw). The only per-paper entry point allowed to inform a recommendation — see
    per-paper-eval-spec.md's "CLI". No checkpoint: a single point is cheap enough not to
    need one. Reranks iff ``cfg.reranker.enabled``, matching ``per_paper_sweep``.
    """
    if variant not in ("production", "budget-matched"):
        raise ValueError(f"variant must be 'production' or 'budget-matched', got {variant!r}")
    searcher = searcher or build_searcher(cfg)
    k = cfg.retrieval.max_k
    rerank = cfg.reranker.enabled
    scopes = build_per_paper_scopes(items, pool, n_papers, seed)

    off_scores = score_items_scoped(
        searcher, items, scopes, per_paper=False, candidates=candidates, k=k, rerank=rerank
    )
    on_scores = score_items_scoped(
        searcher,
        items,
        scopes,
        per_paper=True,
        candidates=candidates,
        k=k,
        rerank=rerank,
        budget_matched=variant == "budget-matched",
    )
    off_success = success_samples(off_scores)
    on_success = success_samples(on_scores)
    return PerPaperArmResult(
        label=f"on ({variant})",
        candidates=candidates,
        mean_pool_size=_mean_pool_size(on_scores),
        success=cluster_bootstrap(on_success),
        mrr=cluster_bootstrap(mrr_samples(on_scores, k)),
        success_delta=paired_delta(on_success, off_success),
        mrr_delta=paired_delta(mrr_samples(on_scores, k), mrr_samples(off_scores, k)),
    )


def format_per_paper_confirm(result: PerPaperArmResult, *, n_papers: int, seed: int) -> str:
    d = result.success_delta
    assert d is not None  # always paired vs off in per_paper_confirm
    lo, hi = result.success.ci_lo, result.success.ci_hi
    straddles = d.ci_lo <= 0.0 <= d.ci_hi
    lines = [
        f"candidates={result.candidates}  n_papers={n_papers}  seed={seed} (fresh draw)",
        f"  {result.label:<20} pool={result.mean_pool_size:.1f}  "
        f"success={result.success.point:.3f} [{lo:.3f}, {hi:.3f}]",
        f"    Δsuccess={d.delta:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]"
        f"{'  (straddles zero — not confirmed)' if straddles else '  *** confirmed'}",
        "",
    ]
    if straddles:
        lines.append(
            f"Not confirmed: the {result.label.split('(')[-1].rstrip(')')} delta at "
            f"candidates={result.candidates} does not clear its CI on an independent "
            f"scope draw. No recommendation change from the current default (off)."
        )
    else:
        lines.append(
            f"Confirmed: {result.label} shows a real benefit at candidates={result.candidates} "
            f"on an independent scope draw (Δsuccess={d.delta:+.3f})."
        )
    return "\n".join(lines)


# --- comparative: measures Searcher.search(per_paper=True) vs False on genuinely
# cross-paper questions (gold spans 2+ papers) generated by comparative_queryset's trial
# loop. Standalone (paperlens-eval comparative gen/sweep/confirm), not a screen tier --
# same reasoning per_paper's own standalone command has (per_paper has no config field to
# feed a confirm-style config.yaml block). See comparative-eval-spec.md for the full design.

# Placeholder, matching what per_paper_sweep's much larger single-paper sweep achieves on
# this pool -- the honest floor for "this pool can produce a cluster count at all
# comparable to what the rest of the harness already treats as thin". Primary-paper
# attribution (comparative_metrics.py) discards every member paper of an item except the
# earliest-drawn one, so comparative_sweep/confirm predictably land at n_clusters far
# below per_paper_sweep's own, and confirm's recommendation language must never speak with
# more confidence than that resolution can support.
COMPARATIVE_MIN_CLUSTERS = 8


def _comparative_mean_pool_size(scores: list[ComparativeQueryScore]) -> float:
    return sum(len(s.candidate_ids) for s in scores) / len(scores) if scores else 0.0


def _n_papers_distribution(items: list[ComparativeQAItem]) -> tuple[int, int, float]:
    sizes = [len(it.sections) for it in items]
    if not sizes:
        return 0, 0, 0.0
    return min(sizes), max(sizes), sum(sizes) / len(sizes)


def score_comparative_items_scoped(
    searcher: Searcher,
    items: list[ComparativeQAItem],
    *,
    per_paper: bool,
    candidates: int,
    k: int,
    rerank: bool,
    budget_matched: bool = False,
    desc: str | None = None,
) -> list[ComparativeQueryScore]:
    """score_items_scoped's comparative counterpart: each item's own gold ``sections``
    list IS its scope -- no separate scope-assignment step, unlike per_paper_sweep's
    randomly-bundled ones (a comparative item already names exactly which papers it
    spans). ``primary_paper_id`` is ``sections[0]``'s paper -- the trial's earliest-drawn
    paper among the matched subset (see ``comparative_queryset.build_comparative_queryset``).
    """
    scores: list[ComparativeQueryScore] = []
    for i, it in enumerate(tqdm(items, desc=desc, disable=desc is None, leave=False)):
        qid = str(i)
        paper_ids = [s.paper_id for s in it.sections]
        cand_ids, ranked = _retrieve_scoped(
            searcher,
            it.query,
            paper_ids=paper_ids,
            per_paper=per_paper,
            candidates=candidates,
            k=k,
            rerank=rerank,
            budget_matched=budget_matched,
        )
        relevant_by_paper = {
            s.paper_id: relevant_ids(searcher.collection, s.paper_id, s.number or "", s.title)
            for s in it.sections
        }
        scores.append(
            ComparativeQueryScore(
                qid=qid,
                candidate_ids=cand_ids,
                ranked=ranked,
                relevant_ids_by_paper=relevant_by_paper,
                primary_paper_id=it.sections[0].paper_id,
            )
        )
    return scores


def _comparative_query_score_to_record(score: ComparativeQueryScore) -> dict[str, Any]:
    return {
        "candidate_ids": score.candidate_ids,
        "ranked": [list(pair) for pair in score.ranked],
        "relevant_ids_by_paper": {
            pid: sorted(ids) for pid, ids in score.relevant_ids_by_paper.items()
        },
        "primary_paper_id": score.primary_paper_id,
    }


def _comparative_query_score_from_record(qid: str, record: dict[str, Any]) -> ComparativeQueryScore:
    return ComparativeQueryScore(
        qid=qid,
        candidate_ids=record["candidate_ids"],
        ranked=[(cid, s) for cid, s in record["ranked"]],
        relevant_ids_by_paper={
            pid: set(ids) for pid, ids in record["relevant_ids_by_paper"].items()
        },
        primary_paper_id=record["primary_paper_id"],
    )


def _comparative_items_fingerprint(items: list[ComparativeQAItem]) -> str:
    """A cheap checkpoint trip-wire: unlike the corpus fingerprint (which only depends on
    paper content), comparative items are produced by non-deterministic LLM calls, so the
    same pool can yield a completely different dev split across two ``gen`` runs. Hashing
    the item queries keeps a stale checkpoint from silently merging with a fresh split's
    scores."""
    h = hashlib.sha256()
    for q in sorted(it.query for it in items):
        h.update(q.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class ComparativeArmResult:
    label: str  # "off" | "on (production)" | "on (budget-matched)"
    candidates: int  # nominal grid point
    mean_pool_size: float
    success: BootResult
    mrr: BootResult
    success_delta: DeltaResult | None  # paired vs the off-arm at the SAME candidates value
    mrr_delta: DeltaResult | None


@dataclass
class ComparativeSweepReport:
    candidates_grid: list[int]
    k: int
    n_queries: int
    n_clusters: int
    n_papers_min: int
    n_papers_max: int
    n_papers_mean: float
    results: list[ComparativeArmResult]  # [off, on-prod, on-matched?] per grid point


def comparative_sweep(
    cfg: Config,
    items: list[ComparativeQAItem],
    *,
    candidates_grid: list[int] | None = None,
    searcher: Searcher | None = None,
    show_progress: bool = False,
    checkpoint_path: Path | None = None,
) -> ComparativeSweepReport:
    """Exploratory paired comparison of ``per_paper=True`` vs ``False`` on genuinely
    cross-paper questions, at each ``candidates`` grid point, as both the production
    variant and -- where it would actually differ for at least one item -- the
    budget-matched variant. Mirrors ``per_paper_sweep``'s shape exactly, swapping the
    scoped-retrieval scoring layer for the comparative one; reports candidate points, not
    a recommendation -- see :func:`comparative_confirm` for the one-shot, held-out-split
    check that alone is allowed to recommend. Reranks iff ``cfg.reranker.enabled``,
    mirroring ``per_paper_sweep``/``run``/``screen_retrieval``.

    Unlike ``per_paper_sweep`` (one fixed ``n_papers`` for every item), each comparative
    item carries its own group size (``len(item.sections)``, 2-6) -- so the
    budget-matched skip check (`candidates // n < k`, per-paper-eval-spec.md's "Retrieval
    primitive") is evaluated per item's own size; a grid point skips budget-matched only
    when it would be identical to production for *every* item present, not just one.
    """
    searcher = searcher or build_searcher(cfg)
    k = cfg.retrieval.max_k
    rerank = cfg.reranker.enabled
    grid = sorted(candidates_grid or DEFAULT_PER_PAPER_CANDIDATE_GRID)
    n_papers_min, n_papers_max, n_papers_mean = _n_papers_distribution(items)
    group_sizes = [len(it.sections) for it in items]

    done: dict[str, dict[str, Any]] = {}
    writer: CheckpointWriter | None = None
    if checkpoint_path is not None:
        header = {
            "candidates_grid": grid,
            "k": k,
            "rerank": rerank,
            "items_fingerprint": _comparative_items_fingerprint(items),
            "index_count": searcher.collection.count(),
        }
        done = resume_units(checkpoint_path, header)
        writer = CheckpointWriter(checkpoint_path, header)
        if done:
            print(f"  resuming: {len(done)} cell(s) already scored")

    def scored_arm(
        unit_id: str, *, per_paper: bool, candidates: int, budget_matched: bool
    ) -> list[ComparativeQueryScore]:
        if unit_id in done:
            records = done[unit_id]["scores"]
            return [
                _comparative_query_score_from_record(str(i), rec) for i, rec in enumerate(records)
            ]
        scores = score_comparative_items_scoped(
            searcher,
            items,
            per_paper=per_paper,
            candidates=candidates,
            k=k,
            rerank=rerank,
            budget_matched=budget_matched,
            desc=f"comparative sweep {unit_id}" if show_progress else None,
        )
        if writer is not None:
            writer.append(
                unit_id, {"scores": [_comparative_query_score_to_record(s) for s in scores]}
            )
        return scores

    results: list[ComparativeArmResult] = []
    n_clusters = 0
    for candidates in grid:
        off_scores = scored_arm(
            f"{candidates}:off", per_paper=False, candidates=candidates, budget_matched=False
        )
        off_success = comparative_success_samples(off_scores)
        off_mrr = comparative_mrr_samples(off_scores, k)
        off_boot_success = cluster_bootstrap(off_success)
        n_clusters = max(n_clusters, off_boot_success.n_clusters)
        results.append(
            ComparativeArmResult(
                label="off",
                candidates=candidates,
                mean_pool_size=_comparative_mean_pool_size(off_scores),
                success=off_boot_success,
                mrr=cluster_bootstrap(off_mrr),
                success_delta=None,
                mrr_delta=None,
            )
        )

        prod_scores = scored_arm(
            f"{candidates}:production", per_paper=True, candidates=candidates, budget_matched=False
        )
        prod_success = comparative_success_samples(prod_scores)
        prod_mrr = comparative_mrr_samples(prod_scores, k)
        results.append(
            ComparativeArmResult(
                label="on (production)",
                candidates=candidates,
                mean_pool_size=_comparative_mean_pool_size(prod_scores),
                success=cluster_bootstrap(prod_success),
                mrr=cluster_bootstrap(prod_mrr),
                success_delta=paired_delta(prod_success, off_success),
                mrr_delta=paired_delta(prod_mrr, off_mrr),
            )
        )

        if any(candidates // n < k for n in group_sizes):
            matched_scores = scored_arm(
                f"{candidates}:budget_matched",
                per_paper=True,
                candidates=candidates,
                budget_matched=True,
            )
            matched_success = comparative_success_samples(matched_scores)
            matched_mrr = comparative_mrr_samples(matched_scores, k)
            results.append(
                ComparativeArmResult(
                    label="on (budget-matched)",
                    candidates=candidates,
                    mean_pool_size=_comparative_mean_pool_size(matched_scores),
                    success=cluster_bootstrap(matched_success),
                    mrr=cluster_bootstrap(matched_mrr),
                    success_delta=paired_delta(matched_success, off_success),
                    mrr_delta=paired_delta(matched_mrr, off_mrr),
                )
            )

    if writer is not None:
        writer.finish()

    return ComparativeSweepReport(
        candidates_grid=grid,
        k=k,
        n_queries=len(items),
        n_clusters=n_clusters,
        n_papers_min=n_papers_min,
        n_papers_max=n_papers_max,
        n_papers_mean=n_papers_mean,
        results=results,
    )


def format_comparative_sweep(report: ComparativeSweepReport) -> str:
    """Exploratory report: resolution and the ``n_papers`` distribution first (pooling
    across group sizes can hide a heterogeneous per_paper effect even with paired deltas
    -- comparative-eval-spec.md's "Metrics"), then every arm grouped by grid point, ending
    in a plain list of candidate points (never a recommendation -- that's confirm's job).
    """
    warning = resolution_warning(report.n_clusters)
    lines = [
        f"n_queries={report.n_queries}  "
        f"n_papers_per_item={report.n_papers_min}..{report.n_papers_max} "
        f"(mean {report.n_papers_mean:.1f})  n_clusters={report.n_clusters} papers",
    ]
    if warning:
        lines.append(f"  ⚠ {warning}")
    by_candidates: dict[int, list[ComparativeArmResult]] = {}
    for r in report.results:
        by_candidates.setdefault(r.candidates, []).append(r)

    candidate_points: list[str] = []
    saw_budget_matched = False
    for candidates in report.candidates_grid:
        arms = by_candidates.get(candidates, [])
        if not arms:
            continue
        off = arms[0]
        lines.append(f"candidates={candidates}   n_clusters={off.success.n_clusters} papers")
        for arm in arms:
            lo, hi = arm.success.ci_lo, arm.success.ci_hi
            mlo, mhi = arm.mrr.ci_lo, arm.mrr.ci_hi
            lines.append(
                f"  {arm.label:<20} pool={arm.mean_pool_size:.1f}  "
                f"success={arm.success.point:.3f} [{lo:.3f}, {hi:.3f}]  "
                f"MRR@{report.k}={arm.mrr.point:.3f} [{mlo:.3f}, {mhi:.3f}]"
            )
            if arm.success_delta is not None:
                d = arm.success_delta
                straddles = d.ci_lo <= 0.0 <= d.ci_hi
                flag = "  (straddles zero — not worth tuning here)" if straddles else "  ***"
                lines.append(f"    Δsuccess={d.delta:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]{flag}")
                if arm.label == "on (production)" and off.mean_pool_size > 0:
                    ratio = arm.mean_pool_size / off.mean_pool_size
                    if not straddles and ratio >= 1.5:
                        lines.append(
                            f"    ⚠ pool {ratio:.1f}x the off-arm's — may reflect fetch "
                            f"volume, not allocation; see the budget-matched row before "
                            f"attributing this to crowding"
                        )
                if not straddles:
                    variant = "production" if arm.label == "on (production)" else "budget-matched"
                    candidate_points.append(f"candidates={candidates} --variant {variant}")
            if arm.label == "on (budget-matched)":
                saw_budget_matched = True
        if len(arms) == 2:
            lines.append(
                "    (budget-matched not computed — production pool already ≈ "
                "off-arm's at this grid point)"
            )

    n_variants = 3 if saw_budget_matched else 2
    lines.append("")
    lines.append(
        f"Exploratory only — grid searched over {len(report.candidates_grid)} candidates "
        f"values x up to {n_variants} variants x 2 metrics."
    )
    if candidate_points:
        lines.append("Candidate points to confirm (test split, one at a time):")
        for cp in candidate_points:
            lines.append(f"  comparative confirm {cp}")
    else:
        lines.append(
            "Candidate points to confirm: none — no arm's success delta cleared its CI "
            "at any grid point in this run."
        )
    return "\n".join(lines)


@dataclass
class ComparativeConfirmResult:
    off: ComparativeArmResult
    on: ComparativeArmResult  # success_delta/mrr_delta already paired vs off
    n_papers_min: int
    n_papers_max: int
    n_papers_mean: float


def comparative_confirm(
    cfg: Config,
    items: list[ComparativeQAItem],
    *,
    candidates: int,
    variant: str,
    searcher: Searcher | None = None,
) -> ComparativeConfirmResult:
    """One candidates value, one variant, scored against the held-out items passed in --
    the only comparative entry point allowed to inform a recommendation. No ``--seed``:
    unlike ``per_paper_confirm``, there's no retrieval-time scope randomness left to
    redraw here (each item's scope is fixed at generation time) -- the test split's own
    "never touched by sweep" guarantee is what confirm's independence comes from instead.
    No checkpoint: a single point is cheap enough not to need one. Reranks iff
    ``cfg.reranker.enabled``, matching ``comparative_sweep``.
    """
    if variant not in ("production", "budget-matched"):
        raise ValueError(f"variant must be 'production' or 'budget-matched', got {variant!r}")
    searcher = searcher or build_searcher(cfg)
    k = cfg.retrieval.max_k
    rerank = cfg.reranker.enabled

    off_scores = score_comparative_items_scoped(
        searcher, items, per_paper=False, candidates=candidates, k=k, rerank=rerank
    )
    on_scores = score_comparative_items_scoped(
        searcher,
        items,
        per_paper=True,
        candidates=candidates,
        k=k,
        rerank=rerank,
        budget_matched=variant == "budget-matched",
    )
    off_success = comparative_success_samples(off_scores)
    on_success = comparative_success_samples(on_scores)
    off_mrr = comparative_mrr_samples(off_scores, k)
    on_mrr = comparative_mrr_samples(on_scores, k)
    n_papers_min, n_papers_max, n_papers_mean = _n_papers_distribution(items)

    off_result = ComparativeArmResult(
        label="off",
        candidates=candidates,
        mean_pool_size=_comparative_mean_pool_size(off_scores),
        success=cluster_bootstrap(off_success),
        mrr=cluster_bootstrap(off_mrr),
        success_delta=None,
        mrr_delta=None,
    )
    on_result = ComparativeArmResult(
        label=f"on ({variant})",
        candidates=candidates,
        mean_pool_size=_comparative_mean_pool_size(on_scores),
        success=cluster_bootstrap(on_success),
        mrr=cluster_bootstrap(on_mrr),
        success_delta=paired_delta(on_success, off_success),
        mrr_delta=paired_delta(on_mrr, off_mrr),
    )
    return ComparativeConfirmResult(
        off=off_result,
        on=on_result,
        n_papers_min=n_papers_min,
        n_papers_max=n_papers_max,
        n_papers_mean=n_papers_mean,
    )


def format_comparative_confirm(result: ComparativeConfirmResult, *, candidates: int) -> str:
    """One-block report: off + on rows, delta + CI, and a floor-gated verdict. Below
    ``COMPARATIVE_MIN_CLUSTERS``, this NEVER prints ``"Confirmed"``/``"***"`` regardless
    of what the delta's CI says -- see ``COMPARATIVE_MIN_CLUSTERS``'s own docstring and
    comparative-eval-spec.md's "CLI" hard floor. Primary-paper attribution predictably
    lands here more often than not on a small pool, so this floor is the one thing that
    has to exist before this command's output reaches docs/features.md or a UI tooltip.
    """
    d = result.on.success_delta
    assert d is not None  # always paired vs off in comparative_confirm
    variant = result.on.label.split("(")[-1].rstrip(")")
    # The delta's own n_clusters, not either arm's standalone one: paired_delta conditions
    # on the intersection of both arms' eligible items, which is what the "Confirmed"
    # claim below is actually about. These happen to be numerically equal today (success
    # eligibility is a metadata lookup, arm-independent), but the floor should reference
    # what it's gating, not a same-valued-by-coincidence field.
    n_clusters = d.n_clusters
    straddles = d.ci_lo <= 0.0 <= d.ci_hi
    lines = [
        f"candidates={candidates}  variant={variant}  "
        f"n_papers_per_item={result.n_papers_min}..{result.n_papers_max} "
        f"(mean {result.n_papers_mean:.1f})  (test split, touched once)",
        f"  {result.off.label:<20} pool={result.off.mean_pool_size:.1f}  "
        f"success={result.off.success.point:.3f} "
        f"[{result.off.success.ci_lo:.3f}, {result.off.success.ci_hi:.3f}]",
        f"  {result.on.label:<20} pool={result.on.mean_pool_size:.1f}  "
        f"success={result.on.success.point:.3f} "
        f"[{result.on.success.ci_lo:.3f}, {result.on.success.ci_hi:.3f}]",
    ]
    confirmed = n_clusters >= COMPARATIVE_MIN_CLUSTERS and not straddles
    flag = "  *** confirmed" if confirmed else ""
    lines.append(f"    Δsuccess={d.delta:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]{flag}")
    lines.append("")
    if n_clusters < COMPARATIVE_MIN_CLUSTERS:
        lines.append(
            f"Too few clusters to confirm anything (n_clusters={n_clusters} < "
            f"COMPARATIVE_MIN_CLUSTERS={COMPARATIVE_MIN_CLUSTERS}) — the delta above "
            f"{'straddles zero' if straddles else 'cleared its CI'}, but at this few "
            f'papers that\'s not enough to say "confirmed." Read as anecdotal, not '
            f"decision-grade. No recommendation change from the current default (off)."
        )
    elif straddles:
        lines.append(
            f"Not confirmed: the {variant} delta at candidates={candidates} does not "
            f"clear its CI. No recommendation change from the current default (off)."
        )
    else:
        lines.append(
            f"Confirmed: on ({variant}) shows a real benefit on cross-paper synthesis "
            f"questions at candidates={candidates} on this pool (Δsuccess={d.delta:+.3f})."
        )
    return "\n".join(lines)
