"""Tier-A optimizer: screen ``reranker.enabled`` and ``retrieval.candidates`` — no re-index.

Tier A is "no re-index" **and** "no re-retrieve." Across every arm the collection, the
embedder, and the cross-encoder scores are fixed; only two knobs vary, and both are pure
post-processing of a single dense pool:

* ``retrieval.candidates`` slices the cached distance-ordered pool to ``[:c]``.
* ``reranker.enabled`` orders that slice by cached rerank scores (on) or by cached cosine
  distance (off).

So we retrieve **once per query** at ``max(grid)`` candidates, score that pool with the
cross-encoder **once**, and cache ``(candidate_ids, distances, rerank_scores, relevant_ids)``.
Every arm is then an in-memory slice + sort — the plan's "near-instant at any pool size."

The equivalence that licenses this: a cross-encoder's ``(query, doc)`` score is independent
of pool depth, so slicing to top-``c`` and sorting by cached scores is bit-identical to what
:func:`eval.harness._retrieve` computes at ``candidates=c``. :func:`score_from_cache` at the
default arm therefore reproduces :func:`eval.harness.score_items` exactly (asserted in tests).

Screening is one-factor-at-a-time around the default (per the plan): each non-default arm
changes exactly one knob and is compared **paired vs default on the same queries** via
:func:`eval.stats.paired_delta`. The full ``reranker × candidates`` grid is Phase 5's ``sweep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from rag.config import Config, LLMRerankerCfg
from rag.llm import build_llm
from rag.reranker import Reranker, build_reranker
from rag.search import Searcher

from .harness import build_searcher
from .metrics import QueryScore, relevant_ids
from .queryset import QAItem
from .stats import (
    BootResult,
    DeltaResult,
    cluster_bootstrap,
    mrr_samples,
    paired_delta,
    resolution_warning,
    success_samples,
)

DEFAULT_CANDIDATE_GRID = [10, 20, 30, 50]


@dataclass
class Arm:
    """One config point in the screen: a candidates depth and a reranker on/off."""

    label: str
    candidates: int
    rerank: bool


@dataclass
class QueryCache:
    """Everything retrieved once per query, reused by every arm.

    ``candidate_ids`` is the dense pool ordered by ascending cosine distance (Chroma's own
    order) at ``max(grid)`` depth; ``rerank_scores`` maps each of those ids to its
    cross-encoder score over the full pool; ``relevant_ids`` is the gold-section set.
    """

    qid: str
    paper_id: str
    candidate_ids: list[str]
    rerank_scores: dict[str, float]
    relevant_ids: set[str]


@dataclass
class ArmResult:
    arm: Arm
    success: BootResult  # this arm's stage-1 ceiling + CI (one-arm bootstrap)
    mrr: BootResult  # this arm's stage-2 MRR@k + CI
    # Paired delta vs the default arm; None for the default arm itself.
    success_delta: DeltaResult | None
    mrr_delta: DeltaResult | None


@dataclass
class ScreenReport:
    default: Arm
    k: int
    n_queries: int
    n_clusters: int
    results: list[ArmResult]


def _build_reranker(cfg: Config) -> Reranker:
    """The config's reranker, built regardless of ``enabled`` — the on-arm always needs it."""
    llm = build_llm(cfg.llm.chat) if isinstance(cfg.reranker, LLMRerankerCfg) else None
    return build_reranker(cfg.reranker, llm=llm)


def build_cache(
    searcher: Searcher,
    items: list[QAItem],
    *,
    max_candidates: int,
    reranker: Reranker,
) -> list[QueryCache]:
    """Retrieve once per query at ``max_candidates`` depth and score the pool with ``reranker``.

    One dense query + one rerank pass per item; every arm reads from the returned cache.
    """
    caches: list[QueryCache] = []
    for i, it in enumerate(items):
        embed_query = getattr(searcher.embedder, "embed_query", None)
        qvec = embed_query([it.query]) if embed_query else searcher.embedder([it.query])
        # Chroma's stubs narrow query_embeddings/results more tightly than runtime accepts;
        # the requested include= keys are always present and non-None here (mirrors search.py).
        res = cast(
            dict[str, Any],
            searcher.collection.query(
                query_embeddings=qvec,  # ty: ignore[invalid-argument-type]  # list invariance vs Chroma's Sequence param
                n_results=max_candidates,
                include=["documents", "distances"],
            ),
        )
        ids: list[str] = res["ids"][0]
        scores = reranker.score(it.query, res["documents"][0]) if ids else []
        caches.append(
            QueryCache(
                qid=str(i),
                paper_id=it.paper_id,
                candidate_ids=ids,
                rerank_scores=dict(zip(ids, scores, strict=True)),
                relevant_ids=relevant_ids(
                    searcher.collection, it.paper_id, it.section_number, it.section_title
                ),
            )
        )
    return caches


def score_from_cache(cache: QueryCache, *, candidates: int, k: int, rerank: bool) -> QueryScore:
    """Derive one arm's :class:`QueryScore` by slicing the cache — no retrieval.

    Slices the distance-ordered pool to top-``candidates``, then either sorts that slice by
    cached rerank score (on) or keeps the distance order (off), and takes the top ``k``.
    """
    cand = cache.candidate_ids[:candidates]
    if rerank:
        ordered = sorted(cand, key=lambda cid: cache.rerank_scores[cid], reverse=True)
        ranked = [(cid, cache.rerank_scores[cid]) for cid in ordered[:k]]
    else:
        # Chroma returned candidate_ids in ascending-distance (descending-similarity) order.
        ranked = [(cid, 0.0) for cid in cand[:k]]
    return QueryScore(
        qid=cache.qid,
        candidate_ids=cand,
        ranked=ranked,
        relevant_ids=cache.relevant_ids,
        paper_id=cache.paper_id,
    )


def _arms(cfg: Config, grid: list[int]) -> list[Arm]:
    """OFAT arm set: the default, the reranker toggle, and one arm per candidates value."""
    c0 = cfg.retrieval.candidates
    r0 = cfg.reranker.enabled
    arms = [
        Arm(label="default", candidates=c0, rerank=r0),
        Arm(label=f"rerank={'off' if r0 else 'on'}", candidates=c0, rerank=not r0),
    ]
    arms += [Arm(label=f"candidates={c}", candidates=c, rerank=r0) for c in grid if c != c0]
    return arms


def screen_tier_a(
    cfg: Config,
    items: list[QAItem],
    *,
    candidate_grid: list[int] | None = None,
    searcher: Searcher | None = None,
) -> ScreenReport:
    """Screen reranker on/off and candidates depth over the loaded pool, paired vs default.

    ``searcher`` is injectable for offline tests; production builds one from ``cfg``.
    """
    searcher = searcher or build_searcher(cfg)
    grid = candidate_grid or DEFAULT_CANDIDATE_GRID
    k = cfg.retrieval.k
    arms = _arms(cfg, grid)
    max_candidates = max(a.candidates for a in arms)
    # Reach past Searcher's public `.reranker` property on purpose: that property lazily builds
    # the default hf cross-encoder, which is the WRONG reranker for an `llm`-type config.
    # `_reranker` is what build_searcher injected (set iff the config enabled it); when it's
    # None we build the config's actual variant, so the rerank-on arm always uses the right one.
    reranker = searcher._reranker or _build_reranker(cfg)
    cache = build_cache(searcher, items, max_candidates=max_candidates, reranker=reranker)

    def arm_scores(arm: Arm) -> list[QueryScore]:
        return [
            score_from_cache(c, candidates=arm.candidates, k=k, rerank=arm.rerank) for c in cache
        ]

    default_scores = arm_scores(arms[0])
    default_succ = success_samples(default_scores)
    default_mrr = mrr_samples(default_scores, k)

    results: list[ArmResult] = []
    for i, arm in enumerate(arms):
        is_default = i == 0
        scores = default_scores if is_default else arm_scores(arm)
        succ = success_samples(scores)
        mrr = mrr_samples(scores, k)
        results.append(
            ArmResult(
                arm=arm,
                success=cluster_bootstrap(succ),
                mrr=cluster_bootstrap(mrr),
                success_delta=None if is_default else paired_delta(succ, default_succ),
                mrr_delta=None if is_default else paired_delta(mrr, default_mrr),
            )
        )

    # Resolution is the WEAKEST comparison in the report, not the friendliest: a paired MRR
    # delta conditions on gold-in-pool-in-both-arms and can cluster on fewer papers than the
    # ceiling. Report that minimum so resolution_warning gates on the number the rankings
    # actually rest on, never on the most optimistic one.
    delta_clusters = [
        d.n_clusters for r in results for d in (r.success_delta, r.mrr_delta) if d is not None
    ]
    return ScreenReport(
        default=arms[0],
        k=k,
        n_queries=len(items),
        n_clusters=min([results[0].success.n_clusters, *delta_clusters]),
        results=results,
    )


def _delta_cell(d: DeltaResult | None) -> str:
    if d is None:
        return "     —"
    # A "detectable" effect must be nonzero: |Δ| must clear the MDD AND the CI must exclude
    # zero. The CI guard matters at the degenerate boundary — a rerank toggle leaves the
    # candidate pool (and so success@candidates) exactly unchanged, giving Δ=0, SE=0, MDD=0;
    # without it `abs(0) >= 0` would falsely star a guaranteed-null difference.
    detectable = (d.ci_lo > 0 or d.ci_hi < 0) and abs(d.delta) >= d.mdd
    flag = "*" if detectable else " "
    return f"{d.delta:+.3f} [{d.ci_lo:+.3f},{d.ci_hi:+.3f}]{flag}"


def format_screen_report(report: ScreenReport) -> str:
    """Human-readable screen. Leads with resolution (n_clusters, small-cluster warning), then
    a per-arm table of ``success`` / ``MRR@k`` with paired Δ-vs-default and its CI; a ``*``
    marks a delta that clears the paired MDD (a difference the pool can reliably detect).
    """
    d = report.default
    lines = [
        f"Tier-A screen (no re-index) — n_clusters={report.n_clusters} papers, "
        f"{report.n_queries} queries, k={report.k}",
        f"  default arm: candidates={d.candidates} rerank={'on' if d.rerank else 'off'}",
    ]
    warning = resolution_warning(report.n_clusters)
    if warning:
        lines.append(f"  ⚠ {warning}")
    lines.append("  (* = |Δ| clears the paired MDD — reliably detectable on this pool)")
    lines.append(
        f"  {'arm':<16}{'success':>9}  {'Δ vs default [CI]':<26}"
        f"{'MRR@' + str(report.k):>8}  {'Δ vs default [CI]':<26}"
    )
    for r in report.results:
        lines.append(
            f"  {r.arm.label:<16}{r.success.point:>9.3f}  {_delta_cell(r.success_delta):<26}"
            f"{r.mrr.point:>8.3f}  {_delta_cell(r.mrr_delta):<26}"
        )
    return "\n".join(lines)
