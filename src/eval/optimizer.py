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

from dataclasses import dataclass, field, replace
from typing import Any, cast

from rag.config import ChunkingCfg, Config, LLMRerankerCfg
from rag.embedders import build_embedder
from rag.llm import build_llm
from rag.reranker import Reranker, build_reranker
from rag.search import Searcher

from .harness import build_searcher, score_items
from .index_isolated import build_isolated_searcher
from .metrics import QueryScore, relevant_ids
from .queryset import QAItem
from .stats import (
    BootResult,
    DeltaResult,
    ceiling_saturation_note,
    cluster_bootstrap,
    mrr_samples,
    paired_delta,
    resolution_warning,
    success_samples,
)

# Caveats surfaced in Tier-B reports so a reader doesn't misread the table (Dre review):
# the max_tokens confound is screen-only (the sweep disentangles it); the eligibility caveat
# applies wherever arms condition on different goldable / gold-in-pool sets (both).
_CAVEAT_CONFOUND = (
    "max_tokens Δ is at fixed candidates — it entangles with pool depth; "
    "confirm direction in `sweep`."
)
_CAVEAT_ELIGIBILITY = (
    "point/CI use each arm's own eligible set (goldable / gold-in-pool differ across arms); "
    "trust the starred paired Δ."
)

DEFAULT_CANDIDATE_GRID = [10, 20, 30, 50]
# OFAT grids for the Tier-B chunking screen: values tried per knob (the config's own value
# is always the paired-against default and is skipped if it appears here). Overridable.
DEFAULT_CHUNK_GRIDS: dict[str, list[float]] = {
    "max_tokens": [256, 1024],
    "overlap_tokens": [0, 128],
    "min_tokens": [12, 48],
    "noise_ratio": [0.3, 0.5],
}
DEFAULT_MAX_TOKENS_GRID = [256, 512, 1024]  # Tier-B re-index axis of the sweep grid


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

    return ScreenReport(
        default=arms[0],
        k=k,
        n_queries=len(items),
        n_clusters=_min_report_clusters(results),
        results=results,
    )


def _min_report_clusters(results: list[Any]) -> int:
    """The WEAKEST comparison's cluster count — never the friendliest.

    A paired MRR delta conditions on gold-in-pool-in-both-arms and can cluster on fewer
    papers than the ceiling, so ``resolution_warning`` must gate on this minimum, not on the
    most optimistic number. Duck-typed over any result with ``success`` / ``*_delta`` fields
    (Tier A ``ArmResult`` and Tier B ``TierBArmResult`` both qualify).
    """
    delta_clusters = [
        d.n_clusters for r in results for d in (r.success_delta, r.mrr_delta) if d is not None
    ]
    return min([results[0].success.n_clusters, *delta_clusters])


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


# --- Shared screen rendering (Tier A + Tier B) — the structs differ per tier, the table does
# not, so only the renderer is shared (the gnarly _delta_cell / column alignment lives once).


def _resolution_header(
    title: str, default_desc: str, *, n_clusters: int, n_queries: int, k: int
) -> list[str]:
    """The lead-with-resolution preamble every screen/sweep shares."""
    lines = [
        f"{title} — n_clusters={n_clusters} papers, {n_queries} queries, k={k}",
        f"  default arm: {default_desc}",
    ]
    warning = resolution_warning(n_clusters)
    if warning:
        lines.append(f"  ⚠ {warning}")
    lines.append("  (* = |Δ| clears the paired MDD — reliably detectable on this pool)")
    lines.append(
        f"  {'arm':<20}{'success':>9}  {'Δ vs default [CI]':<26}"
        f"{'MRR@' + str(k):>8}  {'Δ vs default [CI]':<26}"
    )
    return lines


def _arm_row(
    label: str,
    success: BootResult,
    success_delta: DeltaResult | None,
    mrr: BootResult,
    mrr_delta: DeltaResult | None,
) -> str:
    """One table row: point estimates + paired Δ-vs-default cells (``*`` = clears the MDD)."""
    return (
        f"  {label:<20}{success.point:>9.3f}  {_delta_cell(success_delta):<26}"
        f"{mrr.point:>8.3f}  {_delta_cell(mrr_delta):<26}"
    )


def format_screen_report(report: ScreenReport) -> str:
    """Human-readable Tier-A screen. Leads with resolution (n_clusters, small-cluster warning),
    then a per-arm table of ``success`` / ``MRR@k`` with paired Δ-vs-default and its CI.
    """
    d = report.default
    lines = _resolution_header(
        "Tier-A screen (no re-index)",
        f"candidates={d.candidates} rerank={'on' if d.rerank else 'off'}",
        n_clusters=report.n_clusters,
        n_queries=report.n_queries,
        k=report.k,
    )
    saturated = ceiling_saturation_note(report.results[0].success.point)
    if saturated:
        lines.append(f"  ⚠ {saturated}")
    for r in report.results:
        lines.append(_arm_row(r.arm.label, r.success, r.success_delta, r.mrr, r.mrr_delta))
    return "\n".join(lines)


# --- Tier B: re-index knobs (chunking), each arm an isolated collection (Guard 1) ----------


@dataclass
class TierBArm:
    """One re-index point: a label and the ``ChunkingCfg`` it re-chunks the pool at."""

    label: str
    chunking: ChunkingCfg


@dataclass
class TierBArmResult:
    label: str
    success: BootResult
    mrr: BootResult
    success_delta: DeltaResult | None  # paired vs the default cell; None for default itself
    mrr_delta: DeltaResult | None


@dataclass
class TierBReport:
    default_desc: str
    k: int
    n_queries: int
    n_clusters: int
    results: list[TierBArmResult]
    title: str = "Tier-B screen (re-index per cell)"
    caveats: list[str] = field(default_factory=list)


def tier_b_arms(cfg: Config, *, grids: dict[str, list[float]] | None = None) -> list[TierBArm]:
    """OFAT chunking arms around ``cfg.chunking``: the default plus one arm per off-default
    knob value in ``grids``. Each arm changes exactly one knob (``dataclasses.replace``), so a
    non-null delta attributes to that knob alone. ``ChunkingCfg`` validation still applies —
    an invalid combo (e.g. ``overlap_tokens >= max_tokens``) raises at ``replace`` time.
    """
    ch = cfg.chunking
    grids = grids or DEFAULT_CHUNK_GRIDS
    arms = [TierBArm("default", ch)]
    current = {
        "max_tokens": ch.max_tokens,
        "overlap_tokens": ch.overlap_tokens,
        "min_tokens": ch.min_tokens,
        "noise_ratio": ch.noise_ratio,
    }
    for knob, cur in current.items():
        for v in grids.get(knob, []):
            if v != cur:
                arms.append(TierBArm(f"{knob}={v}", replace(ch, **{knob: v})))
    return arms


def _tier_b_default_desc(cfg: Config) -> str:
    ch = cfg.chunking
    return (
        f"max_tokens={ch.max_tokens} overlap={ch.overlap_tokens} min={ch.min_tokens} "
        f"noise={ch.noise_ratio} candidates={cfg.retrieval.candidates} "
        f"rerank={'on' if cfg.reranker.enabled else 'off'}"
    )


def screen_tier_b(
    cfg: Config,
    pool: dict[str, str],
    items: list[QAItem],
    *,
    db_dir: str,
    grids: dict[str, list[float]] | None = None,
    embedder: Any = None,
    reranker: Reranker | None = None,
) -> TierBReport:
    """OFAT screen over chunking knobs, each arm re-indexed into its own isolated collection.

    Each arm re-chunks the pool and re-embeds it (``build_isolated_searcher``, Guard 1), then
    scores the dev set at ``cfg``'s retrieval settings and compares **paired vs the default
    cell** via :func:`~eval.stats.paired_delta` — qids align by item index, so the delta is a
    true paired comparison even though the two arms live in different collections. The embedder
    (fixed across chunking arms) and the reranker load **once** and are reused for every cell.

    ``db_dir`` is a throwaway directory the caller owns; ``embedder`` is injectable so offline
    tests skip the model download.
    """
    arms = tier_b_arms(cfg, grids=grids)
    embedder = embedder if embedder is not None else build_embedder(cfg.embedding)
    if reranker is None and cfg.reranker.enabled:
        reranker = _build_reranker(cfg)
    candidates, k, rerank = cfg.retrieval.candidates, cfg.retrieval.k, cfg.reranker.enabled

    def arm_samples(arm: TierBArm):
        searcher = build_isolated_searcher(
            pool, arm.chunking, cfg.embedding, db_dir=db_dir, embedder=embedder, reranker=reranker
        )
        scores = score_items(searcher, items, candidates=candidates, k=k, rerank=rerank)
        return success_samples(scores), mrr_samples(scores, k)

    default_succ, default_mrr = arm_samples(arms[0])
    results = [
        TierBArmResult(
            label=arms[0].label,
            success=cluster_bootstrap(default_succ),
            mrr=cluster_bootstrap(default_mrr),
            success_delta=None,
            mrr_delta=None,
        )
    ]
    for arm in arms[1:]:
        succ, mrr = arm_samples(arm)
        results.append(
            TierBArmResult(
                label=arm.label,
                success=cluster_bootstrap(succ),
                mrr=cluster_bootstrap(mrr),
                success_delta=paired_delta(succ, default_succ),
                mrr_delta=paired_delta(mrr, default_mrr),
            )
        )
    # The max_tokens/candidates confound caveat only applies when a max_tokens arm is present;
    # the eligibility caveat always does (each arm's own goldable / gold-in-pool set).
    caveats = [_CAVEAT_ELIGIBILITY]
    if any(r.label.startswith("max_tokens") for r in results):
        caveats.insert(0, _CAVEAT_CONFOUND)
    return TierBReport(
        default_desc=_tier_b_default_desc(cfg),
        k=k,
        n_queries=len(items),
        n_clusters=_min_report_clusters(results),
        results=results,
        caveats=caveats,
    )


def format_tier_b_report(report: TierBReport) -> str:
    """Human-readable Tier-B screen/sweep — same resolution-led table as Tier A, plus the
    caveats that keep a reader from misreading the confounded/eligibility-shifted columns."""
    lines = _resolution_header(
        report.title,
        report.default_desc,
        n_clusters=report.n_clusters,
        n_queries=report.n_queries,
        k=report.k,
    )
    saturated = ceiling_saturation_note(report.results[0].success.point)
    if saturated:
        lines.append(f"  ⚠ {saturated}")
    for r in report.results:
        lines.append(_arm_row(r.label, r.success, r.success_delta, r.mrr, r.mrr_delta))
    for caveat in report.caveats:
        lines.append(f"  · {caveat}")
    return "\n".join(lines)


@dataclass
class SweepArm:
    """One sweep cell: a chunking depth (re-index) × a retrieval slice (cached)."""

    label: str
    max_tokens: int
    candidates: int
    rerank: bool


def _sweep_arms(
    cfg: Config, max_tokens_grid: list[int], candidate_grid: list[int]
) -> list[SweepArm]:
    """The full staged grid: (max_tokens) × (candidates) × (rerank on/off), default first.

    The default cell (``cfg``'s chunking / candidates / reranker state) leads so every other
    arm pairs against it.
    """
    c0, mt0, rr0 = cfg.retrieval.candidates, cfg.chunking.max_tokens, cfg.reranker.enabled
    default = SweepArm("default", mt0, c0, rr0)
    arms = [default]
    for mt in max_tokens_grid:
        for c in candidate_grid:
            for rr in (True, False):
                if (mt, c, rr) == (mt0, c0, rr0):
                    continue
                arms.append(SweepArm(f"mt={mt} c={c} rr={'on' if rr else 'off'}", mt, c, rr))
    return arms


def sweep(
    cfg: Config,
    pool: dict[str, str],
    items: list[QAItem],
    *,
    db_dir: str,
    max_tokens_grid: list[int] | None = None,
    candidate_grid: list[int] | None = None,
    embedder: Any = None,
    reranker: Reranker | None = None,
) -> TierBReport:
    """Staged Tier-B × Tier-A grid: re-index once per ``max_tokens`` (the expensive axis),
    then derive every ``candidates × rerank`` slice of that cell from **one cached** dense +
    rerank pass (:func:`build_cache` / :func:`score_from_cache`, Phase 4's exactness property).
    So cost is ``|max_tokens_grid|`` re-indexes, not the full grid. Reports each cell paired
    vs the default via :func:`~eval.stats.paired_delta`.
    """
    mt_grid = max_tokens_grid or DEFAULT_MAX_TOKENS_GRID
    cand_grid = candidate_grid or DEFAULT_CANDIDATE_GRID
    k = cfg.retrieval.k
    embedder = embedder if embedder is not None else build_embedder(cfg.embedding)
    # rerank on-arms always need the reranker, even if the config has it off (mirrors Tier A).
    reranker = reranker if reranker is not None else _build_reranker(cfg)
    max_c = max(cand_grid)

    # One isolated cell + one cached dense/rerank pass per distinct max_tokens.
    caches: dict[int, list[QueryCache]] = {}
    for mt in sorted({cfg.chunking.max_tokens, *mt_grid}):
        searcher = build_isolated_searcher(
            pool,
            replace(cfg.chunking, max_tokens=mt),
            cfg.embedding,
            db_dir=db_dir,
            embedder=embedder,
            reranker=reranker,
        )
        caches[mt] = build_cache(searcher, items, max_candidates=max_c, reranker=reranker)

    def arm_samples(arm: SweepArm):
        scores = [
            score_from_cache(c, candidates=arm.candidates, k=k, rerank=arm.rerank)
            for c in caches[arm.max_tokens]
        ]
        return success_samples(scores), mrr_samples(scores, k)

    arms = _sweep_arms(cfg, mt_grid, cand_grid)
    default_succ, default_mrr = arm_samples(arms[0])
    results = [
        TierBArmResult(
            label=arms[0].label,
            success=cluster_bootstrap(default_succ),
            mrr=cluster_bootstrap(default_mrr),
            success_delta=None,
            mrr_delta=None,
        )
    ]
    for arm in arms[1:]:
        succ, mrr = arm_samples(arm)
        results.append(
            TierBArmResult(
                label=arm.label,
                success=cluster_bootstrap(succ),
                mrr=cluster_bootstrap(mrr),
                success_delta=paired_delta(succ, default_succ),
                mrr_delta=paired_delta(mrr, default_mrr),
            )
        )
    return TierBReport(
        default_desc=_tier_b_default_desc(cfg),
        k=k,
        n_queries=len(items),
        n_clusters=_min_report_clusters(results),
        results=results,
        title="Tier-B sweep (re-index per max_tokens × cached candidates/rerank)",
        # No confound caveat — the sweep IS the disentangling grid. Eligibility still varies
        # across max_tokens cells (different gold-in-pool sets), so keep that one.
        caveats=[_CAVEAT_ELIGIBILITY],
    )
