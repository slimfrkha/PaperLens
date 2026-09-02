"""Retrieval optimizer: screen ``reranker.enabled`` and ``retrieval.candidates`` — no re-index.

This is "no re-index" **and** "no re-retrieve." Across every arm the collection, the
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

Screening is one-factor-at-a-time around the default: each non-default arm changes exactly
one knob and is compared **paired vs default on the same queries** via
:func:`eval.stats.paired_delta`. The full ``reranker × candidates`` grid is ``sweep``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rag.config import BM25Cfg, ChunkingCfg, Config, LLMRerankerCfg
from rag.embedders import build_embedder
from rag.llm import LLMBackend, build_llm
from rag.query_expansion import generate_paraphrases
from rag.reranker import Reranker, build_reranker
from rag.search import RecallPoolSnapshot, Searcher
from rag.sparse import BM25Index, build_sparse_index

from .checkpoint import CheckpointWriter, resume_units
from .harness import build_searcher, score_items
from .index_isolated import build_isolated_searcher
from .metrics import QueryScore, relevant_ids
from .queryset import QAItem
from .stats import (
    BootResult,
    DeltaResult,
    Sample,
    ceiling_saturation_note,
    cluster_bootstrap,
    mrr_samples,
    paired_delta,
    resolution_warning,
    success_samples,
)

# Caveats surfaced in chunking reports so a reader doesn't misread the table (Dre review):
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
_RECALL_POOL_CACHE_SCHEMA_VERSION = 2
# OFAT grids for the chunking screen: values tried per knob (the config's own value
# is always the paired-against default and is skipped if it appears here). Overridable.
DEFAULT_CHUNK_GRIDS: dict[str, list[float]] = {
    "max_tokens": [256, 1024],
    "overlap_tokens": [0, 128],
    "min_tokens": [12, 48],
    "noise_ratio": [0.3, 0.5],
}
DEFAULT_MAX_TOKENS_GRID = [256, 512, 1024]  # chunking re-index axis of the sweep grid


@dataclass
class Arm:
    """One config point in the screen: a candidates depth, a reranker on/off, hybrid on/off,
    and multi-query on/off."""

    label: str
    candidates: int
    rerank: bool
    sparse: bool = False
    multi_query: bool = False


@dataclass
class QueryCache:
    """Everything retrieved once per query, reused by every arm.

    ``snapshot`` contains every raw ranked source list at the widest required depth. It owns
    later dense/hybrid/multi-query materialization, so this cache never reimplements RRF.
    ``rerank_scores`` maps every hydrated snapshot id to its cross-encoder score.
    """

    qid: str
    paper_id: str
    snapshot: RecallPoolSnapshot
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
    caveats: list[str] = field(default_factory=list)


def build_reranker_for_cfg(cfg: Config) -> Reranker:
    """The config's reranker, built regardless of ``enabled`` — the on-arm always needs it."""
    llm = build_llm(cfg.llm.chat) if isinstance(cfg.reranker, LLMRerankerCfg) else None
    return build_reranker(cfg.reranker, llm=llm)


def _query_cache_to_record(cache: QueryCache) -> dict[str, Any]:
    return {
        "paper_id": cache.paper_id,
        "snapshot": cache.snapshot.to_record(),
        "rerank_scores": cache.rerank_scores,
        "relevant_ids": sorted(cache.relevant_ids),
    }


def _query_cache_from_record(qid: str, record: dict[str, Any]) -> QueryCache:
    return QueryCache(
        qid=qid,
        paper_id=record["paper_id"],
        snapshot=RecallPoolSnapshot.from_record(record["snapshot"]),
        rerank_scores=record["rerank_scores"],
        relevant_ids=set(record["relevant_ids"]),
    )


def _build_cache_header(
    max_candidates: int,
    *,
    hybrid: bool,
    fetch_multiplier: int,
    index_count: int | None,
    multi_query_n: int = 0,
    multi_query_fetch_multiplier: int = 3,
) -> dict[str, Any]:
    return {
        "recall_pool_cache_schema": _RECALL_POOL_CACHE_SCHEMA_VERSION,
        "max_candidates": max_candidates,
        "hybrid": hybrid,
        "fetch_multiplier": fetch_multiplier,
        "index_count": index_count,
        "multi_query_n": multi_query_n,
        "multi_query_fetch_multiplier": multi_query_fetch_multiplier,
    }


def build_cache(
    searcher: Searcher,
    items: list[QAItem],
    *,
    max_candidates: int,
    reranker: Reranker,
    desc: str | None = None,
    sparse: BM25Index | None = None,
    fetch_multiplier: int = 3,
    checkpoint_path: Path | None = None,
    index_count: int | None = None,
    checkpoint_finish: bool = True,
    multi_query_llm: LLMBackend | None = None,
    multi_query_n: int = 0,
    multi_query_fetch_multiplier: int = 3,
) -> list[QueryCache]:
    """Retrieve once per query and score the pool with ``reranker``.

    At ``max_candidates`` depth when ``sparse`` is ``None``. When ``sparse`` is given, both the
    dense query and the BM25 query widen to ``max_candidates * fetch_multiplier`` — the same
    over-fetch-before-fusion margin ``Searcher.search`` uses (Revision note 2 in
    ``hybrid_retrieval_plan.md``) — so a hybrid arm's cache has the same fusion margin the real
    ``Searcher`` would give it, not an artificially narrowed one.

    One dense query (+ one BM25 query when ``sparse`` is given) + one rerank pass per item over
    the **union** of both pools; every arm reads from the returned cache. The default (non-
    hybrid) arm stays exact: its ``candidate_ids`` are the dense pool alone, unaffected by
    whether ``sparse`` widened the cache underneath it. ``desc`` labels a progress bar over
    ``items`` (this is the one expensive pass every retrieval arm derives from); ``None``
    (offline tests) runs silent.

    ``multi_query_llm``/``multi_query_n``/``multi_query_fetch_multiplier`` add multi-query
    expansion to the cache build: when ``multi_query_n > 0``, each item gets one extra LLM call
    (paraphrase generation) plus one extra dense query (+ one extra BM25 query when ``sparse`` is
    also given) per paraphrase, at ``max_candidates * multi_query_fetch_multiplier`` depth —
    deliberately its **own** multiplier, independent of ``fetch_multiplier`` (sparse's), mirroring
    ``Searcher.search``'s own ``multi_query_fetch_multiplier``. Reusing ``fetch_multiplier`` here
    would silently give the multi-query arm zero fusion headroom whenever ``sparse`` is ``None``
    (the common case — screening multi-query without also screening hybrid), understating its
    real recall lift relative to what production would deliver. Unlike hybrid's BM25 pool, this
    widening is genuinely NOT free (each paraphrase needs its own embedding), so it's costed
    explicitly here rather than folded silently into every screen run. Every variant's ids that
    aren't already in the union get backfilled and reranked too, so ``score_from_cache``'s
    ``multi_query`` fuse always has a rerank score for whatever it selects.

    ``checkpoint_path`` makes this resumable — same pattern as ``harness.score_items``: each
    ``QueryCache`` is appended (flushed) the moment it's computed. ``index_count`` is folded
    into the header as the same index-identity trip-wire ``score_items`` uses — pass
    ``searcher.collection.count()`` when ``searcher`` reads a persistent, externally-mutable
    index (retrieval screen); pass (or leave) ``None`` when it's a fresh isolated collection
    whose identity is already pinned by the caller's own checkpoint filename (``sweep``), so
    no searcher needs to exist yet just to compute this. ``checkpoint_finish``, when ``False``,
    leaves a fully-populated checkpoint file on disk instead of deleting it on completion — for
    a caller (``sweep``) that owns several such files across one command and wants to defer
    cleanup until *all* of them are done, not just this one.
    """
    done: dict[str, dict[str, Any]] = {}
    writer: CheckpointWriter | None = None
    if checkpoint_path is not None:
        header = _build_cache_header(
            max_candidates,
            hybrid=sparse is not None,
            fetch_multiplier=fetch_multiplier,
            index_count=index_count,
            multi_query_n=multi_query_n,
            multi_query_fetch_multiplier=multi_query_fetch_multiplier,
        )
        done = resume_units(checkpoint_path, header)
        writer = CheckpointWriter(checkpoint_path, header)
        if done:
            print(f"  resuming: {len(done)}/{len(items)} queries already cached")

    caches: list[QueryCache] = []
    for i, it in enumerate(tqdm(items, desc=desc, disable=desc is None, leave=False)):
        qid = str(i)
        if qid in done:
            caches.append(_query_cache_from_record(qid, done[qid]))
            continue
        paraphrases: list[str] = []
        if multi_query_n > 0:
            assert multi_query_llm is not None  # enforced by the caller (screen_retrieval)
            paraphrases = generate_paraphrases(it.query, multi_query_llm, n=multi_query_n)
        captured = searcher.capture_recall_snapshot(
            it.query,
            max_candidates,
            sparse=sparse,
            fetch_multiplier=fetch_multiplier,
            variants=paraphrases,
            multi_query_fetch_multiplier=multi_query_fetch_multiplier,
            base_fetch_multiplier=(
                multi_query_fetch_multiplier
                if paraphrases
                else fetch_multiplier
                if sparse is not None
                else 1
            ),
        )
        union_ids = list(captured.results_by_id)
        scores = (
            reranker.score(it.query, [captured.results_by_id[cid].text for cid in union_ids])
            if union_ids
            else []
        )
        cache = QueryCache(
            qid=qid,
            paper_id=it.paper_id,
            snapshot=captured.snapshot,
            rerank_scores=dict(zip(union_ids, scores, strict=True)),
            relevant_ids=relevant_ids(
                searcher.collection, it.paper_id, it.section_number, it.section_title
            ),
        )
        caches.append(cache)
        if writer is not None:
            writer.append(qid, _query_cache_to_record(cache))
    if writer is not None:
        if checkpoint_finish:
            writer.finish()
        else:
            writer.close()
    return caches


def score_from_cache(
    cache: QueryCache,
    *,
    candidates: int,
    k: int,
    rerank: bool,
    sparse: bool = False,
    rrf_k: int = 60,
    fetch_multiplier: int = 3,
    multi_query: bool = False,
    multi_query_fetch_multiplier: int = 3,
) -> QueryScore:
    """Derive one arm's :class:`QueryScore` by slicing the cache — no retrieval.

    Non-hybrid, non-multi-query arms (the default and every existing OFAT arm) slice the dense
    pool to top-``candidates`` — unaffected by whether the cache was built wide for a hybrid or
    multi-query arm elsewhere, since list slicing truncates the same either way. Hybrid arms
    RRF-fuse an over-fetched slice of both pools first — the same over-fetch-then-truncate shape
    ``Searcher.search`` uses — then either sort the fused pool by cached rerank score (on) or
    keep fused order (off), and take the top ``k``. ``multi_query`` does the same, but as one
    flat RRF pass over the default pool plus every cached paraphrase-variant pool at once —
    mirroring ``Searcher.search``'s own flat fuse over all variants, not a fusion of per-variant
    fusions — using ``multi_query_fetch_multiplier``, its own knob, **not** ``fetch_multiplier``
    (sparse's): reusing the latter would give the multi-query arm zero fusion headroom whenever
    ``sparse`` is off. Mutually exclusive with ``sparse`` in this screen (each is its own OFAT
    arm); if both were somehow requested together, ``multi_query`` wins since its fuse already
    includes the default pool.
    """
    selection = cache.snapshot.materialize(
        candidates,
        sparse=sparse,
        multi_query=multi_query,
        rrf_k=rrf_k,
        fetch_multiplier=fetch_multiplier,
        multi_query_fetch_multiplier=multi_query_fetch_multiplier,
    )
    cand = list(selection.candidate_ids)
    if rerank:
        ordered = sorted(cand, key=lambda cid: cache.rerank_scores[cid], reverse=True)
        ranked = [(cid, cache.rerank_scores[cid]) for cid in ordered[:k]]
    else:
        # Non-hybrid: Chroma returned candidate_ids in ascending-distance order. Hybrid: `cand`
        # is already RRF-fused order.
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


def screen_retrieval(
    cfg: Config,
    items: list[QAItem],
    *,
    candidate_grid: list[int] | None = None,
    searcher: Searcher | None = None,
    show_progress: bool = False,
    hybrid: bool = False,
    multi_query: bool = False,
    checkpoint_path: Path | None = None,
) -> ScreenReport:
    """Screen reranker on/off and candidates depth over the loaded pool, paired vs default.

    ``searcher`` is injectable for offline tests; production builds one from ``cfg``.
    ``show_progress`` shows a bar over the one cache-build pass; off by default (offline
    tests), the CLI opts in. ``hybrid`` adds one more arm, ``"hybrid=on"``, that RRF-fuses BM25
    into the default arm's candidates/rerank settings — gated on this explicit flag, **not** on
    ``cfg.sparse.enabled``: the harness's whole purpose is proposing config changes before
    they're committed, so requiring ``sparse.enabled: true`` first would make it impossible to
    measure the one case that matters (deciding whether to turn it on). ``multi_query`` adds
    ``"multi_query=on"`` the same way, gated on this flag rather than ``cfg.multi_query.enabled``
    — but unlike ``hybrid``, it is NOT free: each paraphrase needs a fresh embedding, so it
    widens the cache-build pass itself (one LLM call plus ``cfg.multi_query.n_paraphrases`` extra
    dense queries per item) rather than being a free re-slice of an already-cached pool — see
    :func:`build_cache`. ``checkpoint_path`` is threaded straight through to :func:`build_cache`
    — every arm derived from that cache is free, so only the cache-build pass itself needs to be
    resumable.
    """
    searcher = searcher or build_searcher(cfg)
    grid = candidate_grid or DEFAULT_CANDIDATE_GRID
    k = cfg.retrieval.max_k
    arms = _arms(cfg, grid)
    rrf_k, fetch_multiplier = cfg.sparse.rrf_k, cfg.sparse.fetch_multiplier
    sparse_index: BM25Index | None = None
    caveats: list[str] = []
    if hybrid:
        c0, r0 = cfg.retrieval.candidates, cfg.reranker.enabled
        arms = [*arms, Arm(label="hybrid=on", candidates=c0, rerank=r0, sparse=True)]
        k1 = cfg.sparse.k1 if isinstance(cfg.sparse, BM25Cfg) else 1.5
        b = cfg.sparse.b if isinstance(cfg.sparse, BM25Cfg) else 0.75
        sparse_index = build_sparse_index(searcher.collection, k1=k1, b=b)
        caveats.append(
            f"hybrid=on uses rrf_k={rrf_k} and fetch_multiplier={fetch_multiplier} as-is from "
            "config — neither is OFAT-screened in this pass (RRF is fairly insensitive to k in "
            "the 10-100 range; fetch_multiplier's shipped default, 3, is the "
            "peer-implementation reference value, not something screened here)"
        )
    multi_query_llm: LLMBackend | None = None
    multi_query_n = 0
    multi_query_fetch_multiplier = cfg.multi_query.fetch_multiplier
    if multi_query:
        c0, r0 = cfg.retrieval.candidates, cfg.reranker.enabled
        arms = [*arms, Arm(label="multi_query=on", candidates=c0, rerank=r0, multi_query=True)]
        multi_query_llm = build_llm(cfg.llm.chat)
        multi_query_n = cfg.multi_query.n_paraphrases
        caveats.append(
            f"multi_query=on generates {multi_query_n} paraphrases per query via llm.chat and "
            "widens the cache-build pass accordingly — one extra LLM call plus "
            f"{multi_query_n} extra dense queries per item, unlike hybrid=on's free re-slice of "
            "an already-cached pool"
        )
    max_candidates = max(a.candidates for a in arms)
    # Reach past Searcher's public `.reranker` property on purpose: that property lazily builds
    # the default hf cross-encoder, which is the WRONG reranker for an `llm`-type config.
    # `_reranker` is what build_searcher injected (set iff the config enabled it); when it's
    # None we build the config's actual variant, so the rerank-on arm always uses the right one.
    reranker = searcher._reranker or build_reranker_for_cfg(cfg)
    cache = build_cache(
        searcher,
        items,
        max_candidates=max_candidates,
        reranker=reranker,
        desc="retrieval cache" if show_progress else None,
        sparse=sparse_index,
        fetch_multiplier=fetch_multiplier,
        checkpoint_path=checkpoint_path,
        index_count=searcher.collection.count() if checkpoint_path is not None else None,
        multi_query_llm=multi_query_llm,
        multi_query_n=multi_query_n,
        multi_query_fetch_multiplier=multi_query_fetch_multiplier,
    )

    def arm_scores(arm: Arm) -> list[QueryScore]:
        return [
            score_from_cache(
                c,
                candidates=arm.candidates,
                k=k,
                rerank=arm.rerank,
                sparse=arm.sparse,
                rrf_k=rrf_k,
                fetch_multiplier=fetch_multiplier,
                multi_query=arm.multi_query,
                multi_query_fetch_multiplier=multi_query_fetch_multiplier,
            )
            for c in cache
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
        caveats=caveats,
    )


def _min_report_clusters(results: list[Any]) -> int:
    """The WEAKEST comparison's cluster count — never the friendliest.

    A paired MRR delta conditions on gold-in-pool-in-both-arms and can cluster on fewer
    papers than the ceiling, so ``resolution_warning`` must gate on this minimum, not on the
    most optimistic number. Duck-typed over any result with ``success`` / ``*_delta`` fields
    (``ArmResult`` from the retrieval screen and ``ChunkingArmResult`` from the chunking one
    both qualify).
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


def _cell_eta_line(tag: str, idx: int, total: int, t0: float) -> str:
    """``[i/n] tag: done (elapsed, ~remaining)`` — an aggregate ETA across re-index cells.

    Not a fabricated up-front guess (the plan's Guard against that): it's a straight
    extrapolation from cells *already measured* in this run, printed after each one
    completes — the number a long ``screen --tier chunking``/``sweep`` needs and a
    per-cell tqdm bar can't give, since that bar only knows the cell it's timing.
    """
    elapsed = time.monotonic() - t0
    remaining = elapsed / idx * (total - idx)
    return f"  {tag}: done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"


# --- Shared screen rendering (retrieval + chunking) — the structs differ per screen, the table
# does not, so only the renderer is shared (the gnarly _delta_cell / column alignment lives once).


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
    """Human-readable retrieval screen. Leads with resolution (n_clusters, small-cluster
    warning), then a per-arm table of ``success`` / ``MRR@k`` with paired Δ-vs-default and its
    CI.
    """
    d = report.default
    lines = _resolution_header(
        "Retrieval screen (no re-index)",
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
    for caveat in report.caveats:
        lines.append(f"  · {caveat}")
    return "\n".join(lines)


# --- Chunking: re-index knobs, each arm an isolated collection (Guard 1) -------------------


@dataclass
class ChunkingArm:
    """One re-index point: a label and the ``ChunkingCfg`` it re-chunks the pool at."""

    label: str
    chunking: ChunkingCfg


@dataclass
class ChunkingArmResult:
    label: str
    success: BootResult
    mrr: BootResult
    success_delta: DeltaResult | None  # paired vs the default cell; None for default itself
    mrr_delta: DeltaResult | None


@dataclass
class ChunkingReport:
    default_desc: str
    k: int
    n_queries: int
    n_clusters: int
    results: list[ChunkingArmResult]
    title: str = "Chunking screen (re-index per cell)"
    caveats: list[str] = field(default_factory=list)


def chunking_arms(cfg: Config, *, grids: dict[str, list[float]] | None = None) -> list[ChunkingArm]:
    """OFAT chunking arms around ``cfg.chunking``: the default plus one arm per off-default
    knob value in ``grids``. Each arm changes exactly one knob (``dataclasses.replace``), so a
    non-null delta attributes to that knob alone. ``ChunkingCfg`` validation still applies —
    an invalid combo (e.g. ``overlap_tokens >= max_tokens``) raises at ``replace`` time.
    """
    ch = cfg.chunking
    grids = grids or DEFAULT_CHUNK_GRIDS
    arms = [ChunkingArm("default", ch)]
    current = {
        "max_tokens": ch.max_tokens,
        "overlap_tokens": ch.overlap_tokens,
        "min_tokens": ch.min_tokens,
        "noise_ratio": ch.noise_ratio,
    }
    for knob, cur in current.items():
        for v in grids.get(knob, []):
            if v != cur:
                arms.append(ChunkingArm(f"{knob}={v}", replace(ch, **{knob: v})))
    return arms


def _chunking_default_desc(cfg: Config) -> str:
    ch = cfg.chunking
    return (
        f"max_tokens={ch.max_tokens} overlap={ch.overlap_tokens} min={ch.min_tokens} "
        f"noise={ch.noise_ratio} candidates={cfg.retrieval.candidates} "
        f"rerank={'on' if cfg.reranker.enabled else 'off'}"
    )


def _cell_samples_to_record(success: list[Sample], mrr: list[Sample]) -> dict[str, Any]:
    def serialize(samples: list[Sample]) -> list[dict[str, Any]]:
        return [
            {"qid": s.qid, "paper_id": s.paper_id, "eligible": s.eligible, "value": s.value}
            for s in samples
        ]

    return {"success_samples": serialize(success), "mrr_samples": serialize(mrr)}


def _cell_samples_from_record(record: dict[str, Any]) -> tuple[list[Sample], list[Sample]]:
    def deserialize(data: list[dict[str, Any]]) -> list[Sample]:
        return [
            Sample(qid=d["qid"], paper_id=d["paper_id"], eligible=d["eligible"], value=d["value"])
            for d in data
        ]

    return deserialize(record["success_samples"]), deserialize(record["mrr_samples"])


def screen_chunking(
    cfg: Config,
    pool: dict[str, str],
    items: list[QAItem],
    *,
    db_dir: str,
    grids: dict[str, list[float]] | None = None,
    embedder: Any = None,
    reranker: Reranker | None = None,
    show_progress: bool = False,
    checkpoint_path: Path | None = None,
) -> ChunkingReport:
    """OFAT screen over chunking knobs, each arm re-indexed into its own isolated collection.

    Each arm re-chunks the pool and re-embeds it (``build_isolated_searcher``, Guard 1), then
    scores the dev set at ``cfg``'s retrieval settings and compares **paired vs the default
    cell** via :func:`~eval.stats.paired_delta` — qids align by item index, so the delta is a
    true paired comparison even though the two arms live in different collections. The embedder
    (fixed across chunking arms) and the reranker load **once** and are reused for every cell.

    ``db_dir`` is a throwaway directory the caller owns; ``embedder`` is injectable so offline
    tests skip the model download. ``show_progress`` shows embed/score bars per cell (off by
    default, the CLI opts in); the per-cell ``done (elapsed, ~remaining)`` line always prints —
    it's one line per cell, not a bar, and is the aggregate-ETA number a long screen needs.

    ``checkpoint_path`` makes this resumable at cell granularity: a cell's raw
    ``success``/``mrr`` :class:`~eval.stats.Sample` lists (not just the point estimate — the
    default cell's samples are needed to compute every other cell's paired delta) are appended
    the moment that cell finishes, so a cell already in the checkpoint skips re-embedding and
    re-scoring entirely. The header pins the grid and ``cfg.chunking``'s own values — either
    changing invalidates the checkpoint (a different arm set).
    """
    arms = chunking_arms(cfg, grids=grids)
    embedder = embedder if embedder is not None else build_embedder(cfg.embedding)
    if reranker is None and cfg.reranker.enabled:
        reranker = build_reranker_for_cfg(cfg)
    candidates, k, rerank = cfg.retrieval.candidates, cfg.retrieval.max_k, cfg.reranker.enabled

    n_arms = len(arms)
    t0 = time.monotonic()

    done: dict[str, dict[str, Any]] = {}
    writer: CheckpointWriter | None = None
    if checkpoint_path is not None:
        header = {"grids": grids, "base_chunking": asdict(cfg.chunking), "n_items": len(items)}
        done = resume_units(checkpoint_path, header)
        writer = CheckpointWriter(checkpoint_path, header)
        if done:
            print(f"  resuming: {len(done)}/{n_arms} cells already done")

    def arm_samples(arm: ChunkingArm, idx: int):
        tag = f"[{idx}/{n_arms}] {arm.label}"
        if arm.label in done:
            succ, mrr = _cell_samples_from_record(done[arm.label])
            print(_cell_eta_line(tag, idx, n_arms, t0))
            return succ, mrr
        searcher = build_isolated_searcher(
            pool,
            arm.chunking,
            cfg.embedding,
            db_dir=db_dir,
            embedder=embedder,
            reranker=reranker,
            desc=f"{tag} embed" if show_progress else None,
        )
        scores = score_items(
            searcher,
            items,
            candidates=candidates,
            k=k,
            rerank=rerank,
            desc=f"{tag} score" if show_progress else None,
        )
        print(_cell_eta_line(tag, idx, n_arms, t0))
        succ, mrr = success_samples(scores), mrr_samples(scores, k)
        if writer is not None:
            writer.append(arm.label, _cell_samples_to_record(succ, mrr))
        return succ, mrr

    default_succ, default_mrr = arm_samples(arms[0], 1)
    results = [
        ChunkingArmResult(
            label=arms[0].label,
            success=cluster_bootstrap(default_succ),
            mrr=cluster_bootstrap(default_mrr),
            success_delta=None,
            mrr_delta=None,
        )
    ]
    for i, arm in enumerate(arms[1:], 2):
        succ, mrr = arm_samples(arm, i)
        results.append(
            ChunkingArmResult(
                label=arm.label,
                success=cluster_bootstrap(succ),
                mrr=cluster_bootstrap(mrr),
                success_delta=paired_delta(succ, default_succ),
                mrr_delta=paired_delta(mrr, default_mrr),
            )
        )
    if writer is not None:
        writer.finish()
    # The max_tokens/candidates confound caveat only applies when a max_tokens arm is present;
    # the eligibility caveat always does (each arm's own goldable / gold-in-pool set).
    caveats = [_CAVEAT_ELIGIBILITY]
    if any(r.label.startswith("max_tokens") for r in results):
        caveats.insert(0, _CAVEAT_CONFOUND)
    return ChunkingReport(
        default_desc=_chunking_default_desc(cfg),
        k=k,
        n_queries=len(items),
        n_clusters=_min_report_clusters(results),
        results=results,
        caveats=caveats,
    )


def format_chunking_report(report: ChunkingReport) -> str:
    """Human-readable chunking screen/sweep — same resolution-led table as the retrieval
    screen, plus the caveats that keep a reader from misreading the confounded/
    eligibility-shifted columns."""
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
    show_progress: bool = False,
    checkpoint_dir: Path | None = None,
    fingerprint: str | None = None,
) -> ChunkingReport:
    """Staged chunking × retrieval grid: re-index once per ``max_tokens`` (the expensive axis),
    then derive every ``candidates × rerank`` slice of that cell from **one cached** dense +
    rerank pass (:func:`build_cache` / :func:`score_from_cache`'s exactness property).
    So cost is ``|max_tokens_grid|`` re-indexes, not the full grid. Reports each cell paired
    vs the default via :func:`~eval.stats.paired_delta`. ``show_progress`` shows embed/cache
    bars per cell (off by default, the CLI opts in); the per-cell ``done`` line always prints.

    ``checkpoint_dir``/``fingerprint`` (both required together) make each ``max_tokens`` cell's
    cache-build resumable — one checkpoint file per cell
    (``<checkpoint_dir>/<fingerprint>.sweep.mt<N>.ckpt.jsonl``), since each is an independent
    ``build_cache`` pass over its own isolated collection. A cell whose cache is already fully
    checkpointed still re-embeds nothing further, same as an uninterrupted call.
    """
    mt_grid = max_tokens_grid or DEFAULT_MAX_TOKENS_GRID
    cand_grid = candidate_grid or DEFAULT_CANDIDATE_GRID
    k = cfg.retrieval.max_k
    embedder = embedder if embedder is not None else build_embedder(cfg.embedding)
    # rerank on-arms always need the reranker, even if the config has it off (mirrors the
    # retrieval screen).
    reranker = reranker if reranker is not None else build_reranker_for_cfg(cfg)
    max_c = max(cand_grid)

    # One isolated cell + one cached dense/rerank pass per distinct max_tokens.
    caches: dict[int, list[QueryCache]] = {}
    cells = sorted({cfg.chunking.max_tokens, *mt_grid})
    checkpoint_paths: dict[int, Path] = (
        {mt: checkpoint_dir / f"{fingerprint}.sweep.mt{mt}.ckpt.jsonl" for mt in cells}
        if checkpoint_dir is not None and fingerprint is not None
        else {}
    )
    t0 = time.monotonic()
    for i, mt in enumerate(cells, 1):
        tag = f"[{i}/{len(cells)}] max_tokens={mt}"
        ckpt_path = checkpoint_paths.get(mt)
        if ckpt_path is not None:
            header = _build_cache_header(max_c, hybrid=False, fetch_multiplier=3, index_count=None)
            done = resume_units(ckpt_path, header)
            if len(done) == len(items):
                # Whole cell already cached from a prior run — skip the re-index entirely,
                # not just the per-query retrieval (that's the point of not deleting a
                # cell's checkpoint until the whole sweep succeeds — see checkpoint_finish
                # below).
                caches[mt] = [
                    _query_cache_from_record(str(j), done[str(j)]) for j in range(len(items))
                ]
                print(f"  resuming: {tag} already fully cached — skipping re-index")
                print(_cell_eta_line(tag, i, len(cells), t0))
                continue
        searcher = build_isolated_searcher(
            pool,
            replace(cfg.chunking, max_tokens=mt),
            cfg.embedding,
            db_dir=db_dir,
            embedder=embedder,
            reranker=reranker,
            desc=f"{tag} embed" if show_progress else None,
        )
        caches[mt] = build_cache(
            searcher,
            items,
            max_candidates=max_c,
            reranker=reranker,
            desc=f"{tag} cache" if show_progress else None,
            checkpoint_path=ckpt_path,
            checkpoint_finish=False,  # sweep deletes all cell checkpoints itself, once every
            # cell has succeeded — not per-cell, so an already-finished earlier cell is still
            # skippable if a later cell is what gets interrupted.
        )
        print(_cell_eta_line(tag, i, len(cells), t0))
    for ckpt_path in checkpoint_paths.values():
        ckpt_path.unlink(missing_ok=True)

    def arm_samples(arm: SweepArm):
        scores = [
            score_from_cache(c, candidates=arm.candidates, k=k, rerank=arm.rerank)
            for c in caches[arm.max_tokens]
        ]
        return success_samples(scores), mrr_samples(scores, k)

    arms = _sweep_arms(cfg, mt_grid, cand_grid)
    default_succ, default_mrr = arm_samples(arms[0])
    results = [
        ChunkingArmResult(
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
            ChunkingArmResult(
                label=arm.label,
                success=cluster_bootstrap(succ),
                mrr=cluster_bootstrap(mrr),
                success_delta=paired_delta(succ, default_succ),
                mrr_delta=paired_delta(mrr, default_mrr),
            )
        )
    return ChunkingReport(
        default_desc=_chunking_default_desc(cfg),
        k=k,
        n_queries=len(items),
        n_clusters=_min_report_clusters(results),
        results=results,
        title="Sweep (re-index per max_tokens × cached candidates/rerank)",
        # No confound caveat — the sweep IS the disentangling grid. Eligibility still varies
        # across max_tokens cells (different gold-in-pool sets), so keep that one.
        caveats=[_CAVEAT_ELIGIBILITY],
    )
