"""Paper-clustered bootstrap: resolution (MDD) and delta CIs over the loaded pool.

The statistical unit is the **paper, not the query**. Twenty questions about one paper are
~one observation, not twenty — they share the paper's vocabulary, structure, and whatever
made retrieval easy or hard. So every interval here resamples *papers* with replacement
(cluster bootstrap) and reports ``n_clusters`` alongside the number, because a null result
on a pool of 11 papers means "little to resolve here," not "no effect exists."

Two things this module exists to compute:

* **Resolution (MDD).** The minimum detectable difference at *this pool's* paper count.
  Printed up front so every reported delta is read against it. A pool whose MDD swamps any
  plausible config gain is one where the honest answer is "the default is fine" — that is a
  finding, not a failure.
* **Paired delta CIs.** Two arms scored on the *same* questions; the CI on ``arm_a − arm_b``
  from the paired cluster bootstrap. Paired is where the power is — the arms are highly
  correlated (same queries, small config change), so the paired SE is far below either
  arm's standalone SE.

**Conditioning is intersection, not per-arm** (:func:`paired_delta`). Each metric drops
ineligible queries (``success`` keeps *goldable*, ``MRR`` keeps *gold-in-pool*), and
eligibility can differ between arms — a query gold-in-pool for arm A but not B. Conditioning
each arm on its own subset would compare two means over different denominators — not a paired
comparison at all. So the paired delta keeps only queries eligible in **both** arms.

Everything is a mean of a per-query ``value`` over the eligible subset — mathematically the
same quantity the aggregate metric reports, decomposed per query so the bootstrap can
resample it. That decomposition *is* the "reuse the metric as the statistic" move: no
ratio-of-means bias, because the metric is recomputed on each resampled set, never averaged
from per-paper means.

Small-cluster caveat: the percentile cluster bootstrap under-covers with few clusters, so a
pool below :data:`MIN_CLUSTERS` triggers :func:`resolution_warning`. The honest fix at very
few clusters is a wild/paired cluster bootstrap with a t(G−1) reference — deferred; not built
until a real pool needs it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .metrics import QueryScore, reciprocal_rank

# Standard normal quantiles (no scipy dependency): z_{0.975} and z_{0.80}.
Z_ALPHA = 1.959964  # two-sided 95%
Z_POWER = 0.841621  # 80% power
MDD_FACTOR = Z_ALPHA + Z_POWER  # ≈ 2.80 — see mdd()

# Below this many informative clusters the percentile cluster bootstrap under-covers;
# the reported CI/MDD read more confident than they are. Warn, don't hide the number.
# 25 is a rule-of-thumb threshold where cluster-bootstrap coverage starts to degrade — not
# a derived constant; tighten it if a pool's calibration ever gets measured.
MIN_CLUSTERS = 25

N_BOOT = 2000


@dataclass
class Sample:
    """One query's contribution to a metric: ``value`` counts only when ``eligible``.

    ``value`` is the per-query metric term (1.0/0.0 hit for success; reciprocal rank for
    MRR); ``eligible`` is the metric's conditioning predicate (goldable / gold-in-pool).
    ``paper_id`` is the cluster key; ``qid`` aligns the two arms of a paired comparison.
    """

    qid: str
    paper_id: str
    eligible: bool
    value: float


@dataclass
class BootResult:
    point: float
    ci_lo: float
    ci_hi: float
    se: float
    n_clusters: int  # papers with >=1 eligible query — the effective sample
    n_eligible: int


@dataclass
class DeltaResult:
    delta: float  # arm_a − arm_b, over queries eligible in BOTH arms
    ci_lo: float
    ci_hi: float
    se: float
    mdd: float
    n_clusters: int
    n_paired: int  # queries eligible in both arms — the paired sample


def success_samples(scores: list[QueryScore]) -> list[Sample]:
    """Per-query terms for ``success@candidates``: eligible iff goldable, 1.0 iff in pool."""
    return [
        Sample(qid=s.qid, paper_id=s.paper_id, eligible=s.goldable, value=float(s.gold_in_pool))
        for s in scores
    ]


def mrr_samples(scores: list[QueryScore], k: int) -> list[Sample]:
    """Per-query terms for ``MRR@k``: eligible iff gold-in-pool, value is reciprocal rank.

    Shares :func:`~eval.metrics.reciprocal_rank` with ``mrr_at_k`` so the bootstrapped mean
    is the same quantity the aggregate reports.
    """
    return [
        Sample(
            qid=s.qid,
            paper_id=s.paper_id,
            eligible=s.gold_in_pool,
            value=reciprocal_rank(s, k),
        )
        for s in scores
    ]


def mdd(se: float, *, factor: float = MDD_FACTOR) -> float:
    """Minimum detectable difference from a paired-delta SE.

    ``factor = z_{1-α/2} + z_{1-β}`` (default ≈ 2.80 for α=0.05, 80% power) — the smallest
    true delta a design with this SE would *reliably detect*, not merely observe once. Pass
    ``factor=Z_ALPHA`` for the plain 95%-CI half-width instead.
    """
    return factor * se


def resolution_warning(n_clusters: int) -> str | None:
    """A one-line caveat when the cluster count is too small for the bootstrap to be trusted."""
    if n_clusters >= MIN_CLUSTERS:
        return None
    return (
        f"n_clusters={n_clusters} < {MIN_CLUSTERS}: the cluster bootstrap under-covers at this "
        f"few papers — the CI/MDD are optimistic; treat rankings as exploratory."
    )


def _group_eligible(samples: list[Sample]) -> tuple[list[str], dict[str, list[float]]]:
    """Papers with >=1 eligible query, and each paper's eligible values. Sorted for determinism."""
    groups: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        if s.eligible:
            groups[s.paper_id].append(s.value)
    return sorted(groups), dict(groups)


def _boot_means(
    papers: list[str], values_by_paper: dict[str, list[float]], *, n_boot: int, seed: int
) -> np.ndarray:
    """Bootstrap distribution of the pooled mean, resampling papers (clusters) with replacement.

    Callers pass only papers with >=1 value, so a resample always pools >=1 value — no
    empty-mean case to guard.
    """
    rng = np.random.default_rng(seed)
    g = len(papers)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, g, size=g)
        pooled = [v for i in idx for v in values_by_paper[papers[i]]]
        boots[b] = np.mean(pooled)
    return boots


def cluster_bootstrap(samples: list[Sample], *, n_boot: int = N_BOOT, seed: int = 0) -> BootResult:
    """Point estimate + percentile CI for one arm's metric, clustered over papers."""
    papers, values_by_paper = _group_eligible(samples)
    n_eligible = sum(len(v) for v in values_by_paper.values())
    if not papers:
        return BootResult(point=0.0, ci_lo=0.0, ci_hi=0.0, se=0.0, n_clusters=0, n_eligible=0)
    point = float(np.mean([v for vs in values_by_paper.values() for v in vs]))
    boots = _boot_means(papers, values_by_paper, n_boot=n_boot, seed=seed)
    ci_lo, ci_hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    se = float(np.std(boots, ddof=1))
    return BootResult(
        point=point,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        se=se,
        n_clusters=len(papers),
        n_eligible=n_eligible,
    )


def paired_delta(
    samples_a: list[Sample], samples_b: list[Sample], *, n_boot: int = N_BOOT, seed: int = 0
) -> DeltaResult:
    """Paired cluster bootstrap of ``arm_a − arm_b`` over queries eligible in BOTH arms.

    Aligns arms by ``qid`` and keeps only queries eligible in both (intersection
    conditioning — see module docstring), then resamples papers over the per-query
    differences. The CI is on the delta; :func:`mdd` turns its SE into the resolution.
    """
    by_qid_a = {s.qid: s for s in samples_a}
    by_qid_b = {s.qid: s for s in samples_b}
    diffs_by_paper: dict[str, list[float]] = defaultdict(list)
    for qid in by_qid_a.keys() & by_qid_b.keys():
        a, b = by_qid_a[qid], by_qid_b[qid]
        if a.eligible and b.eligible:
            diffs_by_paper[a.paper_id].append(a.value - b.value)
    papers = sorted(diffs_by_paper)
    n_paired = sum(len(v) for v in diffs_by_paper.values())
    if not papers:
        return DeltaResult(
            delta=0.0, ci_lo=0.0, ci_hi=0.0, se=0.0, mdd=0.0, n_clusters=0, n_paired=0
        )
    delta = float(np.mean([d for ds in diffs_by_paper.values() for d in ds]))
    boots = _boot_means(papers, diffs_by_paper, n_boot=n_boot, seed=seed)
    ci_lo, ci_hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    se = float(np.std(boots, ddof=1))
    return DeltaResult(
        delta=delta,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        se=se,
        mdd=mdd(se),
        n_clusters=len(papers),
        n_paired=n_paired,
    )
