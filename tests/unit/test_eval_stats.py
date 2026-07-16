"""Phase 3 stats: paper-clustered paired bootstrap, delta CIs, and MDD.

Pure numeric tests on hand-built ``Sample`` lists — no Chroma, no models. The properties
that matter and are easy to get silently wrong: the bootstrap clusters on *papers* (not
queries), the paired delta conditions on the *intersection* of eligible queries, a known
gap is recovered with a CI clear of zero, a known null straddles zero, and results are
deterministic under a fixed seed.
"""

from __future__ import annotations

import pytest

from eval.metrics import QueryScore
from eval.stats import (
    MDD_FACTOR,
    Z_ALPHA,
    Sample,
    cluster_bootstrap,
    mdd,
    mrr_samples,
    paired_delta,
    resolution_warning,
    success_samples,
)


def _arm(diffs_base: float, spread: float, n: int, side: int) -> list[Sample]:
    """One arm over ``n`` papers (1 query each). Arm A is ``base`` above arm B, ± spread."""
    out: list[Sample] = []
    for i in range(n):
        # per-paper diff alternates base±spread so the paired SE is nonzero (not degenerate)
        d = diffs_base + (spread if i % 2 else -spread)
        val = 0.5 + (d / 2 if side > 0 else -d / 2)
        out.append(Sample(qid=f"p{i}", paper_id=f"p{i}", eligible=True, value=val))
    return out


def test_paired_delta_recovers_a_known_gap_with_ci_clear_of_zero():
    a = _arm(0.3, 0.05, 20, side=1)
    b = _arm(0.3, 0.05, 20, side=-1)
    res = paired_delta(a, b, seed=0)
    assert res.delta == pytest.approx(0.3, abs=1e-9)
    assert res.ci_lo > 0.0  # the whole interval is above zero → a real, resolvable gap
    assert res.n_clusters == 20
    assert res.n_paired == 20
    assert res.se > 0.0
    assert res.mdd == pytest.approx(MDD_FACTOR * res.se)


def test_paired_delta_on_a_null_straddles_zero():
    # Same query set, per-paper differences symmetric around 0 → no real effect.
    a: list[Sample] = []
    b: list[Sample] = []
    for i in range(20):
        d = 0.2 if i % 2 else -0.2
        a.append(Sample(qid=f"p{i}", paper_id=f"p{i}", eligible=True, value=0.5 + d))
        b.append(Sample(qid=f"p{i}", paper_id=f"p{i}", eligible=True, value=0.5))
    res = paired_delta(a, b, seed=0)
    assert res.delta == pytest.approx(0.0, abs=1e-9)
    assert res.ci_lo < 0.0 < res.ci_hi  # straddles zero → nothing to resolve


def test_bootstrap_clusters_on_papers_not_queries():
    # Identical value spread, once inside a single paper, once across 20 papers. Resampling
    # papers leaves the 1-paper case with no between-cluster variance (SE 0) while the
    # 20-paper case has real variance — the whole point of clustering on the document.
    vals = [i / 20 for i in range(20)]
    one = [Sample(qid=str(i), paper_id="solo", eligible=True, value=v) for i, v in enumerate(vals)]
    many = [
        Sample(qid=str(i), paper_id=f"p{i}", eligible=True, value=v) for i, v in enumerate(vals)
    ]

    r_one = cluster_bootstrap(one, seed=0)
    r_many = cluster_bootstrap(many, seed=0)

    assert r_one.n_clusters == 1
    assert r_many.n_clusters == 20
    assert r_one.point == pytest.approx(r_many.point)  # same underlying mean
    assert r_one.se == pytest.approx(0.0, abs=1e-12)  # one cluster → every resample identical
    assert r_many.se > 1e-3


def test_paired_delta_conditions_on_the_intersection():
    # q1 is eligible in A but not B; it must drop out of the paired sample entirely, so the
    # delta is computed only over q0 (not charged the arms' differing denominators).
    a = [
        Sample(qid="q0", paper_id="p0", eligible=True, value=1.0),
        Sample(qid="q1", paper_id="p1", eligible=True, value=1.0),
    ]
    b = [
        Sample(qid="q0", paper_id="p0", eligible=True, value=0.0),
        Sample(qid="q1", paper_id="p1", eligible=False, value=0.0),
    ]
    res = paired_delta(a, b, seed=0)
    assert res.n_paired == 1
    assert res.n_clusters == 1
    assert res.delta == pytest.approx(1.0)


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    many = [Sample(qid=str(i), paper_id=f"p{i}", eligible=True, value=i / 20) for i in range(20)]
    assert cluster_bootstrap(many, seed=0) == cluster_bootstrap(many, seed=0)
    assert cluster_bootstrap(many, seed=0).ci_lo != cluster_bootstrap(many, seed=1).ci_lo


def test_ineligible_queries_are_excluded_from_the_estimate():
    samples = [
        Sample(qid="0", paper_id="p0", eligible=True, value=1.0),
        Sample(qid="1", paper_id="p0", eligible=False, value=0.0),  # excluded
    ]
    res = cluster_bootstrap(samples, seed=0)
    assert res.n_eligible == 1
    assert res.point == pytest.approx(1.0)


def test_mdd_scales_with_se_and_factor():
    assert mdd(0.1) == pytest.approx(MDD_FACTOR * 0.1)
    assert mdd(0.2) > mdd(0.1)  # monotone in SE
    assert pytest.approx(2.80, abs=0.01) == MDD_FACTOR  # α=0.05 + 80% power
    assert mdd(0.1, factor=Z_ALPHA) == pytest.approx(Z_ALPHA * 0.1)  # plain 95% half-width


def test_resolution_warning_fires_only_below_the_threshold():
    assert resolution_warning(11) is not None
    assert resolution_warning(25) is None


def _qs(qid: str, paper: str, ranked, relevant, candidates=None) -> QueryScore:
    return QueryScore(
        qid=qid,
        candidate_ids=candidates if candidates is not None else [c for c, _ in ranked],
        ranked=ranked,
        relevant_ids=set(relevant),
        paper_id=paper,
    )


def test_success_samples_track_goldable_and_in_pool():
    hit = _qs("0", "p1", [("c1", 0.9)], {"c1"})
    miss = _qs("1", "p1", [("x", 0.9)], {"c1"}, candidates=["x"])  # gold section absent from pool
    ungoldable = _qs("2", "p1", [("x", 0.9)], set(), candidates=["x"])  # no gold in index

    s = {x.qid: x for x in success_samples([hit, miss, ungoldable])}
    assert s["0"].eligible and s["0"].value == 1.0 and s["0"].paper_id == "p1"
    assert s["1"].eligible and s["1"].value == 0.0  # goldable but missed the pool
    assert not s["2"].eligible  # ungoldable → excluded from the denominator


def test_mrr_samples_give_reciprocal_rank_conditioned_on_pool():
    rank2 = _qs("0", "p1", [("x", 1.0), ("c1", 0.9)], {"c1"})  # gold at rank 2
    absent = _qs("1", "p1", [("x", 1.0)], {"c1"}, candidates=["x"])  # not in pool → ineligible

    m = {x.qid: x for x in mrr_samples([rank2, absent], k=5)}
    assert m["0"].eligible and m["0"].value == pytest.approx(0.5)
    assert not m["1"].eligible
