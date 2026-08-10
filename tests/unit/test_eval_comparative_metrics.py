"""Comparative metrics: strict-AND success, worst-case (minimum) reciprocal rank across
gold papers.

The sharpest tests here pin the two departures from the single-paper metrics directly:
`test_gold_in_pool_requires_every_paper` (AND, not OR/union) and
`test_reciprocal_rank_is_the_minimum_across_papers` (worst-case, not best-case).
"""

from __future__ import annotations

from eval.comparative_metrics import (
    ComparativeQueryScore,
    comparative_mrr_at_k,
    comparative_mrr_samples,
    comparative_reciprocal_rank,
    comparative_success_at_candidates,
    comparative_success_samples,
    n_ungoldable_comparative,
)


def test_goldable_requires_every_paper_indexed():
    both_indexed = ComparativeQueryScore(
        qid="0", candidate_ids=[], relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}}
    )
    one_missing = ComparativeQueryScore(
        qid="1", candidate_ids=[], relevant_ids_by_paper={"a": {"a1"}, "b": set()}
    )
    none_indexed = ComparativeQueryScore(qid="2", candidate_ids=[], relevant_ids_by_paper={})
    assert both_indexed.goldable
    assert not one_missing.goldable  # paper b's gold section absent from the index
    assert not none_indexed.goldable


def test_gold_in_pool_requires_every_paper_not_just_one():
    # Strict AND: paper a's chunk is in the pool, paper b's is not -> not gold_in_pool,
    # even though a plain union of relevant ids would count this as a partial hit.
    partial = ComparativeQueryScore(
        qid="0",
        candidate_ids=["a1", "x"],
        relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}},
    )
    full = ComparativeQueryScore(
        qid="1",
        candidate_ids=["a1", "b1"],
        relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}},
    )
    assert not partial.gold_in_pool
    assert full.gold_in_pool


def test_success_excludes_ungoldable_from_denominator():
    scores = [
        ComparativeQueryScore(
            qid="0", candidate_ids=["a1", "b1"], relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}}
        ),  # goldable, full hit
        ComparativeQueryScore(
            qid="1", candidate_ids=["a1"], relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}}
        ),  # goldable, partial hit -> miss
        ComparativeQueryScore(
            qid="2", candidate_ids=["x"], relevant_ids_by_paper={"a": {"a1"}, "b": set()}
        ),  # ungoldable (paper b missing), excluded
    ]
    assert comparative_success_at_candidates(scores) == 1 / 2  # 1 hit over 2 goldable
    assert n_ungoldable_comparative(scores) == 1
    assert comparative_success_at_candidates([]) == 0.0


def test_reciprocal_rank_is_the_minimum_across_papers():
    # paper a's chunk ranks 1st (rr=1.0), paper b's ranks 3rd (rr=1/3) -> the item's score
    # is bottlenecked by b, not a's better rank.
    s = ComparativeQueryScore(
        qid="0",
        candidate_ids=["a1", "x", "b1"],
        ranked=[("a1", 3.0), ("x", 2.0), ("b1", 1.0)],
        relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}},
    )
    assert comparative_reciprocal_rank(s, k=5) == 1 / 3


def test_reciprocal_rank_zero_when_any_paper_falls_outside_k():
    # paper a's chunk is in the top-k, paper b's exists in the pool but outside the k
    # cutoff -> b's reciprocal rank is 0, and the min forces the whole item to 0.
    s = ComparativeQueryScore(
        qid="0",
        candidate_ids=["a1", "b1"],
        ranked=[("a1", 3.0), ("x", 2.0), ("b1", 1.0)],
        relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}},
    )
    assert comparative_reciprocal_rank(s, k=1) == 0.0  # only a1 survives the k=1 cutoff


def test_mrr_conditioned_on_gold_in_pool():
    hit = ComparativeQueryScore(
        qid="0",
        candidate_ids=["a1", "b1"],
        ranked=[("a1", 2.0), ("b1", 1.0)],
        relevant_ids_by_paper={"a": {"a1"}, "b": {"b1"}},
    )  # both gold papers at ranks 1 and 2 -> min(1.0, 0.5) = 0.5
    miss = ComparativeQueryScore(
        qid="1", candidate_ids=["x"], ranked=[("x", 1.0)], relevant_ids_by_paper={"a": {"a1"}}
    )  # ungoldable (paper b never even indexed) -> excluded, not scored 0
    assert comparative_mrr_at_k([hit, miss], k=5) == 0.5  # averaged over gold_in_pool only
    assert comparative_mrr_at_k([miss], k=5) == 0.0  # nobody conditioned -> 0.0


def test_success_and_mrr_samples_key_on_primary_paper_id():
    s = ComparativeQueryScore(
        qid="0",
        candidate_ids=["a1"],
        ranked=[("a1", 1.0)],
        relevant_ids_by_paper={"a": {"a1"}},
        primary_paper_id="b",  # not even one of this item's own gold papers -- must still
        # be respected as-is, the cluster key is whatever the caller set it to
    )
    success_samples = comparative_success_samples([s])
    mrr_samples = comparative_mrr_samples([s], k=5)
    assert success_samples[0].paper_id == "b"
    assert mrr_samples[0].paper_id == "b"
    assert success_samples[0].eligible and success_samples[0].value == 1.0
    assert mrr_samples[0].eligible and mrr_samples[0].value == 1.0
