"""Metrics: section-identity relevance, success@candidates, MRR@k.

Section is ONE relevant unit, so both metrics must be chunking-independent — the sharpest
test here (`test_mrr_is_chunking_independent`) pins that property directly.
"""

from __future__ import annotations

from eval.metrics import (
    QueryScore,
    elbow_cutoffs,
    mean_returned_at_elbow,
    mrr_at_k,
    n_conditioned,
    n_ungoldable,
    precision_at_elbow,
    recall_at_elbow,
    relevant_ids,
    success_at_candidates,
)


def test_goldable_and_gold_in_pool():
    goldable_hit = QueryScore(qid="0", candidate_ids=["x", "a"], relevant_ids={"a", "b"})
    goldable_miss = QueryScore(qid="1", candidate_ids=["x", "y"], relevant_ids={"a", "b"})
    ungoldable = QueryScore(qid="2", candidate_ids=["x"], relevant_ids=set())
    assert goldable_hit.goldable and goldable_hit.gold_in_pool
    assert goldable_miss.goldable and not goldable_miss.gold_in_pool
    assert not ungoldable.goldable and not ungoldable.gold_in_pool


def test_success_excludes_ungoldable_from_denominator():
    scores = [
        QueryScore(qid="0", candidate_ids=["a"], relevant_ids={"a"}),  # goldable hit
        QueryScore(qid="1", candidate_ids=["b"], relevant_ids={"z"}),  # goldable miss
        QueryScore(qid="2", candidate_ids=["c"], relevant_ids=set()),  # ungoldable, excluded
    ]
    assert success_at_candidates(scores) == 1 / 2  # 1 hit over 2 goldable, not 3
    assert n_ungoldable(scores) == 1
    assert success_at_candidates([]) == 0.0


def test_mrr_reciprocal_of_first_relevant_rank():
    at1 = QueryScore(qid="0", candidate_ids=["a"], ranked=[("a", 9.0)], relevant_ids={"a"})
    at3 = QueryScore(
        qid="1",
        candidate_ids=["a"],
        ranked=[("x", 3.0), ("y", 2.0), ("a", 1.0)],
        relevant_ids={"a"},
    )
    assert mrr_at_k([at1], k=5) == 1.0
    assert mrr_at_k([at3], k=5) == 1 / 3


def test_mrr_is_chunking_independent():
    # Same real outcome — gold section's first chunk at rank 2 — whether the section split
    # into 1 chunk or 3. MRR must be identical; this is the property nDCG lacked.
    one_chunk = QueryScore(
        qid="0", candidate_ids=["a"], ranked=[("x", 2.0), ("a", 1.0)], relevant_ids={"a"}
    )
    three_chunks = QueryScore(
        qid="1",
        candidate_ids=["a1"],
        ranked=[("x", 2.0), ("a1", 1.0), ("a2", 0.5)],
        relevant_ids={"a1", "a2", "a3"},
    )
    assert mrr_at_k([one_chunk], k=5) == mrr_at_k([three_chunks], k=5) == 1 / 2


def test_mrr_respects_the_k_cutoff():
    # Gold chunk at rank 3 is outside k=2 → no relevant in the cutoff → 0 for that query.
    s = QueryScore(
        qid="0",
        candidate_ids=["a"],
        ranked=[("x", 3.0), ("y", 2.0), ("a", 1.0)],
        relevant_ids={"a"},
    )
    assert mrr_at_k([s], k=2) == 0.0


def test_mrr_conditioned_on_gold_in_pool():
    # A stage-1 miss (gold not in pool) is excluded from the MRR average, not scored 0 —
    # else a recall miss would be charged to the reranker.
    hit = QueryScore(qid="0", candidate_ids=["a"], ranked=[("a", 1.0)], relevant_ids={"a"})
    miss = QueryScore(qid="1", candidate_ids=["x"], ranked=[("x", 1.0)], relevant_ids={"z"})
    scores = [hit, miss]
    assert n_conditioned(scores) == 1
    assert mrr_at_k(scores, k=5) == 1.0  # averaged over the 1 conditioned query only
    assert mrr_at_k([miss], k=5) == 0.0  # nobody conditioned → 0.0


# --- Additive elbow-cutoff metrics --------------------------------------------------------


def test_elbow_cutoffs_conditions_on_gold_in_pool():
    # A stage-1 miss (gold not in pool) must be excluded, same conditioning as MRR@k — an
    # elbow cutoff can't be charged with a recall failure that already happened upstream.
    hit = QueryScore(
        qid="0",
        candidate_ids=["a", "x"],
        ranked=[("a", 0.95), ("x", 0.40), ("y", 0.38)],
        relevant_ids={"a"},
        paper_id="p1",
    )
    miss = QueryScore(
        qid="1", candidate_ids=["x"], ranked=[("x", 0.9)], relevant_ids={"z"}, paper_id="p2"
    )
    cutoffs = elbow_cutoffs([hit, miss], min_k=1, max_k=3, mad_multiplier=3.0, prominence=0.15)
    assert [s.qid for s, _, _ in cutoffs] == ["0"]


def test_elbow_cutoffs_records_whether_gold_survives_the_cut():
    # Same score shape (one clean cliff after rank 1) for both queries; only whether the
    # gold chunk id sits inside the elbow-cut prefix differs.
    survives = QueryScore(
        qid="0",
        candidate_ids=["a"],
        ranked=[("a", 0.95), ("x", 0.40), ("y", 0.38)],
        relevant_ids={"a"},
        paper_id="p1",
    )
    starved = QueryScore(
        qid="1",
        candidate_ids=["a"],
        ranked=[("x", 0.95), ("y", 0.40), ("a", 0.38)],
        relevant_ids={"a"},
        paper_id="p2",
    )
    cutoffs = elbow_cutoffs(
        [survives, starved], min_k=1, max_k=3, mad_multiplier=3.0, prominence=0.15
    )
    hits = {s.qid: hit for s, _, hit in cutoffs}
    assert hits == {"0": True, "1": False}


def test_mean_returned_precision_recall_at_elbow():
    survives = QueryScore(
        qid="0",
        candidate_ids=["a"],
        ranked=[("a", 0.95), ("x", 0.40), ("y", 0.38)],
        relevant_ids={"a"},
        paper_id="p1",
    )
    starved = QueryScore(
        qid="1",
        candidate_ids=["a"],
        ranked=[("x", 0.95), ("y", 0.40), ("a", 0.38)],
        relevant_ids={"a"},
        paper_id="p2",
    )
    cutoffs = elbow_cutoffs(
        [survives, starved], min_k=1, max_k=3, mad_multiplier=3.0, prominence=0.15
    )
    assert mean_returned_at_elbow(cutoffs) == 1.0  # both cut to a single chunk
    assert recall_at_elbow(cutoffs) == 0.5  # only "survives" kept its gold chunk
    assert precision_at_elbow(cutoffs) == 0.5  # (1/1 + 0/1) / 2


def test_elbow_metrics_empty_input_returns_zero():
    assert mean_returned_at_elbow([]) == 0.0
    assert recall_at_elbow([]) == 0.0
    assert precision_at_elbow([]) == 0.0


def test_relevant_ids_returns_all_chunks_of_the_gold_section(make_searcher, seed_chunks):
    docs = [
        seed_chunks("p1", "Method", "latent attention cache", doc_id="p1-method-0"),
        seed_chunks("p1", "Method", "low rank projection", doc_id="p1-method-1"),
        seed_chunks("p1", "Training", "fp8 precision", doc_id="p1-train-0"),
        seed_chunks("p2", "Method", "different paper", doc_id="p2-method-0"),
    ]
    ctx = make_searcher(docs)
    got = relevant_ids(ctx.collection, "p1", "1", "Method")  # seed_chunks uses section_number "1"
    assert got == {"p1-method-0", "p1-method-1"}
