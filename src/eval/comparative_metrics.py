"""Retrieval metrics for cross-paper comparative questions: strict-AND success, worst-
case (minimum) reciprocal rank across the gold papers.

Kept out of :mod:`metrics` on purpose: that module's docstring makes single-gold-section
claims ("no graded-gain or tie subtlety... so no metrics library") that would quietly stop
being fully true the moment a multi-gold, min-across-papers variant sat next to it.
Keeping the two families in separate files means neither docstring has to hedge.

``per_paper`` only changes what reaches the candidate pool (stage 1) — same framing
:mod:`metrics` uses. What changes here is what counts as *covering* an item's gold, now
that gold is N sections across N papers instead of one:

* **``gold_in_pool`` (strict AND)**: the pool must contain at least one chunk from
  *every* gold paper's section, not a fractional/partial-credit score. A synthesis answer
  missing even one of its required sources is still a wrong answer, and partial credit
  would blur exactly the crowding failure this eval exists to catch. This compounds two
  approximations pulling in opposite directions: "any chunk of the gold section" is
  already a documented upper bound on true relevance (a chunk can carry the right section
  without carrying the specific fact needed — ``docs/harness.md``'s "Section-localization,
  not answer-localization"), so requiring it N times over makes the metric harder to
  satisfy as group size grows, but each individual conjunct stays exactly as loose as it
  always was — a "success" at N=5 means five independently-loose bounds all cleared at
  once, a weaker guarantee than the strict-AND framing suggests on a first read.
* **``comparative_reciprocal_rank`` (minimum across gold papers)**: a synthesis answer's
  usable quality is bottlenecked by its worst-covered source, not its best — if paper A's
  chunk ranks 1st but paper B's ranks 9th (or falls outside the top-k entirely, reciprocal
  rank 0), the item's score is B's, not A's.

Reuses ``eval.stats``'s ``Sample``/``cluster_bootstrap``/``paired_delta`` wholesale (see
their own docstrings) — the two ``Sample``-adapter functions here import ``Sample`` from
there rather than any new bootstrap math, same pattern ``success_samples``/``mrr_samples``
already establish for the single-paper case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import Sample


@dataclass
class ComparativeQueryScore:
    """Everything needed to score one comparative question, retrieval already run.

    ``relevant_ids_by_paper`` maps each gold paper's id to the full set of its gold
    section's chunk ids in the index (from ``eval.metrics.relevant_ids``, called once per
    gold section) — never a single flat set, since coverage must be checked per paper, not
    as a union. ``primary_paper_id`` is the bootstrap cluster key: the paper that appears
    earliest in the item's own ``sections`` order, i.e. the trial's own random paper-draw
    order restricted to the matched subset (deterministic from the trial's seed,
    independent of the LLM's own output ordering).
    """

    qid: str
    candidate_ids: list[str]
    ranked: list[tuple[str, float]] = field(default_factory=list)
    relevant_ids_by_paper: dict[str, set[str]] = field(default_factory=dict)
    primary_paper_id: str = ""

    @property
    def goldable(self) -> bool:
        """Every gold paper's section must be indexed — if even one is missing, no arm
        could ever succeed on this item regardless of per_paper (same reasoning as
        ``QueryScore.goldable``, applied conjunctively)."""
        return bool(self.relevant_ids_by_paper) and all(self.relevant_ids_by_paper.values())

    @property
    def gold_in_pool(self) -> bool:
        """Strict AND: the pool must contain at least one chunk from EVERY gold paper's
        section."""
        cand = set(self.candidate_ids)
        return self.goldable and all(ids & cand for ids in self.relevant_ids_by_paper.values())


def comparative_success_at_candidates(scores: list[ComparativeQueryScore]) -> float:
    """Mirrors ``success_at_candidates`` exactly: fraction of *goldable* items with
    ``gold_in_pool``."""
    goldable = [s for s in scores if s.goldable]
    if not goldable:
        return 0.0
    return sum(s.gold_in_pool for s in goldable) / len(goldable)


def comparative_reciprocal_rank(score: ComparativeQueryScore, k: int) -> float:
    """Minimum reciprocal rank across the gold papers — see module docstring for why."""
    per_paper_rr = [
        next((1.0 / (i + 1) for i, (cid, _) in enumerate(score.ranked[:k]) if cid in ids), 0.0)
        for ids in score.relevant_ids_by_paper.values()
    ]
    return min(per_paper_rr) if per_paper_rr else 0.0


def comparative_mrr_at_k(scores: list[ComparativeQueryScore], k: int) -> float:
    """Averages :func:`comparative_reciprocal_rank` over ``gold_in_pool`` items, same
    conditioning ``mrr_at_k`` already uses (a stage-1 miss isn't charged to the reranker)."""
    rr = [comparative_reciprocal_rank(s, k) for s in scores if s.gold_in_pool]
    return sum(rr) / len(rr) if rr else 0.0


def n_ungoldable_comparative(scores: list[ComparativeQueryScore]) -> int:
    """Mirrors ``n_ungoldable`` exactly: items where even one gold paper's section is
    missing from the index entirely — a generation/indexing artifact, not a retrieval
    miss, reported separately and excluded from
    :func:`comparative_success_at_candidates`'s denominator, never silently folded in."""
    return sum(1 for s in scores if not s.goldable)


def comparative_success_samples(scores: list[ComparativeQueryScore]) -> list[Sample]:
    """Per-item terms for ``comparative_success_at_candidates``, keyed for the bootstrap
    on ``primary_paper_id`` (see :class:`ComparativeQueryScore`), not a real per-paper
    multi-way clustering — a known, accepted approximation for this exploratory eval."""
    return [
        Sample(
            qid=s.qid, paper_id=s.primary_paper_id, eligible=s.goldable, value=float(s.gold_in_pool)
        )
        for s in scores
    ]


def comparative_mrr_samples(scores: list[ComparativeQueryScore], k: int) -> list[Sample]:
    """Per-item terms for ``comparative_mrr_at_k``, same ``primary_paper_id`` clustering
    key as :func:`comparative_success_samples`."""
    return [
        Sample(
            qid=s.qid,
            paper_id=s.primary_paper_id,
            eligible=s.gold_in_pool,
            value=comparative_reciprocal_rank(s, k),
        )
        for s in scores
    ]
