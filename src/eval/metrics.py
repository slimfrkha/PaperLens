"""Retrieval metrics for a single config run: stage-1 success@candidates + stage-2 MRR@k.

**Relevance is section identity, and the section is ONE relevant unit — not N (its chunks).**
A question is answerable from its gold section; splitting that section into more chunks must
not change the score, or a chunking sweep would rank arms by how finely they chunk rather
than by retrieval quality. So both metrics collapse the section to a single target and stay
chunking-independent:

* **Stage 1 — ``success@candidates``**: did *any* chunk of the gold section reach the dense
  pool? Averaged over *goldable* questions only. A question whose gold section produced zero
  chunks in the index — the gold generator (``iter_sections``) and the chunker
  (``chunk_markdown``) apply slightly different substantive-section filters, so they disagree
  on a few sections — is a generation artifact, not a retrieval miss: it is excluded from the
  denominator and reported separately (:func:`n_ungoldable`).
* **Stage 2 — ``MRR@k``**: reciprocal rank of the gold section's first chunk in the reranked
  top-k, **conditioned** on the section having reached the pool (else a stage-1 miss would be
  charged to the reranker). One relevant unit ⇒ the number of chunks the section split into
  cannot move the score.

Both are exact set / rank operations over section identity — no graded-gain or tie subtlety —
so no metrics library. (When graded relevance returns — multi-question sections, or window
gold scored by overlap fraction, both deferred — reach for ``ir_measures`` then, not now.)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryScore:
    """Everything needed to score one question, retrieval already run.

    ``candidate_ids`` is the dense pool (pre-rerank, length ``<= candidates``);
    ``ranked`` is the reranked top-k as ``(chunk_id, score)``; ``relevant_ids`` is the
    full set of gold-section chunk ids in the index (from :func:`relevant_ids`), which is
    empty iff the gold section is absent from the index (ungoldable).
    """

    qid: str
    candidate_ids: list[str]
    ranked: list[tuple[str, float]] = field(default_factory=list)
    relevant_ids: set[str] = field(default_factory=set)
    paper_id: str = ""  # cluster key for the paper-clustered bootstrap (stats.py)

    @property
    def goldable(self) -> bool:
        """Does the gold section exist in the index at all? If not, no arm can retrieve it."""
        return bool(self.relevant_ids)

    @property
    def gold_in_pool(self) -> bool:
        """Did a gold-section chunk reach the dense candidate pool (stage-1 hit)?"""
        return bool(self.relevant_ids & set(self.candidate_ids))


def relevant_ids(collection, paper_id: str, section_number: str, section_title: str) -> set[str]:
    """All chunk ids in ``collection`` belonging to the gold section.

    Query-independent ground truth (a metadata filter, not a search). ``section_number`` is
    matched as stored by ``rag.index`` — "" for unnumbered sections, never ``None``.
    """
    res = collection.get(
        where={
            "$and": [
                {"paper_id": {"$eq": paper_id}},
                {"section_number": {"$eq": section_number}},
                {"section_title": {"$eq": section_title}},
            ]
        }
    )
    return set(res["ids"])


def success_at_candidates(scores: list[QueryScore]) -> float:
    """Stage-1 ceiling: fraction of *goldable* questions whose gold section reached the pool.

    Ungoldable questions (gold section absent from the index) are excluded from the
    denominator — they measure a generation/indexing gap, not retrieval — see
    :func:`n_ungoldable`.
    """
    goldable = [s for s in scores if s.goldable]
    if not goldable:
        return 0.0
    return sum(s.gold_in_pool for s in goldable) / len(goldable)


def reciprocal_rank(score: QueryScore, k: int) -> float:
    """Reciprocal rank of the gold section's first chunk in the reranked top-k (0.0 if absent).

    The single source of truth for the rank term, shared by :func:`mrr_at_k` and the
    per-query decomposition the bootstrap resamples (``stats.mrr_samples``) — they must
    report the same quantity, so they must not compute it twice.
    """
    rank = next(
        (i + 1 for i, (cid, _) in enumerate(score.ranked[:k]) if cid in score.relevant_ids), None
    )
    return 1.0 / rank if rank else 0.0


def mrr_at_k(scores: list[QueryScore], k: int) -> float:
    """Stage-2 ``MRR@k``: reciprocal rank of the gold section's first chunk in the reranked
    top-k, averaged over questions whose gold section reached the pool (:func:`n_conditioned`).

    One relevant unit (the section), so the count of chunks it split into cannot move the
    score — the property a chunking sweep needs. Returns 0.0 if no question qualifies.
    """
    rr = [reciprocal_rank(s, k) for s in scores if s.gold_in_pool]
    return sum(rr) / len(rr) if rr else 0.0


def n_conditioned(scores: list[QueryScore]) -> int:
    """Questions whose gold section reached the pool — MRR@k's denominator."""
    return sum(s.gold_in_pool for s in scores)


def n_ungoldable(scores: list[QueryScore]) -> int:
    """Questions whose gold section has zero chunks in the index — a generation artifact,
    not a retrieval miss. Reported, never hidden in the success@candidates denominator."""
    return sum(1 for s in scores if not s.goldable)
