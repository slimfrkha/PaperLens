"""Optional gen-time triage: discard parametric-knowledge-leaked questions.

An LLM writing a question "from" a paper span has also read well-known papers in
pretraining, and sometimes writes a question answerable from general knowledge rather
than the span. Per harness_plan.md, this is a gen-time filter, not a scorer: ask the
model the *closed-book* question (no span, no gold text) and compare its answer against
the gold `QAItem.answer` with a cheap deterministic overlap score, no second "judge" LLM
call (mirrors metrics.py's no-`ir_measures`-for-exact-operations precedent). Off by
default (harness_plan.md Phase 7) — add only if MDD comes out too high to be useful.

Known limitation, accepted not solved: the gold `answer` is the *generation* LLM's own
paraphrase (not text extracted from the section), and the closed-book call reuses the
same model — so a match can reflect model-style convergence as much as real recall. And
ML papers share dense jargon (attention, FP8, cosine schedule...), so a plausible-but-
generic closed-book guess can token-overlap with the gold answer without the model
actually knowing the specific paper. Neither is fixed by a smarter heuristic here; both
are why every checked item — not just discards — is logged (see cli.py's cmd_gen) so a
threshold is never trusted blind on a new pool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.llm import LLMBackend

from .queryset import QAItem

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "an", "the"}  # matches SQuAD's normalize_answer(); punctuation is
# already excluded by _WORD, so this is the rest of it


@dataclass(frozen=True)
class GenFilterConfig:
    """Off-by-default closed-book leakage filter, frozen like GenConfig (queryset.py) —
    swapping the threshold must never silently change what's in the eval set without a
    visible flag/flag-value recorded in meta.json."""

    enabled: bool = False
    match_threshold: float = 0.5  # UNVALIDATED default — see cli.py's audit log; hand-check
    # a real pool's <fp>.genfilter.jsonl before trusting this


_CLOSED_BOOK_SYSTEM = (
    "Answer the following question about a machine-learning research paper using only "
    "your own knowledge. If you don't know, say so briefly. Respond with just the short "
    "answer, no explanation."
)


def _normalize_tokens(text: str) -> set[str]:
    """Lowercase word tokens, punctuation stripped, stopwords dropped — SQuAD's
    normalize_answer(), token *set* (not multiset): answers here are short phrases, not
    passages, so repeats carry no signal."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def answer_overlap_f1(predicted: str, gold: str) -> float:
    """Token-set F1 between a closed-book answer and the gold short answer.

    F1 (not exact match) because free-text answers paraphrase ("linear attention" vs.
    "a linear attention mechanism") — exact match would undercount real leakage. Not an
    embedding/semantic similarity either — no new dependency or model load, matching
    metrics.py's precedent of exact set operations over a library for a well-defined
    comparison. 0.0 if either side has no tokens (nothing to compare).

    Read the score with the module docstring's caveat in mind: shared ML jargon between
    an unrelated closed-book guess and the gold answer can produce real overlap that
    isn't actual leakage. This function measures overlap, not recall of the paper — the
    caller (check_leak / a hand-audit of the genfilter log) decides what overlap means.
    """
    pred, gold_t = _normalize_tokens(predicted), _normalize_tokens(gold)
    overlap = pred & gold_t
    if not pred or not gold_t or not overlap:
        return 0.0
    precision, recall = len(overlap) / len(pred), len(overlap) / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def closed_book_answer(question: str, llm: LLMBackend) -> str:
    """One completion call, no span/context — the model answers from parametric
    knowledge alone (or admits it can't). Split out so check_leak's fail-open wrapping
    and the scoring logic are each independently testable."""
    return llm.complete(system=_CLOSED_BOOK_SYSTEM, user=question)


@dataclass(frozen=True)
class LeakCheck:
    """The full result of one closed-book check — not just the bool, so the caller can
    log `predicted`/`score` for every item (leaked or not) as an audit/calibration
    trail. See cli.py's cmd_gen and the module docstring's accuracy caveat: `leaked`
    alone hides exactly the false-positive risk this filter is prone to.

    `error` distinguishes *why* a check produced no real signal — "no_gold_answer" vs.
    "llm_error: ..." vs. `None` (a real closed-book attempt, however low the score) — so
    a hand-audit of the genfilter log can tell a genuine low-overlap miss apart from a
    skipped/failed check instead of both collapsing into the same `score=0.0` row.
    """

    predicted: str
    score: float
    leaked: bool
    error: str | None = None


def check_leak(item: QAItem, llm: LLMBackend, threshold: float) -> LeakCheck:
    """Closed-book-answer `item.query`, score it against `item.answer`, decide leaked.

    Fails open: no `item.answer` to compare against, or any exception from the
    closed-book call (timeout, malformed response), keeps the item (`leaked=False`).
    Mirrors iter_queryset's per-item try/except/skip, but inverted — there, failure
    skips generation; here, failure in the *filter* must never discard an otherwise-good
    item on ambiguous evidence. Only a positive score >= threshold discards.

    An LLM-call failure is printed (like iter_queryset's per-section failures) and
    recorded in `LeakCheck.error` — silently returning a bare `False` here would make a
    systematic outage during a `--genfilter` run indistinguishable, in both the console
    output and the audit log, from "this pool has no leaked questions."
    """
    if not item.answer:
        return LeakCheck(predicted="", score=0.0, leaked=False, error="no_gold_answer")
    try:
        predicted = closed_book_answer(item.query, llm)
    except Exception as e:
        print(f"  ! genfilter check failed for {item.paper_id!r}: {e}")
        return LeakCheck(predicted="", score=0.0, leaked=False, error=f"llm_error: {e}")
    score = answer_overlap_f1(predicted, item.answer)
    return LeakCheck(predicted=predicted, score=score, leaked=score >= threshold)
