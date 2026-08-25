"""Closed-book leakage filter. answer_overlap_f1 is pure and gets exact-value
coverage; check_leak's fail-open behavior (no gold answer, LLM error) is the sharpest thing
to pin, since a filter that discards on ambiguous evidence would be worse than no filter.
Per the Slim Shady review, the F1 heuristic has a known false-positive risk on ML's shared
jargon — these tests pin the *mechanism*'s correctness (does it do what it says), not that
the mechanism is a good leakage detector on
real papers (that needs a hand-audit of a live <fp>.genfilter.jsonl, not a unit test)."""

from __future__ import annotations

import pytest

from eval.genfilter import LeakCheck, answer_overlap_f1, check_leak
from eval.queryset import QAItem
from rag.llm import LLMBackend


def _item(query: str = "What method is used?", answer: str = "") -> QAItem:
    return QAItem(
        query=query,
        paper_id="p",
        gold_span=(0, 1),
        source_unit="1 Method",
        section_number="1",
        section_title="Method",
        answer=answer,
    )


def test_answer_overlap_f1_identical_answers_is_one():
    assert answer_overlap_f1("low-rank latent space", "low-rank latent space") == 1.0


def test_answer_overlap_f1_partial_overlap_hand_computed():
    # predicted tokens (stopwords stripped) = {linear, attention, mechanism} (3)
    # gold tokens = {linear, attention} (2); overlap = 2
    # precision = 2/3, recall = 2/2 = 1.0, F1 = 2*(2/3)*1 / (2/3 + 1) = 0.8
    score = answer_overlap_f1("a linear attention mechanism", "linear attention")
    assert score == pytest.approx(0.8)


def test_answer_overlap_f1_no_overlap_is_zero():
    assert answer_overlap_f1("gradient descent", "rotary embeddings") == 0.0


@pytest.mark.parametrize("predicted,gold", [("", "linear attention"), ("linear attention", "")])
def test_answer_overlap_f1_empty_predicted_or_gold_is_zero(predicted, gold):
    assert answer_overlap_f1(predicted, gold) == 0.0


def test_answer_overlap_f1_stopwords_excluded():
    # Both sides reduce to the empty token set after stopword stripping.
    assert answer_overlap_f1("a", "the") == 0.0


def test_check_leak_flags_matching_closed_book_answer(fake_llm):
    llm = fake_llm(answer="low-rank latent space")
    item = _item(answer="low-rank latent space")
    check = check_leak(item, llm, threshold=0.5)
    assert check == LeakCheck(predicted="low-rank latent space", score=1.0, leaked=True)


def test_check_leak_survives_non_matching_closed_book_answer(fake_llm):
    llm = fake_llm(answer="I don't know")
    item = _item(answer="low-rank latent space")
    check = check_leak(item, llm, threshold=0.5)
    assert check.leaked is False


def test_check_leak_skips_call_when_item_has_no_gold_answer(fake_llm):
    llm = fake_llm(answer="anything")
    item = _item(answer="")
    check = check_leak(item, llm, threshold=0.5)
    assert check == LeakCheck(predicted="", score=0.0, leaked=False, error="no_gold_answer")
    assert llm.complete_calls == []  # no wasted call — nothing to compare against


class _BoomLLM(LLMBackend):
    """A backend whose completion always fails — check_leak must fail open, not raise."""

    def __init__(self) -> None:  # no spec / client needed
        pass

    def complete(self, system, user, max_tokens=None):
        raise RuntimeError("server down")

    def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
        raise RuntimeError("server down")

    def run_tools(self, *args, **kwargs):
        return ""


def test_check_leak_fails_open_on_llm_error(capsys):
    item = _item(answer="low-rank latent space")
    check = check_leak(item, _BoomLLM(), threshold=0.5)
    assert check == LeakCheck(predicted="", score=0.0, leaked=False, error="llm_error: server down")
    # The failure must be visible somewhere — a silent fail-open here would make a
    # systematic LLM outage indistinguishable from "nothing leaked" in both the console
    # and (via `error`, asserted above) the audit log.
    assert "genfilter check failed" in capsys.readouterr().out


def test_check_leak_respects_threshold(fake_llm):
    # "a linear attention mechanism" vs "linear attention" -> F1 = 0.8 (see the hand-
    # computed test above), so 0.5 and 0.9 straddle it.
    llm = fake_llm(answer="a linear attention mechanism")
    item = _item(answer="linear attention")
    assert check_leak(item, llm, threshold=0.5).leaked is True
    assert check_leak(item, llm, threshold=0.9).leaked is False
