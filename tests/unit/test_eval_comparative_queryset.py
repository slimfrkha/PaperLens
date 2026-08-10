"""Comparative eval-set generation: the trial loop (offline, scripted LLM)."""

from __future__ import annotations

import json
import random

import pytest

from eval.comparative_queryset import (
    ComparativeGenConfig,
    ComparativeQAItem,
    build_comparative_queryset,
    comparative_item_from_dict,
    comparative_item_to_dict,
    load_comparative_queryset,
)
from rag.config import AnthropicSpec
from rag.llm import LLMBackend

# Real-shaped bodies (>=24 approx tokens, ~1.33/word) so iter_sections' min_section_tokens
# default doesn't filter them out -- no need to override it per test.
_BODY_A = (
    "We use latent attention to compress the key-value cache substantially over many "
    "tokens for meaningful efficiency gains across long-context inference workloads."
)
_BODY_B = (
    "We use grouped query attention to compress the key-value cache moderately over "
    "many tokens for meaningful efficiency gains across long-context inference workloads."
)
_BODY_C = (
    "Benchmark scores show strong accuracy across many evaluation tasks and settings "
    "under a wide range of conditions and configurations for thorough comparison."
)


def _paper(title: str, number: str, section_title: str, body: str) -> str:
    return f"## {title}\n\nAuthors\n\n## {number}. {section_title}\n\n{body}\n"


POOL = {
    "paper-a": _paper("Paper A", "2", "Method", _BODY_A),
    "paper-b": _paper("Paper B", "2", "Method", _BODY_B),
    "paper-c": _paper("Paper C", "2", "Results", _BODY_C),
}

# n_papers_min=n_papers_max=3 forces every trial to draw all 3 pool papers -- removes
# n_papers randomness so a test only has to reason about the spotting/writing responses.
_GEN_3 = ComparativeGenConfig(target_p=1, max_trials=1, n_papers_min=3, n_papers_max=3, seed=0)


class _ScriptedLLM(LLMBackend):
    """Returns each of ``answers`` in call order -- the spotting and writing calls need
    different structured outputs, unlike ``fake_llm``'s single fixed answer. Mirrors the
    local-subclass precedent in ``test_eval_cli.py``'s ``_ScriptedLLM``."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(AnthropicSpec())
        self._answers = list(answers)
        self.calls: list[str] = []

    def complete(self, system, user, max_tokens=None):
        self.calls.append(user)
        return self._answers.pop(0)

    def run_tools(self, *args, **kwargs):
        raise NotImplementedError


class _BoomLLM(LLMBackend):
    """A backend whose completion always fails -- exercises per-trial isolation, mirrors
    ``test_eval_queryset.py``'s ``_BoomLLM``."""

    def __init__(self) -> None:  # no spec/client needed
        pass

    def complete(self, system, user, max_tokens=None):
        raise RuntimeError("server down")

    def run_tools(self, *args, **kwargs):
        raise NotImplementedError


_MATCH_AB = (
    '[[{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}, '
    '{"paper_id": "paper-b", "section_number": "2", "section_title": "Method"}]]'
)
_MATCH_TWO_GROUPS = (
    '[[{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}, '
    '{"paper_id": "paper-b", "section_number": "2", "section_title": "Method"}], '
    '[{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}, '
    '{"paper_id": "paper-c", "section_number": "2", "section_title": "Results"}]]'
)
_WRITE_OK = '{"question": "How do papers A and B differ?", "answer": "A vs B"}'
_NO_MATCH = "[]"


def test_build_comparative_queryset_yields_one_item_per_match_group():
    llm = _ScriptedLLM([_MATCH_AB, _WRITE_OK])
    result = build_comparative_queryset(POOL, llm, _GEN_3)

    assert len(result.items) == 1
    assert result.trials == 1
    assert {s.paper_id for s in result.items[0].sections} == {"paper-a", "paper-b"}
    assert result.paper_pair_frequency[("paper-a", "paper-b")] == 1


def test_one_trial_multiple_match_groups_yields_multiple_items():
    llm = _ScriptedLLM([_MATCH_TWO_GROUPS, _WRITE_OK, _WRITE_OK])
    gen = ComparativeGenConfig(target_p=10, max_trials=1, n_papers_min=3, n_papers_max=3, seed=0)
    result = build_comparative_queryset(POOL, llm, gen)

    assert len(result.items) == 2  # one trial, one spotting call, two match groups
    assert result.trials == 1


def test_target_p_is_a_soft_floor_not_a_hard_cap():
    # target_p=1 but this one trial finds 2 groups -- both must be kept, never truncated.
    llm = _ScriptedLLM([_MATCH_TWO_GROUPS, _WRITE_OK, _WRITE_OK])
    result = build_comparative_queryset(POOL, llm, _GEN_3)  # target_p=1

    assert len(result.items) == 2  # overshoots target_p=1
    assert result.trials == 1  # loop stops after this trial (already >= target_p)


def test_hallucinated_section_reference_is_discarded():
    match_bad = (
        '[[{"paper_id": "paper-a", "section_number": "99", "section_title": "Nonexistent"}, '
        '{"paper_id": "paper-b", "section_number": "2", "section_title": "Method"}]]'
    )
    llm = _ScriptedLLM([match_bad])  # no writing call -- the group is discarded first
    result = build_comparative_queryset(POOL, llm, _GEN_3)

    assert result.items == []
    assert result.trials == 1


def test_group_referencing_same_paper_twice_is_discarded():
    match_dup = (
        '[[{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}, '
        '{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}]]'
    )
    llm = _ScriptedLLM([match_dup])
    result = build_comparative_queryset(POOL, llm, _GEN_3)
    assert result.items == []


def test_group_with_fewer_than_two_papers_is_discarded():
    match_single = '[[{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}]]'
    llm = _ScriptedLLM([match_single])
    result = build_comparative_queryset(POOL, llm, _GEN_3)
    assert result.items == []


def test_unparseable_spot_response_skips_the_trial():
    llm = _ScriptedLLM(["not json at all"])
    result = build_comparative_queryset(POOL, llm, _GEN_3)
    assert result.items == []
    assert result.trials == 1


def test_max_trials_bounds_the_loop_when_nothing_matches():
    llm = _ScriptedLLM([_NO_MATCH] * 5)
    gen = ComparativeGenConfig(target_p=10, max_trials=5, n_papers_min=2, n_papers_max=2, seed=0)
    result = build_comparative_queryset(POOL, llm, gen)

    assert result.items == []
    assert result.trials == 5


def test_spotting_call_failure_is_isolated_per_trial():
    # A crashing LLM must not sink the whole run -- one bad trial is logged and skipped,
    # same tolerance iter_queryset already has for a per-section failure.
    gen = ComparativeGenConfig(target_p=1, max_trials=3, n_papers_min=2, n_papers_max=2, seed=0)
    result = build_comparative_queryset(POOL, _BoomLLM(), gen)
    assert result.items == []
    assert result.trials == 3  # ran to max_trials, never raised


def test_pool_smaller_than_n_papers_min_raises():
    gen = ComparativeGenConfig(n_papers_min=5)
    with pytest.raises(SystemExit, match="n_papers_min"):
        build_comparative_queryset(POOL, _ScriptedLLM([]), gen)


def test_inverted_n_papers_range_raises_cleanly_not_a_bare_randint_crash():
    # Regression guard: cmd_comparative_gen checks this at the CLI boundary, but
    # build_comparative_queryset is a public, directly callable entry point too (tests,
    # scripts) -- without its own check here, this exact config would instead crash with
    # a bare "ValueError: empty range for randrange()" from random.randint deep in the
    # trial loop, not a clear message at the top.
    gen = ComparativeGenConfig(n_papers_min=6, n_papers_max=2)
    with pytest.raises(ValueError, match="n_papers_min"):
        build_comparative_queryset(POOL, _ScriptedLLM([]), gen)


def test_sections_ordered_by_trial_draw_not_llm_output_order():
    match_b_first = (
        '[[{"paper_id": "paper-b", "section_number": "2", "section_title": "Method"}, '
        '{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}]]'
    )
    match_a_first = (
        '[[{"paper_id": "paper-a", "section_number": "2", "section_title": "Method"}, '
        '{"paper_id": "paper-b", "section_number": "2", "section_title": "Method"}]]'
    )
    result_b_first = build_comparative_queryset(
        POOL, _ScriptedLLM([match_b_first, _WRITE_OK]), _GEN_3
    )
    result_a_first = build_comparative_queryset(
        POOL, _ScriptedLLM([match_a_first, _WRITE_OK]), _GEN_3
    )

    order_b_first = [s.paper_id for s in result_b_first.items[0].sections]
    order_a_first = [s.paper_id for s in result_a_first.items[0].sections]
    # Same trial seed -> same draw order -> same sections[0], regardless of which order
    # the LLM happened to list the match group in.
    assert order_b_first == order_a_first
    # Sanity: this really is the trial's own draw order, not always alphabetical/insertion.
    assert order_b_first == [
        pid for pid in random.Random(0).sample(sorted(POOL), 3) if pid in {"paper-a", "paper-b"}
    ]


def test_on_item_callback_fires_per_item_as_produced():
    llm = _ScriptedLLM([_MATCH_TWO_GROUPS, _WRITE_OK, _WRITE_OK])
    gen = ComparativeGenConfig(target_p=10, max_trials=1, n_papers_min=3, n_papers_max=3, seed=0)
    seen: list[ComparativeQAItem] = []

    result = build_comparative_queryset(POOL, llm, gen, on_item=seen.append)

    assert seen == result.items  # called for every item, in the same order


def test_comparative_item_round_trips_through_dict():
    llm = _ScriptedLLM([_MATCH_AB, _WRITE_OK])
    item = build_comparative_queryset(POOL, llm, _GEN_3).items[0]

    loaded = comparative_item_from_dict(json.loads(json.dumps(comparative_item_to_dict(item))))

    assert loaded.query == item.query
    assert loaded.answer == item.answer
    assert [(s.paper_id, s.number, s.title, s.start, s.end) for s in loaded.sections] == [
        (s.paper_id, s.number, s.title, s.start, s.end) for s in item.sections
    ]


def test_load_comparative_queryset_reads_jsonl(tmp_path):
    llm = _ScriptedLLM([_MATCH_AB, _WRITE_OK])
    item = build_comparative_queryset(POOL, llm, _GEN_3).items[0]

    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps(comparative_item_to_dict(item)) + "\n")

    loaded = load_comparative_queryset(str(path))
    assert len(loaded) == 1
    assert loaded[0].query == item.query
