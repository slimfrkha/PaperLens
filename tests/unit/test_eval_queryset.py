"""Harness: span-anchored eval-set generation (offline, fake_llm)."""

from __future__ import annotations

import json

from eval.fingerprint import corpus_fingerprint
from eval.queryset import (
    GenConfig,
    build_queryset,
    item_to_dict,
    iter_queryset,
    iter_sections,
    split_by_paper,
)
from rag.llm import LLMBackend

# A minimal paper: title section (skip), a real section, a References section (skip),
# and a too-short section (skip on min_tokens).
_MD = """## DeepFake-V1 Technical Report

Alice, Bob, and Carol. Institute of Things.

## 2. Method

We introduce multi-head latent attention, which compresses the key-value cache by
projecting keys and values into a shared low-rank latent space before caching them.
This reduces memory bandwidth during autoregressive decoding.

## 2.1 Training

The model is trained with FP8 mixed precision on a cluster of GPUs using a cosine
learning-rate schedule and a warmup of two thousand steps for stability.

## Tiny

Too short.

## References

[1] Someone et al. A paper. 2024.
"""

_QA = '{"question": "How does MLA reduce the KV cache?", "answer": "low-rank latent."}'


def test_iter_sections_spans_slice_real_markdown():
    secs = list(iter_sections(_MD, "deepfake-v1", min_tokens=24))
    titles = [s.title for s in secs]
    # Title, References, and the tiny section are all skipped.
    assert titles == ["Method", "Training"]
    for s in secs:
        assert _MD[s.start : s.end] == s.body  # the exit criterion, at the span level
        assert s.body.strip() == s.body
    assert secs[0].number == "2"
    assert secs[1].number == "2.1"


def test_build_queryset_gold_spans_slice_real_markdown(fake_llm):
    pool = {"deepfake-v1": _MD}
    items = build_queryset(pool, fake_llm(answer=_QA), GenConfig())

    assert len(items) == 2  # one per substantive section
    for it in items:
        start, end = it.gold_span
        assert _MD[start:end].strip()  # non-empty real markdown
        assert it.query == "How does MLA reduce the KV cache?"
        assert it.paper_id == "deepfake-v1"


def test_generate_query_skips_unparsable_output(fake_llm):
    pool = {"p": _MD}
    items = build_queryset(pool, fake_llm(answer="not json at all"), GenConfig())
    assert items == []


class _BoomLLM(LLMBackend):
    """A backend whose completion always fails, to exercise per-section isolation."""

    def __init__(self) -> None:  # no spec / client needed
        pass

    def complete(self, system, user, max_tokens=None):
        raise RuntimeError("server down")

    def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
        raise RuntimeError("server down")

    def run_tools(self, *args, **kwargs):
        return ""


def test_iter_queryset_isolates_per_section_failures():
    # A failing LLM on every section is logged and skipped, never propagated — one
    # server hiccup must not sink a long run.
    pool = {"a": _MD, "b": _MD}
    items = list(iter_queryset(pool, _BoomLLM(), GenConfig()))
    assert items == []


def test_item_to_dict_json_round_trips_span_as_list(fake_llm):
    pool = {"p": _MD}
    it = build_queryset(pool, fake_llm(answer=_QA), GenConfig())[0]
    loaded = json.loads(json.dumps(item_to_dict(it)))
    assert loaded["gold_span"] == list(it.gold_span)


def test_fingerprint_stable_and_sensitive():
    a = {"p1": "one", "p2": "two"}
    assert corpus_fingerprint(a) == corpus_fingerprint(dict(reversed(list(a.items()))))
    assert corpus_fingerprint(a) != corpus_fingerprint({"p1": "one", "p2": "changed"})
    assert corpus_fingerprint(a) != corpus_fingerprint({**a, "p3": "three"})


def test_split_partitions_by_paper_no_leak(fake_llm):
    pool = {f"p{i}": _MD for i in range(4)}
    items = build_queryset(pool, fake_llm(answer=_QA), GenConfig(test_frac=0.25))
    dev, test = split_by_paper(items, pool, GenConfig(test_frac=0.25))

    dev_papers = {it.paper_id for it in dev}
    test_papers = {it.paper_id for it in test}
    assert dev_papers.isdisjoint(test_papers)  # no paper in both splits
    assert len(test_papers) == 1  # round(4 * 0.25)
    assert dev_papers | test_papers == set(pool)


def test_split_single_paper_all_dev(fake_llm):
    pool = {"only": _MD}
    items = build_queryset(pool, fake_llm(answer=_QA), GenConfig())
    dev, test = split_by_paper(items, pool, GenConfig())
    assert test == []
    assert len(dev) == len(items)
