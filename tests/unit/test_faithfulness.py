"""Faithfulness registry, sentence splitting/ref attribution, and pooling."""

from __future__ import annotations

import pytest

from rag.config import FaithfulnessCfg, HFFaithfulnessCfg
from rag.faithfulness import (
    HFFaithfulnessChecker,
    Verdict,
    _to_verdict,
    attribute_refs,
    best_support,
    build_faithfulness_checker,
    split_sentences,
)


def test_build_faithfulness_checker_unknown_variant_raises():
    # The base FaithfulnessCfg is not a registered variant.
    with pytest.raises(ValueError, match="Unknown faithfulness config"):
        build_faithfulness_checker(FaithfulnessCfg())


def test_build_faithfulness_checker_hf_is_lazy():
    c = build_faithfulness_checker(HFFaithfulnessCfg())
    assert isinstance(c, HFFaithfulnessChecker)
    assert c._model is None  # model not loaded until first check_batch()


def test_build_faithfulness_checker_honors_config_fields():
    c = build_faithfulness_checker(
        HFFaithfulnessCfg(
            model="some/model",
            revision="rev",
            max_length=256,
            contradiction_max=0.1,
            entailment_min=0.4,
        )
    )
    assert c.model_name == "some/model"
    assert c.revision == "rev"
    assert c.max_length == 256
    assert c.contradiction_max == 0.1
    assert c.entailment_min == 0.4


def test_check_batch_empty_pairs_makes_no_call():
    checker = HFFaithfulnessChecker("some/model")
    assert checker.check_batch([]) == []
    assert checker._model is None  # never touched the lazy model property


def test_check_batch_uses_injected_model():
    checker = HFFaithfulnessChecker("some/model", contradiction_max=0.05, entailment_min=0.3)

    class _FakeModel:
        def predict(self, pairs):
            return [0.9, 0.02, 0.15]

    checker._model = _FakeModel()
    verdicts = checker.check_batch([("p1", "h1"), ("p2", "h2"), ("p3", "h3")])
    assert verdicts == [
        Verdict(label="entailment", score=0.9),
        Verdict(label="contradiction", score=0.02),
        Verdict(label="neutral", score=0.15),
    ]


@pytest.mark.parametrize(
    "score, expected_label",
    [
        (0.05, "contradiction"),  # at the contradiction_max boundary
        (0.0, "contradiction"),
        (0.3, "entailment"),  # at the entailment_min boundary
        (1.0, "entailment"),
        (0.2, "neutral"),  # strictly between the two thresholds
    ],
)
def test_to_verdict_threshold_boundaries(score, expected_label):
    v = _to_verdict(score, contradiction_max=0.05, entailment_min=0.3)
    assert v.label == expected_label
    assert v.score == score


def test_split_sentences_basic():
    text = "MLA shrinks the KV cache. This works by compression! Does it help?"
    assert split_sentences(text) == [
        "MLA shrinks the KV cache.",
        "This works by compression!",
        "Does it help?",
    ]


def test_attribute_refs_single_ref_per_sentence():
    text = "MLA shrinks the cache [r1]. The MoE layer balances load [r2]."
    out = attribute_refs(text, {"r1", "r2"})
    assert out == {
        "r1": ["MLA shrinks the cache [r1]."],
        "r2": ["The MoE layer balances load [r2]."],
    }


def test_attribute_refs_comma_separated_bracket():
    text = "Both mechanisms help [r1, r2]."
    out = attribute_refs(text, {"r1", "r2"})
    assert out == {
        "r1": ["Both mechanisms help [r1, r2]."],
        "r2": ["Both mechanisms help [r1, r2]."],
    }


def test_attribute_refs_adjacent_brackets():
    text = "Both mechanisms help [r1][r2]."
    out = attribute_refs(text, {"r1", "r2"})
    assert out == {
        "r1": ["Both mechanisms help [r1][r2]."],
        "r2": ["Both mechanisms help [r1][r2]."],
    }


def test_attribute_refs_ref_cited_across_multiple_sentences():
    text = "MLA shrinks the cache [r1]. It also speeds up decoding [r1]."
    out = attribute_refs(text, {"r1"})
    assert out == {
        "r1": [
            "MLA shrinks the cache [r1].",
            "It also speeds up decoding [r1].",
        ]
    }


def test_attribute_refs_uncited_ref_is_absent():
    text = "MLA shrinks the cache [r1]."
    out = attribute_refs(text, {"r1", "r2"})
    assert "r2" not in out


def test_best_support_picks_highest_score():
    verdicts = [
        Verdict(label="neutral", score=0.2),
        Verdict(label="entailment", score=0.8),
        Verdict(label="contradiction", score=0.01),
    ]
    assert best_support(verdicts) == Verdict(label="entailment", score=0.8)
