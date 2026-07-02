"""Reranker registry, the LLM reranker's parsing, and build dispatch."""

from __future__ import annotations

import pytest

from rag.config import RerankerCfg
from rag.reranker import (
    _RERANKERS,
    CrossEncoderReranker,
    LLMReranker,
    build_reranker,
)


def test_registry_has_builtin_rerankers():
    assert {"hf", "llm"} <= set(_RERANKERS)


def test_build_reranker_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown reranker type"):
        build_reranker(RerankerCfg(type="nope"))


def test_build_reranker_hf_is_lazy(make_config):
    r = build_reranker(RerankerCfg())  # type defaults to "hf"
    assert isinstance(r, CrossEncoderReranker)
    assert r._model is None  # model not loaded until first score()


def test_build_reranker_llm_requires_llm(fake_llm):
    with pytest.raises(ValueError, match="needs an LLM"):
        build_reranker(RerankerCfg(type="llm"))  # no llm passed
    r = build_reranker(RerankerCfg(type="llm"), llm=fake_llm())
    assert isinstance(r, LLMReranker)


def test_llm_reranker_parses_scores(fake_llm):
    r = LLMReranker(fake_llm(answer="[3, 9, 1]"))
    assert r.score("q", ["a", "b", "c"]) == [3.0, 9.0, 1.0]


def test_llm_reranker_empty_docs_makes_no_call(fake_llm):
    llm = fake_llm(answer="[1]")
    assert LLMReranker(llm).score("q", []) == []
    assert llm.complete_calls == []


def test_llm_reranker_pads_short_and_truncates_long(fake_llm):
    assert LLMReranker(fake_llm(answer="[5]")).score("q", ["a", "b", "c"]) == [5.0, 0.0, 0.0]
    assert LLMReranker(fake_llm(answer="[1, 2, 3, 4]")).score("q", ["a", "b"]) == [1.0, 2.0]


def test_llm_reranker_falls_back_to_zeros_on_bad_output(fake_llm):
    # Unparsable output -> zeros; with a stable sort that keeps dense order.
    assert LLMReranker(fake_llm(answer="sorry, no JSON here")).score("q", ["a", "b"]) == [0.0, 0.0]
