"""Reranker registry, the LLM reranker's parsing, and build dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config import HFRerankerCfg, LLMRerankerCfg, RerankerCfg, VoyageRerankerCfg
from rag.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    VoyageReranker,
    build_reranker,
)


def test_build_reranker_unknown_variant_raises():
    # The base RerankerCfg is not a registered variant.
    with pytest.raises(ValueError, match="Unknown reranker config"):
        build_reranker(RerankerCfg())


def test_build_reranker_hf_is_lazy(make_config):
    r = build_reranker(HFRerankerCfg())
    assert isinstance(r, CrossEncoderReranker)
    assert r._model is None  # model not loaded until first score()


def test_build_reranker_honors_max_length_and_max_chars(fake_llm):
    hf = build_reranker(HFRerankerCfg(max_length=256))
    assert hf.max_length == 256

    llm_r = build_reranker(LLMRerankerCfg(max_chars=100), llm=fake_llm())
    assert llm_r.max_chars == 100


def test_build_reranker_llm_requires_llm(fake_llm):
    with pytest.raises(ValueError, match="needs an LLM"):
        build_reranker(LLMRerankerCfg())  # no llm passed
    r = build_reranker(LLMRerankerCfg(), llm=fake_llm())
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


# ---- Voyage: dedicated rerank API via a fake litellm.rerank() --------------


def test_build_reranker_voyage_requires_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        build_reranker(VoyageRerankerCfg())


def test_voyage_reranker_reorders_by_index(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "k")

    def fake_rerank(**kwargs):
        assert kwargs["model"] == "voyage/rerank-2.5"
        assert kwargs["query"] == "q"
        assert kwargs["documents"] == ["a", "b", "c"]
        assert kwargs["top_n"] == 3
        assert kwargs["api_key"] == "k"
        # Results come back sorted by relevance (not input order) — index 2 first.
        return SimpleNamespace(
            results=[
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.1},
            ]
        )

    monkeypatch.setattr("rag.reranker.litellm.rerank", fake_rerank)

    r = build_reranker(VoyageRerankerCfg())
    assert isinstance(r, VoyageReranker)
    assert r.score("q", ["a", "b", "c"]) == [0.5, 0.1, 0.9]


def test_voyage_reranker_ignores_out_of_range_index(monkeypatch):
    # A contract-violating response must degrade that one doc to 0.0, not raise.
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    monkeypatch.setattr(
        "rag.reranker.litellm.rerank",
        lambda **kwargs: SimpleNamespace(
            results=[
                {"index": 0, "relevance_score": 0.7},
                {"index": 5, "relevance_score": 0.9},  # out of range for a 2-doc list
            ]
        ),
    )

    r = build_reranker(VoyageRerankerCfg())
    assert r.score("q", ["a", "b"]) == [0.7, 0.0]


def test_voyage_reranker_empty_docs_makes_no_call(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "k")

    def _boom(**kwargs):
        raise AssertionError("must not call litellm.rerank for an empty doc list")

    monkeypatch.setattr("rag.reranker.litellm.rerank", _boom)

    assert VoyageReranker("rerank-2.5").score("q", []) == []
