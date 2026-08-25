"""Tag normalization + generate_tags/normalize_tags over a fake LLM."""

from __future__ import annotations

import pytest

from rag import tagger
from rag.config import AnthropicSpec
from rag.tagger import _excerpt, _filter_remap, _normalize, _RemapOut, _TagsOut, generate_tags


def test_normalize_kebab_dedupe_and_cap():
    tags = _normalize(["Mixture Of Experts", "MoE!!", "mixture-of-experts", "RL"], max_tags=2)
    assert tags == ["mixture-of-experts", "moe"]


def test_excerpt_pulls_title_and_headings():
    md = "## Cool Paper\n\n## Abstract\nWe do things.\n\n## 1 Intro\nbody"
    out = _excerpt(md)
    assert "Title: Cool Paper" in out
    assert "We do things." in out
    assert "## 1 Intro" in out


def test_generate_tags_uses_llm_and_normalizes(monkeypatch):
    class _Fake:
        def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
            return _TagsOut(tags=["Multi-Head Latent Attention", "MLA", "mla"])

    monkeypatch.setattr(tagger, "build_llm", lambda spec: _Fake())
    tags = generate_tags("## Paper\n\n## Abstract\nx", AnthropicSpec(), max_tags=5)
    # kebab-cased + de-duplicated ("MLA"/"mla" collapse).
    assert tags == ["multi-head-latent-attention", "mla"]


def test_generate_tags_honors_min_tags_and_excerpt_chars(monkeypatch):
    prompts: list[str] = []

    class _Fake:
        def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
            prompts.append(user)
            return _TagsOut(tags=[])

    monkeypatch.setattr(tagger, "build_llm", lambda spec: _Fake())
    generate_tags(
        "## Paper\n\n## Abstract\n" + "x" * 100,
        AnthropicSpec(),
        max_tags=8,
        min_tags=3,
        max_excerpt_chars=20,
    )
    assert "3-8 lowercase kebab-case tags" in prompts[0]
    # The excerpt itself (not the whole prompt) is capped to max_excerpt_chars.
    assert "x" * 100 not in prompts[0]


def test_generate_tags_propagates_when_structured_parse_fails(monkeypatch):
    """No local try/except in generate_tags — an exhausted-retry failure raises, and
    it's up to the caller (pipeline.py, ingest.py) to catch and degrade to `[]`."""

    class _Fake:
        def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
            raise ValueError("instructor gave up after retries")

    monkeypatch.setattr(tagger, "build_llm", lambda spec: _Fake())
    with pytest.raises(ValueError):
        generate_tags("## Paper\n\n## Abstract\nx", AnthropicSpec())


def test_filter_remap_keeps_real_remaps_and_canonicalizes_values():
    # Identity ("rl"->"rl") and unknown source ("ghost") dropped; value kebab-cased.
    m = _filter_remap(
        {"moe": "mixture-of-experts", "rl": "rl", "ghost": "x", "cot": "Chain Of Thought"},
        valid={"moe", "rl", "cot"},
    )
    assert m == {"moe": "mixture-of-experts", "cot": "chain-of-thought"}


def test_normalize_tags_uses_llm_over_the_vocabulary(monkeypatch):
    class _Fake:
        def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
            return _RemapOut(remap={"moe": "mixture-of-experts"})

    monkeypatch.setattr(tagger, "build_llm", lambda spec: _Fake())
    m = tagger.normalize_tags(["moe", "mixture-of-experts", "rl"], AnthropicSpec())
    assert m == {"moe": "mixture-of-experts"}


def test_normalize_tags_empty_input_skips_llm(monkeypatch):
    def _boom(spec):
        raise AssertionError("must not build an LLM for an empty vocabulary")

    monkeypatch.setattr(tagger, "build_llm", _boom)
    assert tagger.normalize_tags([], AnthropicSpec()) == {}
