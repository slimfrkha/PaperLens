"""Tag parsing / normalization + generate_tags over a fake LLM."""

from __future__ import annotations

from rag import tagger
from rag.config import LLMSpec
from rag.tagger import _excerpt, _normalize, _parse, generate_tags


def test_parse_json_array():
    assert _parse('Here: ["moe", "rl"] done') == ["moe", "rl"]


def test_parse_falls_back_to_comma_and_newline():
    assert _parse("- moe\n- rl, quant") == ["moe", "rl", "quant"]


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
        def complete(self, system, user, max_tokens=None):
            return '["Multi-Head Latent Attention", "MLA", "mla"]'

    monkeypatch.setattr(tagger, "build_llm", lambda spec: _Fake())
    tags = generate_tags("## Paper\n\n## Abstract\nx", LLMSpec(), max_tags=5)
    # kebab-cased + de-duplicated ("MLA"/"mla" collapse).
    assert tags == ["multi-head-latent-attention", "mla"]
