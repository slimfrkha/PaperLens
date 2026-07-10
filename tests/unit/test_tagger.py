"""Tag parsing / normalization + generate_tags over a fake LLM."""

from __future__ import annotations

from rag import tagger
from rag.config import AnthropicSpec
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
    tags = generate_tags("## Paper\n\n## Abstract\nx", AnthropicSpec(), max_tags=5)
    # kebab-cased + de-duplicated ("MLA"/"mla" collapse).
    assert tags == ["multi-head-latent-attention", "mla"]


def test_generate_tags_honors_min_tags_and_excerpt_chars(monkeypatch):
    prompts: list[str] = []

    class _Fake:
        def complete(self, system, user, max_tokens=None):
            prompts.append(user)
            return "[]"

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
