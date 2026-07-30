"""generate_paraphrases: parsing, graceful degradation, dedupe (offline: FakeLLM only)."""

from __future__ import annotations

from rag.query_expansion import generate_paraphrases


def test_generate_paraphrases_parses_json_array(fake_llm):
    llm = fake_llm(answer='["variant one", "variant two"]')

    out = generate_paraphrases("original query", llm, n=3)

    assert out == ["variant one", "variant two"]


def test_generate_paraphrases_malformed_json_returns_empty(fake_llm):
    llm = fake_llm(answer="not json at all")

    assert generate_paraphrases("q", llm, n=3) == []


def test_generate_paraphrases_llm_exception_returns_empty(fake_llm):
    class RaisingLLM(fake_llm):
        def complete(self, system, user, max_tokens=None):
            raise RuntimeError("backend down")

    assert generate_paraphrases("q", RaisingLLM(), n=3) == []


def test_generate_paraphrases_drops_duplicate_of_original(fake_llm):
    llm = fake_llm(answer='["Original Query", "a real variant"]')

    out = generate_paraphrases("original query", llm, n=3)

    assert out == ["a real variant"]


def test_generate_paraphrases_truncates_to_n(fake_llm):
    llm = fake_llm(answer='["a", "b", "c", "d"]')

    out = generate_paraphrases("q", llm, n=2)

    assert out == ["a", "b"]
