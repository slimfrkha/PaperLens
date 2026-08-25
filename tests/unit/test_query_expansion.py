"""generate_paraphrases: structured output, graceful degradation, dedupe (offline: FakeLLM only)."""

from __future__ import annotations

from rag.query_expansion import _ParaphrasesOut, generate_paraphrases


def test_generate_paraphrases_returns_structured_output(fake_llm):
    llm = fake_llm(structured=_ParaphrasesOut(paraphrases=["variant one", "variant two"]))

    out = generate_paraphrases("original query", llm, n=3)

    assert out == ["variant one", "variant two"]


def test_generate_paraphrases_structured_failure_returns_empty_and_warns(fake_llm, capsys):
    llm = fake_llm(structured_exception=ValueError("instructor gave up after retries"))

    assert generate_paraphrases("q", llm, n=3) == []
    assert "[warn]" in capsys.readouterr().out


def test_generate_paraphrases_llm_exception_returns_empty(fake_llm):
    class RaisingLLM(fake_llm):
        def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
            raise RuntimeError("backend down")

    assert generate_paraphrases("q", RaisingLLM(), n=3) == []


def test_generate_paraphrases_drops_duplicate_of_original(fake_llm):
    llm = fake_llm(structured=_ParaphrasesOut(paraphrases=["Original Query", "a real variant"]))

    out = generate_paraphrases("original query", llm, n=3)

    assert out == ["a real variant"]


def test_generate_paraphrases_truncates_to_n(fake_llm):
    llm = fake_llm(structured=_ParaphrasesOut(paraphrases=["a", "b", "c", "d"]))

    out = generate_paraphrases("q", llm, n=2)

    assert out == ["a", "b"]
