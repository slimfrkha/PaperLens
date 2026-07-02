"""Provider dispatch, api-key resolution, and the OpenAI tool-use loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config import LLMSpec
from rag.llm import (
    AnthropicBackend,
    GeminiBackend,
    OpenAICompatBackend,
    SGLangBackend,
    VLLMBackend,
    _api_key,
    build_llm,
)


@pytest.mark.parametrize(
    "provider,cls",
    [
        ("openai", OpenAICompatBackend),
        ("vllm", VLLMBackend),
        ("sglang", SGLangBackend),
        ("anthropic", AnthropicBackend),
        ("gemini", GeminiBackend),
    ],
)
def test_build_llm_dispatch(provider, cls):
    assert isinstance(build_llm(LLMSpec(provider=provider)), cls)


def test_build_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(LLMSpec(provider="nope"))


def test_api_key_local_openai_needs_no_key(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_KEY", raising=False)
    spec = LLMSpec(
        provider="openai", api_base="http://localhost:1234/v1", api_key_env="LOCAL_LLM_KEY"
    )
    assert _api_key(spec) == "local-no-key"


def test_api_key_missing_key_raises_for_cloud(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    spec = LLMSpec(provider="anthropic", api_key_env="SOME_KEY")
    with pytest.raises(RuntimeError, match="No API key"):
        _api_key(spec)


def test_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    assert _api_key(LLMSpec(api_key_env="MY_KEY")) == "secret"


# ---- OpenAI tool-use loop against a fake streaming client -------------------


def _chunk(*, content=None, tool_call=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_call, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_call(index, cid, name, args):
    fn = SimpleNamespace(name=name, arguments=args)
    return [SimpleNamespace(index=index, id=cid, function=fn)]


class _FakeCompletions:
    """Two rounds: first streams a tool call, then streams the final answer."""

    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return iter(
                [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))]
            )
        return iter([_chunk(content="MLA shrinks the KV cache [r1].")])


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_openai_run_tools_executes_then_answers(monkeypatch):
    backend = OpenAICompatBackend(LLMSpec(provider="openai", api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: _FakeClient())

    executed = []
    texts = []

    def execute(name, args):
        executed.append((name, args))
        return "passage r1"

    out = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=execute,
        on_text=texts.append,
    )

    assert executed == [("search_papers", {"query": "mla"})]
    assert out == "MLA shrinks the KV cache [r1]."
    assert "".join(texts) == out


def test_anthropic_client_built_lazily(monkeypatch):
    # Only runs when the optional `anthropic` extra is installed.
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicBackend(LLMSpec(provider="anthropic"))._client()
    assert client is not None


def test_gemini_client_built_lazily(monkeypatch):
    # Only runs when the optional `gemini` extra is installed.
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = GeminiBackend(LLMSpec(provider="gemini", api_key_env="GEMINI_API_KEY"))._client()
    assert client is not None
