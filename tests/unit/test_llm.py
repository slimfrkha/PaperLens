"""Provider dispatch, api-key resolution, and the OpenAI tool-use loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config import AnthropicSpec, GeminiSpec, LLMSpec, OpenAISpec, SGLangSpec, VLLMSpec
from rag.llm import (
    AnthropicBackend,
    GeminiBackend,
    OpenAICompatBackend,
    SGLangBackend,
    VLLMBackend,
    _api_key,
    _ThinkTagStripper,
    build_llm,
)


@pytest.mark.parametrize(
    "spec,cls",
    [
        (OpenAISpec(api_base="http://x"), OpenAICompatBackend),
        (VLLMSpec(api_base="http://x"), VLLMBackend),
        (SGLangSpec(api_base="http://x"), SGLangBackend),
        (AnthropicSpec(), AnthropicBackend),
        (GeminiSpec(), GeminiBackend),
    ],
)
def test_build_llm_dispatch(spec, cls):
    assert isinstance(build_llm(spec), cls)


def test_build_llm_unknown_spec_raises():
    # The base LLMSpec is not a registered provider variant.
    with pytest.raises(ValueError, match="Unknown LLM spec"):
        build_llm(LLMSpec())


def test_api_key_local_openai_needs_no_key(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_KEY", raising=False)
    spec = OpenAISpec(api_base="http://localhost:1234/v1", api_key_env="LOCAL_LLM_KEY")
    assert _api_key(spec) == "local-no-key"


def test_api_key_missing_key_raises_for_cloud(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    spec = AnthropicSpec(api_key_env="SOME_KEY")
    with pytest.raises(RuntimeError, match="No API key"):
        _api_key(spec)


def test_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    assert _api_key(AnthropicSpec(api_key_env="MY_KEY")) == "secret"


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
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: _FakeClient())

    executed = []
    texts = []

    def execute(name, args):
        executed.append((name, args))
        return "passage r1"

    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=execute,
        on_text=texts.append,
    )

    assert executed == [("search_papers", {"query": "mla"})]
    assert out == "MLA shrinks the KV cache [r1]."
    assert "".join(texts) == out


def _usage_chunk(prompt_tokens, completion_tokens):
    # The usage-only terminal chunk sent when `stream_options.include_usage` is
    # honored: no choices, just a cumulative-for-that-call usage object.
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[], usage=usage)


def test_openai_run_tools_sums_usage_across_rounds(monkeypatch):
    client = _FakeClientSeq(
        [
            [
                _chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}')),
                _usage_chunk(prompt_tokens=100, completion_tokens=10),
            ],
            [
                _chunk(content="Final answer [r1]."),
                _usage_chunk(prompt_tokens=150, completion_tokens=20),
            ],
        ]
    )
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)

    out, usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "passage r1",
    )

    assert out == "Final answer [r1]."
    assert usage.input_tokens == 100 + 150
    assert usage.output_tokens == 10 + 20


def test_openai_run_tools_usage_unknown_when_any_round_omits_it(monkeypatch):
    # A local server that ignores stream_options never sends the usage chunk —
    # report the whole call's usage as unknown rather than under-counting.
    client = _FakeClientSeq(
        [
            [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))],
            [_chunk(content="Final answer [r1]."), _usage_chunk(150, 20)],
        ]
    )
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)

    _out, usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "passage r1",
    )

    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_openai_run_tools_stops_at_max_rounds_without_a_final_answer(monkeypatch):
    # If the LLM keeps calling tools every round (a stuck ReAct loop, or a misbehaving
    # client), the loop must not spin forever — it should stop after exactly `max_rounds`
    # rounds and return whatever text the last round produced, not raise or keep querying.
    def tool_only_round():
        return [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))]

    client = _FakeClientSeq([tool_only_round() for _ in range(3)])
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)

    executed = []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: executed.append((name, args)) or "passage r1",
        max_rounds=3,
    )

    assert client.completions.calls == 3  # queried exactly max_rounds times, not more
    assert len(executed) == 3  # every round's tool call still ran
    assert out == ""  # no round ever produced visible text — nothing to fabricate as an answer


# ---- _ThinkTagStripper: direct unit tests (no fake LLM client needed) -------


def test_think_tag_stripper_feed_then_finish():
    f = _ThinkTagStripper()
    visible = f.feed("<think>plan</think>answer")
    assert visible == "THINK_TAGanswer"
    assert f.reasoning == "plan"
    assert f.finish() == ""


def test_think_tag_stripper_split_across_feed_calls():
    f = _ThinkTagStripper()
    out = f.feed("<th") + f.feed("ink>reason") + f.feed("</thi") + f.feed("nk>tail")
    assert out == "THINK_TAGtail"
    assert f.reasoning == "reason"


def test_think_tag_stripper_finish_flushes_false_start_as_visible_text():
    # A trailing "<" that never completes into "<think>" must not be silently
    # dropped when the stream ends — it's genuine content, held back only because
    # more chunks *could* have completed the tag.
    f = _ThinkTagStripper()
    visible = f.feed("Score is 3 <")
    assert visible == "Score is 3 "  # the lone "<" is withheld pending more input
    assert f.finish() == "<"  # ...and flushed here, not dropped, once the stream ends


def test_think_tag_stripper_finish_folds_unterminated_tail_into_reasoning():
    f = _ThinkTagStripper()
    visible = f.feed("<think>lost")
    assert visible == ""
    assert f.finish() == ""
    assert f.reasoning == "lost"


# ---- inline <think> tag stripping (OpenAI-compatible streaming) -------------


class _FakeCompletionsSeq:
    """Fake streaming client yielding a fixed sequence of chunk-lists, one list per
    `create()` call (round), and recording the `messages` kwarg of each call."""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = 0
        self.messages_seen: list = []

    def create(self, **kwargs):
        self.messages_seen.append(kwargs.get("messages"))
        self.calls += 1
        return iter(self._rounds[self.calls - 1])


class _FakeClientSeq:
    def __init__(self, rounds):
        self.completions = _FakeCompletionsSeq(rounds)
        self.chat = SimpleNamespace(completions=self.completions)


def test_think_tag_in_one_chunk(monkeypatch):
    client = _FakeClientSeq(
        [[_chunk(content="<think>should check X</think>The answer is 42 [r1].")]]
    )
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)
    texts, reasonings = [], []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_text=texts.append,
        on_reasoning=reasonings.append,
    )
    assert out == "THINK_TAGThe answer is 42 [r1]."
    assert "".join(texts) == out
    assert reasonings == ["should check X"]


def test_think_tag_split_across_chunks(monkeypatch):
    client = _FakeClientSeq(
        [
            [
                _chunk(content="<th"),
                _chunk(content="ink>reasoning here"),
                _chunk(content="</thi"),
                _chunk(content="nk>answer text"),
            ]
        ]
    )
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)
    texts, reasonings = [], []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_text=texts.append,
        on_reasoning=reasonings.append,
    )
    assert out == "THINK_TAGanswer text"
    assert reasonings == ["reasoning here"]
    for t in texts:
        assert "<think" not in t
        assert "</think" not in t
        assert "reasoning here" not in t


def test_think_tag_unterminated_at_stream_end(monkeypatch):
    client = _FakeClientSeq([[_chunk(content="<think>lost reasoning")]])
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)
    texts, reasonings = [], []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_text=texts.append,
        on_reasoning=reasonings.append,
    )
    assert out == ""
    assert reasonings == ["lost reasoning"]
    for t in texts:
        assert "<think" not in t
        assert "lost reasoning" not in t


def test_think_tag_mixed_text_before_and_after(monkeypatch):
    client = _FakeClientSeq([[_chunk(content="Intro. <think>plan</think> Conclusion [r1].")]])
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)
    texts, reasonings = [], []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_text=texts.append,
        on_reasoning=reasonings.append,
    )
    assert out == "Intro. THINK_TAG Conclusion [r1]."
    assert reasonings == ["plan"]


def test_no_think_tags_stray_angle_brackets_unaffected(monkeypatch):
    client = _FakeClientSeq([[_chunk(content="Value < 5 and > 3 [r1].")]])
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)
    texts, reasonings = [], []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_text=texts.append,
        on_reasoning=reasonings.append,
    )
    assert out == "Value < 5 and > 3 [r1]."
    assert reasonings == []


def test_think_tag_in_tool_call_round_not_leaked_into_convo(monkeypatch):
    client = _FakeClientSeq(
        [
            [
                _chunk(content="<think>deciding to search</think>"),
                _chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}')),
            ],
            [_chunk(content="Final answer [r1].")],
        ]
    )
    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    monkeypatch.setattr(backend, "_client", lambda: client)
    executed = []
    texts, reasonings = [], []

    def execute(name, args):
        executed.append((name, args))
        return "passage r1"

    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=execute,
        on_text=texts.append,
        on_reasoning=reasonings.append,
    )

    assert executed == [("search_papers", {"query": "mla"})]
    assert out == "Final answer [r1]."
    assert reasonings == ["deciding to search"]

    second_call_messages = client.completions.messages_seen[1]
    assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "THINK_TAG"


def test_anthropic_client_built_lazily(monkeypatch):
    # Only runs when the optional `anthropic` extra is installed.
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicBackend(AnthropicSpec())._client()
    assert client is not None


def test_gemini_client_built_lazily(monkeypatch):
    # Only runs when the optional `gemini` extra is installed.
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = GeminiBackend(GeminiSpec())._client()
    assert client is not None


def test_gemini_run_tools_folds_thinking_tokens_into_output(monkeypatch):
    # thinking_config.include_thoughts=True (always on) bills thinking tokens
    # separately from candidates_token_count — they must still land in output_tokens,
    # or the one backend with visible chain-of-thought would under-report the most.
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    part = SimpleNamespace(function_call=None, thought=False, text="42.")
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    usage_metadata = SimpleNamespace(
        prompt_token_count=100, candidates_token_count=5, thoughts_token_count=40
    )
    chunk = SimpleNamespace(candidates=[candidate], usage_metadata=usage_metadata)

    class _FakeModels:
        def generate_content_stream(self, **kwargs):
            return iter([chunk])

    backend = GeminiBackend(GeminiSpec())
    monkeypatch.setattr(backend, "_client", lambda: SimpleNamespace(models=_FakeModels()))

    _text, usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 5 + 40


def test_client_built_once_and_reused(monkeypatch):
    # Regression: a fresh SDK client per call leaks httpx sockets/fds and blows the
    # open-file limit under a high-volume caller (eval harness: one call per section).
    pytest.importorskip("anthropic")
    import anthropic

    builds = {"n": 0}

    def _fake(**kw):
        builds["n"] += 1
        return SimpleNamespace(kw=kw)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(anthropic, "Anthropic", _fake)

    backend = AnthropicBackend(AnthropicSpec())
    first = backend._client()
    for _ in range(20):
        backend._client()
    assert builds["n"] == 1  # built once, not 21 times
    assert backend._client() is first


def test_openai_client_built_once_and_reused(monkeypatch):
    # The path the eval harness actually exercised (local OpenAI-compatible server).
    pytest.importorskip("openai")
    import openai

    builds = {"n": 0}

    def _fake(**kw):
        builds["n"] += 1
        return SimpleNamespace(kw=kw)

    monkeypatch.setattr(openai, "OpenAI", _fake)

    backend = OpenAICompatBackend(OpenAISpec(api_base="http://x"))
    first = backend._client()
    for _ in range(20):
        backend._client()
    assert builds["n"] == 1
    assert backend._client() is first


# ---- timeout / max_retries: 0 / -1 sentinels omit the kwarg (SDK default) ---


def test_anthropic_client_honors_timeout_and_retries(monkeypatch):
    pytest.importorskip("anthropic")
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    captured: dict = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: captured.update(kw))
    AnthropicBackend(AnthropicSpec(timeout=30.0, max_retries=5))._client()
    assert captured["timeout"] == 30.0
    assert captured["max_retries"] == 5


def test_anthropic_client_omits_unset_timeout_and_retries(monkeypatch):
    pytest.importorskip("anthropic")
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    captured: dict = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: captured.update(kw))
    AnthropicBackend(AnthropicSpec())._client()
    assert "timeout" not in captured
    assert "max_retries" not in captured


def test_openai_client_honors_timeout_and_retries(monkeypatch):
    pytest.importorskip("openai")
    import openai

    captured: dict = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: captured.update(kw))
    OpenAICompatBackend(OpenAISpec(api_base="http://x", timeout=10.0, max_retries=1))._client()
    assert captured["timeout"] == 10.0
    assert captured["max_retries"] == 1


def test_gemini_client_converts_timeout_to_ms_and_sets_retry(monkeypatch):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    captured: dict = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: captured.update(kw))
    GeminiBackend(GeminiSpec(timeout=2.5, max_retries=3))._client()
    http_options = captured["http_options"]
    assert http_options.timeout == 2500
    # Gemini's `attempts` is total attempts; max_retries=3 means 1 + 3 = 4.
    assert http_options.retry_options.attempts == 4


def test_gemini_zero_retries_still_makes_one_attempt(monkeypatch):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    captured: dict = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: captured.update(kw))
    GeminiBackend(GeminiSpec(max_retries=0))._client()
    assert captured["http_options"].retry_options.attempts == 1
