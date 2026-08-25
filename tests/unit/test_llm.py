"""Provider dispatch, api-key resolution, and the LiteLLM-backed tool-use loop."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from rag.config import AnthropicSpec, GeminiSpec, LLMSpec, OpenAISpec, SGLangSpec, VLLMSpec
from rag.llm import LiteLLMBackend, _api_key, _ThinkTagStripper, build_llm


@pytest.mark.parametrize(
    "spec",
    [
        OpenAISpec(api_base="http://x"),
        VLLMSpec(api_base="http://x"),
        SGLangSpec(api_base="http://x"),
        AnthropicSpec(),
        GeminiSpec(),
    ],
)
def test_build_llm_dispatch(spec):
    assert isinstance(build_llm(spec), LiteLLMBackend)


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


# ---- model-string / completion() kwargs construction -------------------------


@pytest.mark.parametrize(
    "spec,expected_model",
    [
        (AnthropicSpec(model="claude-x"), "anthropic/claude-x"),
        (OpenAISpec(model="gpt-x"), "openai/gpt-x"),
        (VLLMSpec(model="local-x", api_base="http://x"), "openai/local-x"),
        (SGLangSpec(model="local-y", api_base="http://x"), "openai/local-y"),
        (GeminiSpec(model="gemini-x"), "gemini/gemini-x"),
    ],
)
def test_model_string_per_spec_variant(spec, expected_model):
    # VLLMSpec/SGLangSpec route through the same "openai" LiteLLM provider prefix
    # as OpenAISpec — they're OpenAI-wire-format servers, just a different registry key.
    assert LiteLLMBackend(spec)._model == expected_model


def test_kwargs_local_server_sets_api_base_and_custom_provider():
    spec = OpenAISpec(api_base="http://localhost:1234/v1", api_key_env="UNUSED_KEY")
    kwargs = LiteLLMBackend(spec)._kwargs()
    assert kwargs["api_base"] == "http://localhost:1234/v1"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["api_key"] == "local-no-key"


def test_kwargs_omits_api_base_for_cloud_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    kwargs = LiteLLMBackend(AnthropicSpec())._kwargs()
    assert "api_base" not in kwargs
    assert "custom_llm_provider" not in kwargs


def test_kwargs_includes_timeout_and_num_retries_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    kwargs = LiteLLMBackend(AnthropicSpec(timeout=30.0, max_retries=5))._kwargs()
    assert kwargs["timeout"] == 30.0
    assert kwargs["num_retries"] == 5


def test_kwargs_omits_timeout_and_num_retries_when_sentinel(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    kwargs = LiteLLMBackend(AnthropicSpec())._kwargs()
    assert "timeout" not in kwargs
    assert "num_retries" not in kwargs


# ---- run_tools loop against a fake litellm.completion() ----------------------


def _chunk(*, content=None, tool_call=None, reasoning_content=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_call, reasoning_content=reasoning_content
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_call(index, cid, name, args):
    fn = SimpleNamespace(name=name, arguments=args)
    return [SimpleNamespace(index=index, id=cid, function=fn)]


def _usage_chunk(prompt_tokens, completion_tokens):
    # The usage-only terminal chunk sent when `stream_options.include_usage` is
    # honored: no choices, just a cumulative-for-that-call usage object.
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[], usage=usage)


class _FakeCompletionSeq:
    """Fake `litellm.completion`: one chunk-list per round, records each call's kwargs."""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = 0
        self.kwargs_seen: list[dict] = []

    def __call__(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        self.calls += 1
        return iter(self._rounds[self.calls - 1])


def _backend(monkeypatch, fake):
    monkeypatch.setattr("rag.llm.litellm.completion", fake)
    return LiteLLMBackend(OpenAISpec(api_base="http://x"))


def test_run_tools_executes_then_answers(monkeypatch):
    fake = _FakeCompletionSeq(
        [
            [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))],
            [_chunk(content="MLA shrinks the KV cache [r1].")],
        ]
    )
    backend = _backend(monkeypatch, fake)

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


def test_run_tools_sums_usage_across_rounds(monkeypatch):
    fake = _FakeCompletionSeq(
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
    backend = _backend(monkeypatch, fake)

    out, usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "passage r1",
    )

    assert out == "Final answer [r1]."
    assert usage.input_tokens == 100 + 150
    assert usage.output_tokens == 10 + 20


def test_run_tools_usage_unknown_when_any_round_omits_it(monkeypatch):
    # A local server that ignores stream_options never sends the usage chunk —
    # report the whole call's usage as unknown rather than under-counting.
    fake = _FakeCompletionSeq(
        [
            [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))],
            [_chunk(content="Final answer [r1]."), _usage_chunk(150, 20)],
        ]
    )
    backend = _backend(monkeypatch, fake)

    _out, usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "passage r1",
    )

    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_run_tools_stops_at_max_rounds_without_a_final_answer(monkeypatch):
    # If the LLM keeps calling tools every round (a stuck ReAct loop, or a misbehaving
    # client), the loop must not spin forever — it should stop after exactly `max_rounds`
    # rounds and return whatever text the last round produced, not raise or keep querying.
    def tool_only_round():
        return [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))]

    fake = _FakeCompletionSeq([tool_only_round() for _ in range(3)])
    backend = _backend(monkeypatch, fake)

    executed = []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: executed.append((name, args)) or "passage r1",
        max_rounds=3,
    )

    assert fake.calls == 3  # queried exactly max_rounds times, not more
    assert len(executed) == 3  # every round's tool call still ran
    assert out == ""  # no round ever produced visible text — nothing to fabricate as an answer


# ---- malformed tool-call JSON: now one guarded site for every provider ------


def test_run_tools_malformed_tool_call_json_falls_back_to_empty_args(monkeypatch):
    # Every provider's tool-call arguments arrive through the same normalized
    # `delta.function.arguments` string now, so one guard covers all of them —
    # this used to only exist for the OpenAI-compat backend.
    fake = _FakeCompletionSeq(
        [
            [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", "not json"))],
            [_chunk(content="done")],
        ]
    )
    backend = _backend(monkeypatch, fake)

    executed = []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: executed.append((name, args)) or "ok",
    )

    assert executed == [("search_papers", {})]
    assert out == "done"


def test_run_tools_empty_tool_call_arguments_call_with_no_args(monkeypatch):
    # BerriAI/litellm#5063: a zero-arg tool call under stream=True can send
    # "arguments": "" instead of "{}" — must resolve to {}, same as malformed JSON,
    # not raise, since a zero-arg tool's correct call is `{}`.
    fake = _FakeCompletionSeq(
        [
            [_chunk(tool_call=_tool_call(0, "call_1", "ping", ""))],
            [_chunk(content="done")],
        ]
    )
    backend = _backend(monkeypatch, fake)

    executed = []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "ping", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: executed.append((name, args)) or "ok",
    )

    assert executed == [("ping", {})]
    assert out == "done"


# ---- stop_check: interrupt a streaming turn early (chat "stop" button) ------


def test_run_tools_stop_check_cuts_the_stream_short(monkeypatch):
    # A stop mid-round returns whatever text streamed before the check fired, and never
    # processes the chunks after it.
    fake = _FakeCompletionSeq(
        [[_chunk(content="Hello "), _chunk(content="world"), _chunk(content="!")]]
    )
    backend = _backend(monkeypatch, fake)

    seen = {"n": 0}

    def stop_check():
        seen["n"] += 1
        return seen["n"] > 1  # let the round start, stop right after the first chunk

    texts = []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_text=texts.append,
        stop_check=stop_check,
    )

    assert out == "Hello "
    assert texts == ["Hello "]


def test_run_tools_stop_check_skips_the_next_round(monkeypatch):
    # A stop that fires after a tool-call round finishes must not start another round,
    # even though the model would keep going.
    fake = _FakeCompletionSeq(
        [
            [_chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}'))],
            [_chunk(content="unreachable")],
        ]
    )
    backend = _backend(monkeypatch, fake)

    # A real Event, not a call-counting fake — is_set() is idempotent, so it doesn't
    # matter exactly how many times (or when within a round) run_tools happens to poll
    # it, only whether it was set before round 2's top-of-loop check.
    stop_event = threading.Event()
    executed = []

    def execute(name, args):
        executed.append((name, args))
        stop_event.set()  # Stop clicked right as round 1's tool call runs
        return "passage r1"

    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "what is MLA?"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=execute,
        stop_check=stop_event.is_set,
    )

    assert executed == [("search_papers", {"query": "mla"})]  # round 1's tool call still ran
    assert fake.calls == 1  # round 2 never fired
    assert out == ""  # round 1 produced no visible text


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


# ---- inline <think> tag stripping (LiteLLM streaming) -------------------------


def test_think_tag_in_one_chunk(monkeypatch):
    fake = _FakeCompletionSeq(
        [[_chunk(content="<think>should check X</think>The answer is 42 [r1].")]]
    )
    backend = _backend(monkeypatch, fake)
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
    fake = _FakeCompletionSeq(
        [
            [
                _chunk(content="<th"),
                _chunk(content="ink>reasoning here"),
                _chunk(content="</thi"),
                _chunk(content="nk>answer text"),
            ]
        ]
    )
    backend = _backend(monkeypatch, fake)
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
    fake = _FakeCompletionSeq([[_chunk(content="<think>lost reasoning")]])
    backend = _backend(monkeypatch, fake)
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
    fake = _FakeCompletionSeq([[_chunk(content="Intro. <think>plan</think> Conclusion [r1].")]])
    backend = _backend(monkeypatch, fake)
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
    fake = _FakeCompletionSeq([[_chunk(content="Value < 5 and > 3 [r1].")]])
    backend = _backend(monkeypatch, fake)
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
    fake = _FakeCompletionSeq(
        [
            [
                _chunk(content="<think>deciding to search</think>"),
                _chunk(tool_call=_tool_call(0, "call_1", "search_papers", '{"query": "mla"}')),
            ],
            [_chunk(content="Final answer [r1].")],
        ]
    )
    backend = _backend(monkeypatch, fake)
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

    second_call_messages = fake.kwargs_seen[1]["messages"]
    assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "THINK_TAG"


# ---- reasoning_content: structured reasoning, not just inline <think> -------


def test_reasoning_content_field_is_forwarded_to_on_reasoning(monkeypatch):
    # LiteLLM normalizes structured chain-of-thought (from any provider that has it)
    # into `delta.reasoning_content` — separate from the inline-<think>-tag path above,
    # which only matters for models that don't use the structured field.
    fake = _FakeCompletionSeq(
        [
            [
                _chunk(reasoning_content="thinking it through"),
                _chunk(content="42."),
            ]
        ]
    )
    backend = _backend(monkeypatch, fake)
    reasonings = []
    out, _usage = backend.run_tools(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search_papers", "description": "d", "input_schema": {"type": "object"}}],
        execute=lambda name, args: "result",
        on_reasoning=reasonings.append,
    )
    assert out == "42."
    assert reasonings == ["thinking it through"]
