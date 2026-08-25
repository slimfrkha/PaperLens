"""Pre-cutover checklist: LiteLLMBackend against a real server, not mocks.

Every test here is skipped unless explicitly pointed at a live LLM endpoint — there's
no CI infra to keep a model server running, and this project already assumes a local
OpenAI-compatible server is the normal day-to-day setup (see docs/configuration.md).
Run these manually before merging a LiteLLM cutover:

    PAPERLENS_LLM_INTEGRATION_BASE_URL=http://localhost:1234/v1 \\
    PAPERLENS_LLM_INTEGRATION_MODEL=openai/gpt-oss-20b \\
        uv run pytest tests/integration/test_llm_litellm_checklist.py -v

The zero-argument-tool-call test targets Anthropic specifically (the shape of
BerriAI/litellm#5063) and is gated on ANTHROPIC_API_KEY separately, since it needs a
real Anthropic call rather than the local server.
"""

from __future__ import annotations

import os

import pytest

from rag.config import AnthropicSpec, OpenAISpec
from rag.llm import LiteLLMBackend

_BASE_URL = os.environ.get("PAPERLENS_LLM_INTEGRATION_BASE_URL")
_MODEL = os.environ.get("PAPERLENS_LLM_INTEGRATION_MODEL", "openai/gpt-oss-20b")

pytestmark = pytest.mark.skipif(
    not _BASE_URL,
    reason="set PAPERLENS_LLM_INTEGRATION_BASE_URL to run against a real local server",
)


def _local_backend() -> LiteLLMBackend:
    return LiteLLMBackend(
        OpenAISpec(model=_MODEL, api_base=_BASE_URL, api_key_env="UNUSED_LOCAL_LLM_KEY")
    )


_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def test_multi_round_tool_use_streaming_end_to_end():
    backend = _local_backend()
    executed = []

    def execute(name, args):
        executed.append((name, args))
        return "sunny, 25C"

    text, usage = backend.run_tools(
        system="You must call get_weather to answer weather questions, then answer "
        "in one sentence.",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[_WEATHER_TOOL],
        execute=execute,
    )

    assert executed, "the model never called the tool"
    assert executed[0][0] == "get_weather"
    assert isinstance(executed[0][1], dict)
    assert text  # a final answer was produced after the tool round
    assert (usage.input_tokens is None) == (usage.output_tokens is None)


def test_mid_stream_stop_check_interrupt():
    backend = _local_backend()
    seen = {"n": 0}

    def stop_check():
        seen["n"] += 1
        return seen["n"] > 2  # let a couple of chunks through, then stop

    texts = []
    text, _usage = backend.run_tools(
        system="Write a long paragraph about the weather in general.",
        messages=[{"role": "user", "content": "Tell me about weather."}],
        tools=[_WEATHER_TOOL],
        execute=lambda name, args: "n/a",
        on_text=texts.append,
        stop_check=stop_check,
    )

    # No hang, no exception from an undrained stream — that's the assertion. A stop
    # this early may or may not have produced visible text yet, so don't require it.
    assert text == "".join(texts)


def test_usage_reporting_is_well_formed():
    # Whether or not this particular server honors stream_options, Usage must never
    # come back half-populated — either both fields are real ints, or both are None.
    backend = _local_backend()
    _text, usage = backend.run_tools(
        system="Reply with one word.",
        messages=[{"role": "user", "content": "Say hi."}],
        tools=[_WEATHER_TOOL],
        execute=lambda name, args: "n/a",
    )
    assert (usage.input_tokens is None) == (usage.output_tokens is None)
    if usage.input_tokens is not None:
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set ANTHROPIC_API_KEY to test the litellm#5063 zero-arg-tool-call shape",
)
def test_anthropic_zero_argument_tool_call():
    # BerriAI/litellm#5063: a zero-arg tool call under stream=True can send
    # "arguments": "" instead of "{}". The tool must still execute with {} — not
    # raise, and not silently skip the call.
    backend = LiteLLMBackend(AnthropicSpec())
    executed = []
    ping_tool = {
        "name": "ping",
        "description": "A no-argument health check.",
        "input_schema": {"type": "object", "properties": {}},
    }

    text, _usage = backend.run_tools(
        system="You must call ping to answer. It takes no arguments.",
        messages=[{"role": "user", "content": "Ping me, then say done."}],
        tools=[ping_tool],
        execute=lambda name, args: executed.append((name, args)) or "pong",
    )

    assert executed == [("ping", {})]
    assert text
