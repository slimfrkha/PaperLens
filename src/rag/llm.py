"""Provider-agnostic LLM clients with a tool-use loop, backed by LiteLLM.

One neutral tool schema (`{name, description, input_schema}`) drives every backend.
LiteLLM (https://docs.litellm.ai/) normalizes the wire-format differences between
Anthropic, OpenAI-compatible servers (vLLM, SGLang, LM Studio, Ollama's /v1,
llama.cpp, Azure, Mistral, Together, ...), and Gemini, so PaperLens only needs one
`LiteLLMBackend` regardless of provider — adding a LiteLLM-supported provider is a
new `LLMSpec` subclass + one arm in `_litellm_provider()`, not a new backend class.
`complete_structured` layers `instructor` (https://python.useinstructor.com/) on top
of the same LiteLLM call for validated, retrying structured output.

Use `build_llm(spec)` to get the right backend for a config `LLMSpec`.
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import instructor
import litellm
from pydantic import BaseModel

from .config import AnthropicSpec, GeminiSpec, LLMSpec, OpenAISpec

T = TypeVar("T", bound=BaseModel)

# A tool the model may call. `input_schema` is a JSON Schema object.
Tool = dict  # {"name": str, "description": str, "input_schema": dict}
ToolExecutor = Callable[[str, dict], str]  # (name, args) -> result text
OnText = Callable[[str], None]  # streamed text / reasoning deltas
# Polled between/within streaming rounds; True -> stop early. Only catches a stop that
# lands between chunks that actually arrived — a call stuck producing nothing (e.g. a
# local model's prefill) isn't interrupted at this layer. `chat_turn._run_agent` is what
# bounds that case, by giving up *waiting* on the call rather than depending on it to
# return; this flag is what lets it return the right (possibly partial) text promptly
# whenever the call *is* still responsive.
StopCheck = Callable[[], bool]


@dataclass
class Usage:
    """Token usage for one `run_tools` call, summed across ReAct rounds.

    `None` means the backend never reported usage for this call (e.g. a local
    server that ignores `stream_options`) — kept distinct from `0`, which would
    misleadingly claim a free/zero-token turn.
    """

    input_tokens: int | None
    output_tokens: int | None


def _api_key(spec: LLMSpec) -> str:
    key = os.environ.get(spec.api_key_env)
    if key:
        return key
    # Local OpenAI-compatible servers (vLLM, SGLang, LM Studio, Ollama) ignore it.
    if isinstance(spec, OpenAISpec) and spec.api_base:
        return "local-no-key"
    raise RuntimeError(
        f"No API key in ${spec.api_key_env} for provider "
        f"{LLMSpec.get_choice_name(type(spec))!r}. "
        f"Export it (or add it to .env), or point config.llm at a local server."
    )


class LLMBackend(ABC):
    """Common interface: a plain completion (tagging), structured output (tagging /
    query expansion), and a tool-use loop (chat)."""

    def __init__(self, spec: LLMSpec):
        self.spec = spec

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str: ...

    @abstractmethod
    def complete_structured(
        self,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int | None = None,
        max_retries: int = 2,
    ) -> T:
        """Like `complete`, but validated against `response_model` with `max_retries`
        re-prompts (feeding the LLM its own validation error) on a bad reply."""
        ...

    @abstractmethod
    def run_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[Tool],
        execute: ToolExecutor,
        on_text: OnText | None = None,
        on_reasoning: OnText | None = None,
        max_rounds: int = 8,
        stop_check: StopCheck | None = None,
    ) -> tuple[str, Usage]: ...


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_PLACEHOLDER = "THINK_TAG"


class _ThinkTagStripper:
    """Streaming filter for inline `<think>...</think>` chain-of-thought that some
    OpenAI-compatible reasoning models emit inside `content` instead of the
    structured `reasoning_content` field. Feed each content fragment via `feed()`;
    it returns only the visible part of that fragment, safe to stream immediately —
    a complete `<think>...</think>` span is replaced with `THINK_TAG` (no angle
    brackets, so it can't be mistaken for Markdown/HTML or a `[rN]` citation marker
    downstream). Extracted think text accumulates in `.reasoning`. A tag split
    across delta boundaries is held in a bounded lookback buffer (<=6 chars) until
    it resolves. `finish()` flushes the tail at end-of-stream: if a `<think>` was
    never closed, its trailing content is folded into `.reasoning` with no
    placeholder emitted — there's no well-formed span to replace.
    """

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""
        self.reasoning = ""

    def feed(self, text: str) -> str:
        self._buf += text
        visible_parts: list[str] = []
        while True:
            tag = _THINK_CLOSE if self._inside else _THINK_OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                break
            before, self._buf = self._buf[:idx], self._buf[idx + len(tag) :]
            if self._inside:
                self.reasoning += before
                visible_parts.append(_THINK_PLACEHOLDER)
            else:
                visible_parts.append(before)
            self._inside = not self._inside

        tag = _THINK_CLOSE if self._inside else _THINK_OPEN
        hold = 0
        for k in range(min(len(tag) - 1, len(self._buf)), 0, -1):
            if tag.startswith(self._buf[-k:]):
                hold = k
                break
        split = len(self._buf) - hold
        emit_now, self._buf = self._buf[:split], self._buf[split:]
        if self._inside:
            self.reasoning += emit_now
        else:
            visible_parts.append(emit_now)
        return "".join(visible_parts)

    def finish(self) -> str:
        tail, self._buf = self._buf, ""
        if self._inside:
            self.reasoning += tail
            return ""
        return tail


def _litellm_provider(spec: LLMSpec) -> str:
    """Map an `LLMSpec` variant to its LiteLLM provider prefix."""
    match spec:
        case AnthropicSpec():
            return "anthropic"
        case GeminiSpec():
            return "gemini"
        case OpenAISpec():  # also covers VLLMSpec/SGLangSpec, which subclass it
            return "openai"
        case _:
            raise ValueError(f"Unknown LLM spec: {type(spec).__name__}")


class LiteLLMBackend(LLMBackend):
    """Every provider via LiteLLM's normalized `completion()` interface.

    LiteLLM speaks each provider's native wire format internally and normalizes the
    response to one OpenAI-shaped `ModelResponse`/streamed-chunk surface, so a single
    implementation covers Anthropic, OpenAI-compatible servers, and Gemini alike.
    """

    def __init__(self, spec: LLMSpec):
        super().__init__(spec)
        self._model = f"{_litellm_provider(spec)}/{spec.model}"
        # JSON_SCHEMA, not the from_litellm default (TOOLS): local OpenAI-compatible
        # servers (LM Studio, vLLM, ...) reject TOOLS' object-shaped `tool_choice` and
        # JSON mode's `response_format` outright — JSON_SCHEMA is the one confirmed to
        # work against a local server, not just against Anthropic/OpenAI proper.
        self._instructor = instructor.from_litellm(
            litellm.completion, mode=instructor.Mode.JSON_SCHEMA
        )

    def _kwargs(self) -> dict:
        kwargs: dict = {"model": self._model, "api_key": _api_key(self.spec)}
        if isinstance(self.spec, OpenAISpec) and self.spec.api_base:
            kwargs["api_base"] = self.spec.api_base
            kwargs["custom_llm_provider"] = "openai"
        if self.spec.timeout > 0:
            kwargs["timeout"] = self.spec.timeout
        if self.spec.max_retries >= 0:
            kwargs["num_retries"] = self.spec.max_retries
        return kwargs

    def complete(self, system, user, max_tokens=None):
        resp = litellm.completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens or self.spec.max_tokens,
            temperature=self.spec.temperature,
            **self._kwargs(),
        )
        return resp.choices[0].message.content or ""

    def complete_structured(self, system, user, response_model, max_tokens=None, max_retries=2):
        return self._instructor.chat.completions.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=response_model,
            max_tokens=max_tokens or self.spec.max_tokens,
            temperature=self.spec.temperature,
            max_retries=max_retries,
            **self._kwargs(),
        )

    def run_tools(
        self,
        system,
        messages,
        tools,
        execute,
        on_text=None,
        on_reasoning=None,
        max_rounds=8,
        stop_check=None,
    ):
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        convo = [{"role": "system", "content": system}] + [dict(m) for m in messages]
        final_text = ""
        total_in = 0
        total_out = 0
        usage_complete = True  # False if any round's server never reports usage

        def usage() -> Usage:
            return Usage(total_in, total_out) if usage_complete else Usage(None, None)

        for _ in range(max_rounds):
            if stop_check and stop_check():
                return final_text, usage()
            stream = litellm.completion(
                messages=convo,
                tools=oai_tools,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                stream=True,
                stream_options={"include_usage": True},
                **self._kwargs(),
            )
            content = ""
            reasoning = ""
            tool_calls: dict[int, dict] = {}
            think_filter = _ThinkTagStripper()
            round_usage = None
            stopped = False
            for chunk in stream:
                # The usage-only final chunk (when the server honors stream_options)
                # carries an empty `choices` list — capture it before skipping below.
                if getattr(chunk, "usage", None):
                    round_usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    visible = think_filter.feed(delta.content)
                    if visible:
                        content += visible
                        if on_text:
                            on_text(visible)
                # Reasoning models expose their chain-of-thought here (if at all).
                rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if rc:
                    reasoning += rc
                for tc in delta.tool_calls or []:
                    slot = tool_calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
                if stop_check and stop_check():
                    stopped = True
                    break

            if stopped:
                # This only catches a stop observed between chunks that actually
                # arrived — a call stuck producing nothing (e.g. a local model's
                # prefill) isn't interrupted here; `chat_turn._run_agent` is what
                # bounds that case, by giving up *waiting* on this call rather than
                # depending on it to return.
                close = getattr(stream, "close", None)
                if close:
                    close()
                tail = think_filter.finish()
                if tail and on_text:
                    on_text(tail)
                final_text = content + tail
                return final_text, usage()

            tail = think_filter.finish()
            if tail:
                content += tail
                if on_text:
                    on_text(tail)
            reasoning += think_filter.reasoning

            if reasoning and on_reasoning:
                on_reasoning(reasoning)

            if round_usage:
                total_in += round_usage.prompt_tokens
                total_out += round_usage.completion_tokens
            else:
                usage_complete = False

            final_text = content
            if not tool_calls:
                return final_text, usage()

            convo.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                        }
                        for c in tool_calls.values()
                    ],
                }
            )
            for c in tool_calls.values():
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": execute(c["name"], args),
                    }
                )
        return final_text, usage()


def build_llm(spec: LLMSpec) -> LLMBackend:
    """Instantiate the LiteLLM-backed backend for a config LLMSpec variant."""
    _litellm_provider(spec)  # eager validation — fail fast on an unknown spec, as before
    return LiteLLMBackend(spec)


def _selftest() -> None:
    """Round-trip a trivial tool call against the configured chat model."""
    from .config import load_config

    spec = load_config().llm.chat
    provider = LLMSpec.get_choice_name(type(spec))
    print(f"provider={provider} model={spec.model} api_base={getattr(spec, 'api_base', None)}")
    llm = build_llm(spec)
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    def execute(name: str, args: dict) -> str:
        print(f"\n[tool call] {name}({args})")
        return "sunny, 25C"

    text, usage = llm.run_tools(
        system="You are a helpful assistant. Use tools when needed.",
        messages=[
            {"role": "user", "content": "What's the weather in Paris? Use the tool, then answer."}
        ],
        tools=tools,
        execute=execute,
        on_text=lambda t: print(t, end="", flush=True),
    )
    print(f"\n\n--- final ---\n{text}\n(usage: {usage})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
