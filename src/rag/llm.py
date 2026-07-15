"""Provider-agnostic LLM clients with a tool-use loop.

One neutral tool schema (`{name, description, input_schema}`) drives every backend.
There are only two real wire formats among the supported providers:

* Anthropic Messages API              -> AnthropicBackend        (type: anthropic)
* OpenAI Chat Completions API         -> OpenAICompatBackend     (type: openai)
    ...which vLLM and SGLang also speak, so they are thin subclasses:
      VLLMBackend  (type: vllm)   /  SGLangBackend (type: sglang)
    The same class also covers LM Studio, Ollama's /v1, llama.cpp, Azure, etc.

Use `build_llm(spec)` to get the right backend for a config `LLMSpec`.
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .config import AnthropicSpec, GeminiSpec, LLMSpec, OpenAISpec, SGLangSpec, VLLMSpec

# A tool the model may call. `input_schema` is a JSON Schema object.
Tool = dict  # {"name": str, "description": str, "input_schema": dict}
ToolExecutor = Callable[[str, dict], str]  # (name, args) -> result text
OnText = Callable[[str], None]  # streamed text / reasoning deltas


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
    """Common interface: a plain completion (tagging) and a tool-use loop (chat)."""

    def __init__(self, spec: LLMSpec):
        self.spec = spec
        # Build the SDK client once and reuse it: each client holds an httpx
        # connection pool (open sockets/fds), so constructing one per call leaks
        # descriptors and eventually raises OSError [too many open files] under a
        # high-volume caller like the eval harness (one call per section).
        self._client_cache: Any = None

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str: ...

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
    ) -> str: ...


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API (tool_use / tool_result blocks, streaming)."""

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        from anthropic import Anthropic

        kwargs: dict = {"api_key": _api_key(self.spec)}
        if self.spec.timeout > 0:
            kwargs["timeout"] = self.spec.timeout
        if self.spec.max_retries >= 0:
            kwargs["max_retries"] = self.spec.max_retries
        self._client_cache = Anthropic(**kwargs)
        return self._client_cache

    def complete(self, system, user, max_tokens=None):
        msg = self._client().messages.create(
            model=self.spec.model,
            max_tokens=max_tokens or self.spec.max_tokens,
            temperature=self.spec.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def run_tools(
        self, system, messages, tools, execute, on_text=None, on_reasoning=None, max_rounds=8
    ):
        client = self._client()
        convo = [dict(m) for m in messages]
        final_text = ""
        for _ in range(max_rounds):
            text_parts: list[str] = []
            with client.messages.stream(
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                system=system,
                tools=tools,
                messages=convo,
            ) as stream:
                for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        text_parts.append(event.delta.text)
                        if on_text:
                            on_text(event.delta.text)
                msg = stream.get_final_message()

            final_text = "".join(text_parts)
            tool_uses = [b for b in msg.content if b.type == "tool_use"]
            if not tool_uses:
                return final_text
            convo.append({"role": "assistant", "content": msg.content})
            convo.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": execute(tu.name, tu.input or {}),
                        }
                        for tu in tool_uses
                    ],
                }
            )
        return final_text


class OpenAICompatBackend(LLMBackend):
    """OpenAI Chat Completions wire format.

    Works for OpenAI and any server that implements it (vLLM, SGLang, LM Studio,
    Ollama /v1, llama.cpp, Azure, Mistral, Together, ...). Set `api_base` to point
    at the server; local servers need no real key.
    """

    spec: OpenAISpec  # always built from an OpenAISpec variant (api_base, ...)

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        from openai import OpenAI

        kwargs: dict = {
            "api_key": _api_key(self.spec),
            "base_url": self.spec.api_base or None,
        }
        if self.spec.timeout > 0:
            kwargs["timeout"] = self.spec.timeout
        if self.spec.max_retries >= 0:
            kwargs["max_retries"] = self.spec.max_retries
        self._client_cache = OpenAI(**kwargs)
        return self._client_cache

    def complete(self, system, user, max_tokens=None):
        resp = self._client().chat.completions.create(
            model=self.spec.model,
            max_tokens=max_tokens or self.spec.max_tokens,
            temperature=self.spec.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def run_tools(
        self, system, messages, tools, execute, on_text=None, on_reasoning=None, max_rounds=8
    ):
        client = self._client()
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
        for _ in range(max_rounds):
            stream = client.chat.completions.create(
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                tools=oai_tools,
                messages=convo,
                stream=True,
            )
            content = ""
            reasoning = ""
            tool_calls: dict[int, dict] = {}
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                    if on_text:
                        on_text(delta.content)
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

            if reasoning and on_reasoning:
                on_reasoning(reasoning)

            final_text = content
            if not tool_calls:
                return final_text

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
        return final_text


class VLLMBackend(OpenAICompatBackend):
    """vLLM's OpenAI-compatible server — same wire format as OpenAI."""


class SGLangBackend(OpenAICompatBackend):
    """SGLang's OpenAI-compatible server — same wire format as OpenAI."""


# ---- Gemini (Google Generative Language API — its own wire format) ---------


def _gemini_type(t: str):
    from google.genai import types

    return {
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "object": types.Type.OBJECT,
        "array": types.Type.ARRAY,
    }.get((t or "string").lower(), types.Type.STRING)


def _gemini_schema(js: dict):
    """Convert a JSON-schema fragment to a Gemini types.Schema."""
    from google.genai import types

    t = (js.get("type") or "object").lower()
    if t == "object":
        props = {k: _gemini_schema(v) for k, v in (js.get("properties") or {}).items()}
        return types.Schema(
            type=types.Type.OBJECT,
            properties=props or None,
            required=js.get("required") or None,
            description=js.get("description"),
        )
    if t == "array":
        return types.Schema(
            type=types.Type.ARRAY,
            items=_gemini_schema(js.get("items") or {"type": "string"}),
            description=js.get("description"),
        )
    return types.Schema(type=_gemini_type(t), description=js.get("description"))


def _gemini_fn(tool: Tool):
    from google.genai import types

    return types.FunctionDeclaration(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=_gemini_schema(tool["input_schema"]),
    )


class GeminiBackend(LLMBackend):
    """Google Gemini via the google-genai SDK.

    Needs a real key in the configured api_key_env (e.g. GEMINI_API_KEY).
    Supports function calling; on 2.5 models it exposes thinking via
    include_thoughts, surfaced through on_reasoning.
    """

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        from google import genai
        from google.genai import types

        http_options = None
        if self.spec.timeout > 0 or self.spec.max_retries >= 0:
            retry_options = None
            if self.spec.max_retries >= 0:
                # Gemini counts total attempts; max_retries counts retries after the first.
                retry_options = types.HttpRetryOptions(attempts=self.spec.max_retries + 1)
            http_options = types.HttpOptions(
                timeout=int(self.spec.timeout * 1000) if self.spec.timeout > 0 else None,
                retry_options=retry_options,
            )
        self._client_cache = genai.Client(api_key=_api_key(self.spec), http_options=http_options)
        return self._client_cache

    def complete(self, system, user, max_tokens=None):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.spec.temperature,
            max_output_tokens=max_tokens or self.spec.max_tokens,
        )
        resp = self._client().models.generate_content(
            model=self.spec.model, contents=user, config=cfg
        )
        return resp.text or ""

    def run_tools(
        self, system, messages, tools, execute, on_text=None, on_reasoning=None, max_rounds=8
    ):
        from google.genai import types

        client = self._client()
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.spec.temperature,
            max_output_tokens=self.spec.max_tokens,
            tools=[types.Tool(function_declarations=[_gemini_fn(t) for t in tools])],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        final_text = ""
        for _ in range(max_rounds):
            text, reasoning, calls = "", "", []
            for chunk in client.models.generate_content_stream(
                model=self.spec.model, contents=contents, config=cfg
            ):
                for cand in chunk.candidates or []:
                    parts = (cand.content.parts if cand.content else None) or []
                    for part in parts:
                        if getattr(part, "function_call", None):
                            calls.append(part.function_call)
                        elif getattr(part, "thought", False) and getattr(part, "text", None):
                            reasoning += part.text
                        elif getattr(part, "text", None):
                            text += part.text
                            if on_text:
                                on_text(part.text)
            if reasoning and on_reasoning:
                on_reasoning(reasoning)
            final_text = text
            if not calls:
                return final_text
            contents.append(
                types.Content(role="model", parts=[types.Part(function_call=fc) for fc in calls])
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"result": execute(fc.name, dict(fc.args or {}))},
                        )
                        for fc in calls
                    ],
                )
            )
        return final_text


def build_llm(spec: LLMSpec) -> LLMBackend:
    """Instantiate the backend for a config LLMSpec variant."""
    match spec:
        case AnthropicSpec():
            return AnthropicBackend(spec)
        case VLLMSpec():  # before OpenAISpec: vllm/sglang subclass it
            return VLLMBackend(spec)
        case SGLangSpec():
            return SGLangBackend(spec)
        case OpenAISpec():
            return OpenAICompatBackend(spec)
        case GeminiSpec():
            return GeminiBackend(spec)
        case _:
            raise ValueError(f"Unknown LLM spec: {type(spec).__name__}")


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

    out = llm.run_tools(
        system="You are a helpful assistant. Use tools when needed.",
        messages=[
            {"role": "user", "content": "What's the weather in Paris? Use the tool, then answer."}
        ],
        tools=tools,
        execute=execute,
        on_text=lambda t: print(t, end="", flush=True),
    )
    print(f"\n\n--- final ---\n{out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
