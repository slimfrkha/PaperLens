---
name: add-llm-backend
description: Add a new LLM provider backend to PaperLens through the registry pattern. Use when the user wants to support a new chat/tagging LLM provider (a new `provider:` value in config.yaml's `llm` section) — one @register_llm-decorated LLMBackend subclass in src/rag/llm.py, a unit test, and the gate.
---

# Add an LLM backend to PaperLens

An **LLM backend** is a provider adapter behind a uniform interface, selected by the
`provider` config string and built by `build_llm`. It powers chat, tagging, and the `llm`
reranker. Backends are discovered through the **registry** (`@register_llm`) — one
decorated class, no other wiring. Read [CONTEXT.md](../../../CONTEXT.md) for the exact
vocabulary before naming anything.

**Before starting:** confirm the provider's wire format. If it speaks the **OpenAI wire
format** (most local servers and many hosted APIs do), you almost certainly do **not** need
a new backend — set `provider: openai` (or subclass `OpenAICompatBackend`, as
`VLLMBackend` / `SGLangBackend` do) and stop. Only write a fresh backend for a genuinely
different API (like `AnthropicBackend`).

## Checklist

1. **Read the existing backends** in `src/rag/llm.py`: `LLMBackend` (the ABC),
   `AnthropicBackend` (a bespoke API), and `OpenAICompatBackend` + its subclasses. Match
   their method signatures exactly — don't invent a new shape.

2. **Subclass `LLMBackend` and decorate it** in `src/rag/llm.py`:

   ```python
   @register_llm("myprovider")
   class MyBackend(LLMBackend):
       def __init__(self, spec: LLMSpec): ...
       def complete(self, system: str, user: str, *, max_tokens: int) -> str: ...
       def run_tools(self, *, system, messages, tools, execute, on_text, on_reasoning): ...
   ```

   - `"myprovider"` is the string users will put under `llm.chat.provider` /
     `llm.tagging.provider`. Keep it short and lowercase.
   - **Import the provider SDK lazily** (inside `__init__`/methods, not at module top) so a
     setup that doesn't use this provider never pays for the import. If it's a cloud SDK,
     add it as an **optional extra** in `pyproject.toml`
     (`[project.optional-dependencies]`) and to the `all` extra, mirroring `anthropic` /
     `gemini`.
   - Read the API key from `spec.api_key_env` (an env var name), never a hard-coded key.
   - `run_tools` drives the ReAct loop for chat — it must call `execute` for each tool call
     and stream text through `on_text`. Study `AnthropicBackend.run_tools` for the contract.

3. **`build_llm` needs no edits** — it dispatches on the registry. Confirm the decorator
   ran (the class is imported when `rag.llm` loads).

4. **Add a unit test** near `tests/unit/test_llm.py`. Use the `fake_llm` pattern; if the
   test needs the provider SDK, guard it with `pytest.importorskip("mysdk")` so it
   self-skips when the extra isn't installed. **No live API calls, no downloads.**

5. **Update the docs** (user-facing change):
   - [docs/how-to.md](../../../docs/how-to.md) — add the provider to *Switch the chat or
     tagging LLM* and/or the *Add a new LLM backend* recipe.
   - [docs/configuration.md](../../../docs/configuration.md) — list the new `provider` value
     for the `llm` keys.

6. **Try it end to end:** set `provider: myprovider` in a `config.yaml` `llm` spec, run
   `uv run paperlens-serve`, ask a question, and confirm the agent still calls
   `search_papers` and cites passages. If it never calls the tool, the provider likely
   lacks tool-calling support.

## The gate — required before done

```bash
uv run ruff format --check src
uv run ruff check src
uv run ty check src
uv run pytest
```

All four must pass. See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
