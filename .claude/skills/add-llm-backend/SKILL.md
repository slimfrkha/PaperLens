---
name: add-llm-backend
description: Add a new LLM provider backend to PaperLens through the ChoiceRegistry pattern. Use when the user wants to support a new chat/tagging LLM provider (a new `type:` value in config.yaml's `llm` section) — one @LLMSpec.register_subclass dataclass in src/rag/config.py, one LLMBackend subclass + build_llm match arm in src/rag/llm.py, a unit test, and the gate.
---

# Add an LLM backend to PaperLens

An **LLM backend** is a provider adapter behind a uniform interface, selected by the LLM
spec's `type` config string and built by `build_llm`. It powers chat, tagging, and the
`llm` reranker. Backends are modelled as a `draccus.ChoiceRegistry`: a `type` decodes to an
`LLMSpec` variant dataclass, and `build_llm` matches that variant to a concrete backend.
Read [CONTEXT.md](../../../CONTEXT.md) for the exact vocabulary before naming anything.

**Before starting:** confirm the provider's wire format. If it speaks the **OpenAI wire
format** (most local servers and many hosted APIs do), you almost certainly do **not** need
a new backend — set `type: openai` (or subclass `OpenAISpec` / `OpenAICompatBackend`, as
`VLLMSpec` / `VLLMBackend` do) and stop. Only write a fresh backend for a genuinely
different API (like `AnthropicBackend`).

## Checklist

1. **Read the existing backends** in `src/rag/llm.py`: `LLMBackend` (the ABC),
   `AnthropicBackend` (a bespoke API), and `OpenAICompatBackend` + its subclasses; and the
   `LLMSpec` variants in `src/rag/config.py`. Match their shapes exactly — don't invent one.

2. **Register an `LLMSpec` variant** in `src/rag/config.py` carrying only the fields this
   provider needs (the base already has `model`, `max_tokens`, `temperature`,
   `api_key_env`):

   ```python
   @LLMSpec.register_subclass("myprovider")
   @dataclass
   class MySpec(LLMSpec):
       api_key_env: str = "MYPROVIDER_API_KEY"   # override defaults; add api_base only if OpenAI-compatible
   ```

   - `"myprovider"` is the string users put under `llm.chat.type` / `llm.tagging.type`.
     Keep it short and lowercase.
   - **Do not add an `Optional`/`X | None` field** — draccus can't register those on the
     CLI. Use `str = ""` as an "unset" sentinel (as `OpenAISpec.api_base` does) and
     normalise with `... or None` at the build site.

3. **Subclass `LLMBackend`** in `src/rag/llm.py` and **add a `build_llm` match arm**:

   ```python
   class MyBackend(LLMBackend):
       def __init__(self, spec: LLMSpec): ...
       def complete(self, system: str, user: str, *, max_tokens: int) -> str: ...
       def run_tools(self, *, system, messages, tools, execute, on_text, on_reasoning): ...

   # in build_llm(spec):
   #     case MySpec(): return MyBackend(spec)   # subclasses before their base
   ```

   - **Import the provider SDK lazily** (inside `__init__`/methods, not at module top) so a
     setup that doesn't use this provider never pays for the import. If it's a cloud SDK,
     add it as an **optional extra** in `pyproject.toml`
     (`[project.optional-dependencies]`) and to the `all` extra, mirroring `anthropic` /
     `gemini`.
   - Read the API key from `spec.api_key_env` (an env var name), never a hard-coded key.
   - `run_tools` drives the ReAct loop for chat — it must call `execute` for each tool call
     and stream text through `on_text`. Study `AnthropicBackend.run_tools` for the contract.

4. **Export the new variant** from `src/rag/__init__.py` (add `MySpec` to the `.config`
   import block and `__all__`), mirroring the other spec variants.

5. **Add a unit test** near `tests/unit/test_llm.py`: assert `build_llm(MySpec())` returns
   `MyBackend`, and add the variant to the dispatch parametrization. Use the `fake_llm`
   pattern; guard SDK-dependent tests with `pytest.importorskip("mysdk")`. **No live API
   calls, no downloads.**

6. **Update the docs** (user-facing change):
   - [docs/how-to.md](../../../docs/how-to.md) — *Switch the chat or tagging LLM* and/or the
     *Add a new LLM backend* recipe.
   - [docs/configuration.md](../../../docs/configuration.md) — list the new `type` value for
     the `llm` keys.

7. **Try it end to end:** set `type: myprovider` in a `config.yaml` `llm` spec, run
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
