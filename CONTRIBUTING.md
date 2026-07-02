# Contributing to PaperLens

**For:** anyone changing PaperLens code. This is the dev setup, the rules that keep the
codebase coherent, and the gate every change must pass. New to the domain? Skim
[CONTEXT.md](CONTEXT.md) (glossary) and [docs/architecture.md](docs/architecture.md) first.

## 🛠️ Dev setup

```bash
uv sync                       # venv + locked deps + the paperlens-* scripts
npm --prefix web install      # frontend deps
uv run pre-commit install     # fast auto-fixing hooks on every commit (see below)
```

Optional cloud LLM backends install as extras: `uv sync --extra anthropic`,
`uv sync --extra gemini`, or `uv sync --all-extras`. Run everything through uv
(`uv run <cmd>`) or activate `.venv`.

## ✅ The gate

Four commands define "done". They are identical locally and in CI — run them before
considering any change finished:

```bash
uv run ruff format --check src   # formatting
uv run ruff check src            # lint
uv run ty check src              # type check
uv run pytest                    # tests
```

Auto-fix the first two with `uv run ruff format src` and `uv run ruff check --fix src`.

`pre-commit install` wires the **fast, auto-fixing subset** (ruff format + `ruff check
--fix`, codespell, whitespace/YAML/TOML hygiene) into every commit and blocks direct
commits to `main`. `ty` and `pytest` stay **out of the hooks** (CI only) so commits stay
fast — you still run the full gate yourself before pushing.

## 🧱 Project layout

```text
config.yaml            # single source of truth (paths, models, server, paper list)
src/
  rag/                 # config-driven core: ingestion + two-stage retrieval
    config.py          # typed config loader, root anchoring (+ .env)
    chunking.py        # section-aware chunking + breadcrumbs
    extract.py         # PDF → markdown (Docling)
    embedders.py       # pluggable embedders (hf | openai | gemini | ollama)
    reranker.py        # pluggable rerankers (hf cross-encoder | llm)
    index.py           # chunk → embed → upsert (Chroma)
    llm.py             # provider-agnostic LLM backends (tool-use loop)
    manifest.py        # papers.json (paper metadata + tags)
    search.py          # Searcher: retrieval + rerank (+ paper filter)
    tagger.py          # LLM tag generation
    pipeline.py        # download → extract → index → tag → manifest
    ingest.py          # headless ingestion CLI (+ --retag)
  server/              # FastAPI backend + in-process ingestion worker (composes rag)
    main.py agent.py worker.py chats.py schemas.py
web/                   # Vite + React + Mantine frontend
tests/                 # unit/ + integration/
docs/                  # documentation hub (see docs/README.md)
data/                  # git-ignored runtime: papers/, rag_db/, chat_history/
```

Import the public API from the package (`from rag import Searcher, load_config`), not the
leaf modules — the internal layout may change.

## 🧭 Architecture you must preserve

The `rag` modules import in **one direction only — no cycles**. The graph is documented in
`src/rag/__init__.py` and explained in [docs/architecture.md](docs/architecture.md):

```text
config  chunking  embedders  extract  manifest   →   llm  index  reranker
   →   tagger  search  pipeline   →   ingest
```

`server` **composes** `rag` behind an HTTP API; `rag` never imports `server`. Don't add an
import that violates this.

## ✍️ Code style

- **Surgical changes.** Touch only what the task requires. Don't reformat, rename, or
  "improve" adjacent code — it buries the real change in the diff. Match the surrounding
  style even if you'd do it differently.
- **Absolute intra-package imports** within `rag`/`server`; re-export public names from the
  package `__init__`.
- **ruff** owns formatting and lint (config in `pyproject.toml`: line length 100,
  double quotes, rules `E,F,I,UP,B,SIM`). Don't hand-format against it.
- **`# type: ignore` / `# ty: ignore` sparingly**, each with a one-line reason for a known
  checker false positive. Don't add new ones without a reason.

## ➕ How to add X

Adding a paper, an LLM backend, or an embedder is a documented recipe — see
[docs/how-to.md](docs/how-to.md). Backends use the **registry** pattern: one
`@register_*`-decorated class, no other wiring.

## 🧪 Test conventions

- Layout: `tests/unit/` (fast, isolated) and `tests/integration/` (end-to-end). **No
  `__init__.py`** — pytest uses `--import-mode=importlib`.
- **Factory fixtures** in `tests/conftest.py`: `make_config`, `make_searcher`,
  `seed_chunks` return functions so each test overrides only what it cares about.
- **Offline seams.** Tests inject fakes instead of loading models or hitting the network:
  `fake_embedder` and `fake_llm`, plus a temp Chroma. `Searcher(embedder=...)` and
  `ChatAgent(client=...)` exist for exactly this. Don't write tests that download models or
  call a live API.
- **Optional-extra tests self-skip** with `pytest.importorskip(...)` (e.g. `anthropic`,
  `google.genai`).
- **Coverage** (branch, over `rag` + `server`):

  ```bash
  uv run pytest --cov=rag --cov=server --cov-branch --cov-report=term-missing
  ```

## 📝 Update docs on user-facing changes

When a change alters user-facing behavior — a `config.yaml` key, a command, an API route,
a supported backend, or the ingestion/retrieval flow — **update the docs in the same
change**:

- config keys, commands, API routes → [docs/configuration.md](docs/configuration.md)
- new task or backend → [docs/how-to.md](docs/how-to.md)
- design/behavior change → [docs/architecture.md](docs/architecture.md)
- a new or renamed domain term → [CONTEXT.md](CONTEXT.md)

Docs that contradict the code are worse than no docs. A change isn't done until they agree.
