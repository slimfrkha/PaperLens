# CLAUDE.md — PaperLens operational map

Local, config-driven **agentic RAG** over arXiv model technical reports. A single YAML
config under `configs/` is the source of truth; two flows hang off it — **ingestion** (arXiv PDF → Docling
markdown → chunk → embed → Chroma index → LLM tags → manifest) and **retrieval** (a
FastAPI backend whose `ChatAgent` runs a ReAct loop with one `search_papers` tool over a
two-stage `Searcher`). A Vite + React frontend streams the answer and its
Thought → Action → Observation trace over SSE. Backend is Python (`rag` core + `server`);
frontend is `web/`.

**New here?** Read [CONTEXT.md](CONTEXT.md) (domain glossary — use these exact terms) and
[docs/architecture.md](docs/architecture.md) (why it's built this way) first.

## ✅ The gate — run before calling anything done

Four commands define "done", identical locally and in CI:

```bash
uv run ruff format --check src   # formatting
uv run ruff check src            # lint
uv run ty check src              # type check
uv run pytest                    # tests
```

Auto-fix the first two with `uv run ruff format src` / `uv run ruff check --fix src`.
`pre-commit` runs the fast auto-fixing subset on commit; `ty` + `pytest` are yours to run.

`web/` has its own 4-command gate, identical locally and in CI:

```bash
npm --prefix web run format:check   # prettier
npm --prefix web run lint           # eslint
npm --prefix web run typecheck      # tsc --noEmit
npm --prefix web run test           # vitest
```

Auto-fix with `npm --prefix web run format` / `npm --prefix web run lint -- --fix`.
`pre-commit` runs the fast auto-fixing subset on commits touching `web/`; `typecheck` +
`test` are yours to run.

## 🏃 Common commands

```bash
uv sync                          # venv + locked deps + paperlens-* scripts
npm --prefix web install         # frontend deps
uv run paperlens-serve           # backend only (FastAPI, port 8000)
uv run paperlens-ingest          # ingest configured papers not yet in the DB
uv run paperlens-ingest --retag  # regenerate tags without re-indexing
make dev                         # backend + Vite dev server together
```

Cloud LLM backends are lazy extras: `uv sync --extra anthropic` / `--extra gemini` /
`--all-extras`. Keys go in `.env`.

## 🧱 Layout

```text
configs/               # run configs (data, not code); default: recent-oss-agentic-models.yaml
  examples/            # copy-me templates + reference.yaml (every key annotated)
src/
  rag/                 # config-driven core: ingestion + two-stage retrieval
    config.py          # typed config loader, project-root anchoring (+ .env)
    chunking.py        # section-aware chunking + breadcrumbs
    extract.py         # PDF → markdown (Docling)
    embedders.py       # pluggable embedders (hf | openai | gemini | ollama)
    reranker.py        # pluggable rerankers (hf cross-encoder | llm)
    index.py           # chunk → embed → upsert (Chroma)
    llm.py             # provider-agnostic LLM backends (tool-use loop)
    manifest.py        # papers.json (paper metadata + tags)
    search.py          # Searcher: dense recall → rerank (+ paper filter)
    tagger.py          # LLM tag generation
    pipeline.py        # ingest_paper: download → extract → index → tag → manifest
    ingest.py          # headless ingestion CLI (+ --retag)
  server/              # FastAPI backend + in-process ingestion worker (composes rag)
    main.py            # create_app: wires manifest, worker, lazy ChatAgent; all routes
    agent.py           # ChatAgent: ReAct loop, search_papers tool, ref/citation registry
    worker.py          # IngestionWorker: background thread over pending papers
    chats.py schemas.py
web/                   # Vite + React + Mantine frontend (SSE chat, trace, paper viewer)
tests/                 # unit/ + integration/ (no __init__.py; importlib mode)
docs/                  # documentation hub (docs/README.md)
data/                  # git-ignored runtime: papers/, rag_db/, chat_history/
```

Import the public API from the package (`from rag import Searcher, load_config`), **not**
the leaf modules — the internal layout may change.

## 🧭 Architecture invariant — do not break

`rag` modules import in **one direction only, no cycles** (graph in `src/rag/__init__.py`):

```text
config  chunking  embedders  extract  manifest  →  llm  index  reranker
   →  tagger  search  pipeline  →  ingest
```

`server` **composes** `rag` behind the HTTP API; `rag` **never** imports `server`. Don't
add an import that violates this.

The ingestion core (`pipeline`, `ingest`, the server's `worker`) is typed to `IngestConfig`
— a frozen projection of `Config` (`Config.for_ingest()`) exposing only ingestion's fields
(`paths`, `collection`, `embedding`, `tagging`, `papers`). Serve uses the full `Config`;
there is deliberately **no** `ServeConfig` (the server hosts the worker, so it reads every
field). Don't widen `IngestConfig` or add a `ServeConfig` for symmetry.

## 💡 Key design facts / gotchas

- **MPS tensor cap.** `bge-m3` defaults to an 8192-token sequence length, which overflows
  Metal's `2**32`-byte per-tensor limit on Apple Silicon at a normal batch size.
  `embedding.max_seq_length` (default 1024, `hf` only) caps it; chunks stay well under.
  Don't raise it blindly on Apple Silicon.
- **Lazy heavy models, warmed at startup.** The cross-encoder and embedder load on first
  use and the `ChatAgent` is built once (lazily, under a lock), but the server warms them in
  a background thread at startup (`warm_models` in `main.py`, a tiny dummy search) so the
  first `/api/chat` skips the 20-30s load while startup stays instant. Cloud clients are
  optional extras imported lazily — an OpenAI-compatible or cloud setup never pays for local
  model downloads.
- **Config anchoring.** Every relative path in `config.yaml` resolves against the config
  file's directory (the **project root**), so every entry point is CWD-independent.
  Located by `--config_path` → `PAPERLENS_CONFIG` → upward search from CWD. Config is
  dataclasses decoded by `draccus` with OmegaConf `${...}` interpolation; `parse_config`
  adds per-field CLI overrides (`--server.port=…`) that feed interpolation.
- **SSE streaming.** `/api/chat` streams tokens and trace steps over Server-Sent Events;
  the agent runs in a thread executor and pushes events onto an asyncio queue. The UI
  renders the answer and the Thought → Action → Observation trace as they happen.
- **Embedder identity is baked into the index.** The embedder's `name()` namespaces the
  Chroma collection. Changing the embedder means re-indexing (delete `paths.rag_db` and
  re-ingest, or use a fresh `collection` name).
- **ChoiceRegistry pattern.** Embedders, rerankers, and LLM backends are selected by a
  config `type` string and modelled as `draccus.ChoiceRegistry` variant dataclasses
  (`EmbeddingCfg`/`RerankerCfg`/`LLMSpec` in `config.py`). Adding a backend is one
  `@Base.register_subclass("name")` dataclass + a `build_*` match arm — no `if/elif`; an
  unknown `type` or stray field fails loudly at load.

## ✍️ Conventions

- **Surgical changes.** Touch only what the task requires. Don't reformat, rename, or
  "improve" adjacent code — match the surrounding style even if you'd do it differently.
- **Absolute intra-package imports** within `rag`/`server`; re-export public names from the
  package `__init__`.
- **ruff owns formatting and lint** (line length 100, double quotes, `E,F,I,UP,B,SIM`).
  Don't hand-format against it.
- **`# type: ignore` / `# ty: ignore` sparingly**, each with a one-line reason for a known
  checker false positive.
- **Tests stay offline.** Inject `fake_embedder` / `fake_llm` and a temp Chroma via the
  `tests/conftest.py` factory fixtures (`make_config`, `make_searcher`, `seed_chunks`).
  `Searcher(embedder=...)` and `ChatAgent(client=...)` exist for this. Never download
  models or hit a live API in a test.
- **Update docs on user-facing change.** A `config.yaml` key, command, API route, or
  supported backend → [docs/configuration.md](docs/configuration.md); a new task/backend →
  [docs/how-to.md](docs/how-to.md); a design change → [docs/architecture.md](docs/architecture.md);
  a new/renamed term → [CONTEXT.md](CONTEXT.md). A change isn't done until docs agree.

Full contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md). Recipes for adding a paper,
LLM backend, or embedder: [docs/how-to.md](docs/how-to.md).
