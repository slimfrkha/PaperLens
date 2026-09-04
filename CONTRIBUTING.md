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

Every LLM backend goes through LiteLLM — no extras to install for any provider. Run
everything through uv (`uv run <cmd>`) or activate `.venv`.

## ✅ The gate

Five commands define "done". They match CI — run them before considering any change
finished:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check src
uv run python scripts/check_docs.py
uv run pytest --cov=rag --cov=server --cov=eval --cov-branch --cov-report=term-missing
```

Auto-fix the first two with `uv run ruff format` and `uv run ruff check --fix`.

`pre-commit install` wires the **fast, auto-fixing subset** (ruff format + `ruff check
--fix`, codespell, whitespace/YAML/TOML hygiene) into every commit and blocks direct
commits to `main`. `ty` and `pytest` stay **out of the hooks** (CI only) so commits stay
fast — you still run the full gate yourself before pushing.

### Frontend gate

`web/` has its own 4-command gate, matching CI:

```bash
npm --prefix web run format:check   # prettier --check
npm --prefix web run lint           # eslint
npm --prefix web run typecheck      # tsc --noEmit
npm --prefix web run test:cov       # vitest + coverage
```

Auto-fix the first two with `npm --prefix web run format` and
`npm --prefix web run lint -- --fix`. The `web-eslint --fix` / `web-prettier --write`
pre-commit hooks run the fast, auto-fixing subset on every commit touching `web/`;
`typecheck` and `test` stay CI-only, same rationale as `ty`/`pytest`.

## 🧱 Project layout

```text
configs/               # run configs (data, not code); examples/ has copy-me templates
  examples/            # copy-me templates + reference.yaml (every key annotated)
src/
  rag/                 # config-driven core: ingestion + two-stage retrieval
    config.py          # typed config loader, root anchoring (+ .env)
    chunking.py        # section-aware chunking + breadcrumbs
    extract.py         # PDF → markdown (Docling)
    embedders.py       # pluggable embedders (hf | openai | gemini | voyage | ollama)
    reranker.py        # pluggable rerankers (hf cross-encoder | llm | voyage)
    index.py           # chunk → embed → upsert (Chroma)
    llm.py             # provider-agnostic LLM backends (tool-use loop)
    manifest.py        # papers.json (paper metadata + tags)
    sparse.py          # BM25 + reciprocal rank fusion for hybrid retrieval
    query_expansion.py # multi-query paraphrase generation
    search.py          # Searcher: dense/hybrid recall → rerank → elbow cutoff
    faithfulness.py    # optional post-generation citation check
    tagger.py          # LLM tag generation
    pipeline.py        # download → extract → index → tag → manifest
    ingest.py          # headless ingestion CLI (+ --retag, --reindex)
  server/              # FastAPI backend + in-process ingestion worker (composes rag)
    main.py agent.py worker.py chats.py schemas.py
  eval/                # per-pool config optimizer (composes rag; see docs/harness.md)
    cli.py fingerprint.py queryset.py genfilter.py metrics.py stats.py
    index_isolated.py optimizer.py harness.py
web/                   # Vite + React + Mantine frontend
tests/                 # unit/ + integration/
docs/                  # documentation hub (see docs/README.md)
data/                  # git-ignored runtime: papers/, rag_db/, chat_history/
evals/                 # git-ignored eval-harness artifacts (per-pool eval sets, confirm log)
```

Import the public API from the package (`from rag import Searcher, load_config`), not the
leaf modules — the internal layout may change.

## 🧭 Architecture you must preserve

The `rag` modules import in **one direction only — no cycles**. The graph is documented in
`src/rag/__init__.py` and explained in [docs/architecture.md](docs/architecture.md):

```text
config  chunking  extract  manifest  sparse  config_writer   →   embedders  llm  index  reranker
   →   tagger  query_expansion  search   →   pipeline   →   ingest
```

`server` **composes** `rag` behind an HTTP API; `rag` never imports `server`. `eval` (the
`paperlens-eval` per-pool config optimizer, see [docs/harness.md](docs/harness.md))
composes `rag` the same way; `rag`/`server` never import `eval`. Don't add an import that
violates either direction.

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
[docs/how-to.md](docs/how-to.md). Backends use the **ChoiceRegistry** pattern: one
`@Base.register_subclass("name")` config-variant dataclass plus a `build_*` match arm.

## 🧪 Test conventions

- Layout: `tests/unit/` (fast, isolated) and `tests/integration/` (end-to-end). **No
  `__init__.py`** — pytest uses `--import-mode=importlib`.
- **Factory fixtures** in `tests/conftest.py`: `make_config`, `make_searcher`,
  `seed_chunks` return functions so each test overrides only what it cares about.
- **Offline seams.** Tests inject fakes instead of loading models or hitting the network:
  `fake_embedder` and `fake_llm`, plus a temp Chroma. `Searcher(embedder=...)` and
  `ChatAgent(client=...)` exist for exactly this. Don't write tests that download models or
  call a live API.
- **HF-specific tests may self-skip** with `pytest.importorskip("sentence_transformers")`
  in a deliberately minimal environment. A normal `uv sync` installs it and every supported
  provider dependency; there are no optional provider extras.
- **Coverage** (branch, over `rag` + `server` + `eval`):

  ```bash
  uv run pytest --cov=rag --cov=server --cov=eval --cov-branch --cov-report=term-missing
  ```

## 📝 Update docs on user-facing changes

When a change alters user-facing behavior — a `config.yaml` key, a command, an API route,
a supported backend, or the ingestion/retrieval flow — **update the docs in the same
change**:

- config keys, commands, API routes → [docs/configuration.md](docs/configuration.md)
- new task or backend → [docs/how-to.md](docs/how-to.md)
- design/behavior change → [docs/architecture.md](docs/architecture.md)
- a new or renamed domain term → [CONTEXT.md](CONTEXT.md)
- a change to `src/eval/` or the `paperlens-eval` flow → [docs/harness.md](docs/harness.md)
- any new user-facing or internal feature → [docs/features.md](docs/features.md)

Docs that contradict the code are worse than no docs. A change isn't done until they agree.
Run `uv run python scripts/check_docs.py` after editing documentation; it checks local links,
heading fragments, code-fence languages, and that reader-facing pages stay within two clicks
of the README.

For the baseline behind the current structure, see the archived
[September 2026 documentation overhaul](docs/audits/2026-09-docs-overhaul.md).
