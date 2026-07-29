# 🔎 PaperLens

> **Local, config-driven RAG over arXiv model technical reports.**

Ask questions in a chat UI; an LLM does agentic retrieval over the papers and
answers with **clickable citations** 📎 that jump to the exact passage. Browse the
full papers, and watch ingestion progress in an admin panel. Everything is driven
by a single `config.yaml`. ⚙️

> 🤖 This repo is **vibecoded with [Claude Code](https://claude.com/claude-code)**.

## 🛠️ How it works

```text
config.yaml ─┬─> ingestion worker: download → markdown (Docling) → index (Chroma) ‖ LLM tags
             └─> FastAPI backend ── agentic RAG ──> LLM (Claude | any OpenAI-compatible server)
                       │  tool: search_papers → Searcher (embedder + reranker)
                       └─> React + Vite + Mantine UI: Chat · Papers · Admin
```

- ✂️ **Chunking** is section-aware with hierarchical breadcrumbs rebuilt from the
  paper's section numbering (`src/rag/chunking.py`).
- 🎯 **Retrieval** is two-stage: dense embedding recall + a cross-encoder reranker
  (`src/rag/search.py`). Both stages are config-swappable — the embedder
  (`embedding.type`: `hf` · `openai` · `gemini` · `ollama`) and the reranker
  (`reranker.type`: `hf` cross-encoder · `llm`, which reuses the chat model).
- 🏷️ **Tags** are LLM-generated at ingestion and power a tag filter that restricts
  search to matching papers.
- 🔌 **The LLM is an opaque config value** (`provider`, `api_base`, `model`) — point
  it at Anthropic, or any OpenAI-compatible server (LM Studio, Ollama, vLLM, cloud).

## 📦 Setup

```bash
# Python — creates a uv-managed venv and installs the locked deps (puts the
# `rag` + `server` packages on the path and installs the console scripts).
uv sync

# Optional cloud LLM backends (only if you use them):
uv sync --extra anthropic   # anthropic provider
uv sync --extra gemini      # gemini provider
uv sync --all-extras        # both

# Frontend deps
npm --prefix web install

# ...or Python + frontend at once:
make install
```

> 💡 `uv sync` installs the packages from `src/` and the console scripts, so
> `uv run paperlens-serve` / `uv run python -m server` work from a clean checkout.
> Run project commands with `uv run <cmd>` (or activate `.venv`).

> 🔑 Provide LLM credentials in a `.env` (e.g. `ANTHROPIC_API_KEY=...`) if using a
> cloud provider — local servers need no key.

> ⚠️ **Running models locally? Check your hardware first.** The default embedder and
> reranker load onto your machine, and pointing `llm` at a local
> server (LM Studio, Ollama, vLLM) loads a chat model on top of that. Make sure your
> RAM/VRAM can hold everything you've chosen before you start — an oversized model
> will swap, crawl, or get OOM-killed mid-ingestion. On Apple Silicon, also mind the
> Metal per-tensor cap: keep `embedding.max_seq_length` at its default (1024). Pick
> smaller models or use a cloud provider if your machine can't carry the load.

## 🚀 Quickstart

```bash
cp configs/examples/local-gpt-oss.yaml configs/my-setup.yaml   # pick a template, edit papers:
make serve CONFIG=configs/my-setup.yaml     # serves http://127.0.0.1:8000, auto-starts ingestion
npm --prefix web run dev                    # UI at http://localhost:5173 (proxies /api → backend)
```

`make serve CONFIG=...` runs `paperlens-serve --config_path ...`; there's no default
config, so `CONFIG=` (or `PAPERLENS_CONFIG`/discovery) is required. Or call
`uv run paperlens-serve --config_path <path>` directly.

Open the UI, watch papers ingest on the **Admin** page, then ask a question on
**Chat**. Full walkthrough with expected output:
[Getting started](docs/getting-started.md). 🎓

Everything is driven by one YAML config (under `configs/`) — paths, models, the
server, and the paper list. The chat model must support tool/function calling (the
agent calls a `search_papers` tool). Every key, command, and API route:
[Configuration & commands](docs/configuration.md).

## 🖥️ Pages

- 💬 **Chat** — agentic RAG with streaming answers; `[rN]` citations are clickable
  and open the paper with the passage highlighted. Optional tag filter scopes
  the search.
- 📄 **Papers** — every paper in the DB with tags; open one to read the full
  markdown (tables + LaTeX rendered).
- 📊 **Admin** — paper/chunk counts, tag explorer, pending papers, and a live
  ingestion progress bar.

## 📚 Documentation

Full docs live in [`docs/`](docs/README.md):

| Page | What you'll find |
|---|---|
| 🎓 [Getting started](docs/getting-started.md) | Install and ask your first question. |
| ⚙️ [Configuration & commands](docs/configuration.md) | Every `config.yaml` key, command, and API route. |
| 🧩 [How-to guides](docs/how-to.md) | Add papers, swap backends, use a cloud provider. |
| 🏛️ [Architecture](docs/architecture.md) | Chunking, two-stage retrieval, the agent loop. |
| 🤝 [CONTRIBUTING](CONTRIBUTING.md) | Dev setup, the gate, conventions. |
| 📖 [CONTEXT](CONTEXT.md) | Domain glossary. |

## 🤝 Contributing

Dev setup, the annotated project layout, the module layering rule, and the
4-command gate (`ruff format --check` · `ruff check` · `ty check` · `pytest`)
live in [CONTRIBUTING.md](CONTRIBUTING.md). New to the codebase? Start with the
[domain glossary](CONTEXT.md). 📖

## 📜 License

See [LICENSE](LICENSE).
