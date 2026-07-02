# PaperLens

Local, config-driven RAG app over arXiv model technical reports. Ask questions
in a chat UI; an LLM does agentic retrieval over the papers, answers with
**clickable citations** that jump to the exact passage, browse the full papers,
and watch ingestion progress in an admin panel. Everything is driven by a single
`config.yaml`.

## How it works

```
config.yaml ─┬─> ingestion worker: download → markdown (Docling) → index (Chroma) → LLM tags
             └─> FastAPI backend ── agentic RAG ──> LLM (Claude | any OpenAI-compatible server)
                       │  tool: search_papers → Searcher (bge-m3 + bge-reranker-v2-m3)
                       └─> React + Vite + Mantine UI: Chat · Papers · Admin
```

- **Chunking** is section-aware with hierarchical breadcrumbs rebuilt from the
  paper's section numbering (`src/rag/chunking.py`).
- **Retrieval** is two-stage: dense `bge-m3` + `bge-reranker-v2-m3` cross-encoder
  (`src/rag/search.py`).
- **Tags** are LLM-generated at ingestion and power a tag filter that restricts
  search to matching papers.
- **The LLM is an opaque config value** (`provider`, `api_base`, `model`) — point
  it at Anthropic, or any OpenAI-compatible server (LM Studio, Ollama, vLLM, cloud).

## Setup

```bash
# Python — editable install (puts the `rag` + `server` packages on the path
# and installs the console scripts). Run inside your venv.
pip install -e .

# Frontend deps
npm --prefix web install

# ...or both at once:
make install
```

> The editable install is required: the packages live under `src/`, so
> `paperlens-serve` / `python -m server` only work after `pip install -e .`.

Provide LLM credentials in a `.env` (copy `.env.example`) if using a cloud
provider — local servers need no key.

## Configuration — `config.yaml`

Single source of truth: paths, embedder, reranker, tagging/chat LLMs, server,
and the **paper list** (arXiv id + name). Add a paper by adding a line and
hitting *Re-scan* in the admin panel (or restarting). Nothing is hardcoded in
scripts. The chat model must support tool/function calling (the agent calls a
`search_papers` tool).

## Running

```bash
# 1) Start the app — this ALSO auto-starts the ingestion worker.
#    On first run the DB is empty; the UI says so while papers ingest.
paperlens-serve            # serves http://127.0.0.1:8000
                                 # (equivalently: python -m server)

# 2) Frontend (dev, hot reload, proxies /api → backend)
npm --prefix web run dev         # http://localhost:5173

# For a single-origin production build served by FastAPI:
npm --prefix web run build       # → web/dist, served at http://127.0.0.1:8000

# ...or run backend + frontend dev server together:
make dev
```

Commands are CWD-independent — `config.yaml` is found by searching upward from
the working directory (override with `--config` or `$PAPERLENS_CONFIG`).

Headless ingestion (same pipeline as the worker), without the server:

```bash
paperlens-ingest           # ingest every config paper not yet in the DB
paperlens-ingest --retag   # regenerate tags for existing papers (no re-index)
```

## Pages

- **Chat** — agentic RAG with streaming answers; `[rN]` citations are clickable
  and open the paper with the passage highlighted. Optional tag filter scopes
  the search.
- **Papers** — every paper in the DB with tags; open one to read the full
  markdown (tables + LaTeX rendered).
- **Admin** — paper/chunk counts, tag explorer, pending papers, and a live
  ingestion progress bar.

## Layout

```
config.yaml            # all configuration + paper list (project root; anchors relative paths)
pyproject.toml         # packaging, deps, console scripts
Makefile               # install / dev / ingest / build
src/
  rag/                 # retrieval + ingestion core
    config.py          # typed config loader, root anchoring (+ .env)
    chunking.py        # section-aware chunking + breadcrumbs
    embedders.py       # pluggable HF / OpenAI embedders
    search.py          # Searcher: retrieval + rerank (+ paper_ids filter)
    index.py           # chunk → embed → upsert (Chroma)
    extract.py         # PDF → markdown (Docling)
    tagger.py          # LLM tag generation
    llm.py             # provider-agnostic LLM client (tool-use loop)
    manifest.py        # papers.json (paper metadata + tags)
    pipeline.py        # download → md → index → tag → manifest
    ingest.py          # headless ingestion CLI (+ --retag)
  server/              # FastAPI backend + ingestion worker
    main.py  agent.py  worker.py  chats.py  schemas.py
web/                   # Vite + React + Mantine frontend
data/                  # git-ignored runtime: papers/, rag_db/, chat_history/
docs/                  # notes (e.g. done.md)
```
