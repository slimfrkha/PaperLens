# ⚙️ Configuration & commands reference

> 👤 **For:** anyone who needs to look up a `config.yaml` key, a command, an environment
> variable, or an API route. Neutral lookup — for step-by-step tasks see
> [How-to guides](how-to.md).

📌 Every default below is the value in the code (`src/rag/config.py`), which the config you
run may override. [`configs/examples/reference.yaml`](../configs/examples/reference.yaml) is
the annotated master listing every key, default, and accepted value.

## 🔍 How the config is found

A config file is the single source of truth. It is located in this order:

1. An explicit `--config_path <path>` (on `paperlens-serve` and `paperlens-ingest`).
2. The `PAPERLENS_CONFIG` environment variable.
3. An upward search for a file literally named `config.yaml` from the current directory.

Configs live under [`configs/`](../configs/); the default run config is
`configs/recent-oss-agentic-models.yaml` (there is no `config.yaml` at the repo root, so a
bare command with no `--config_path`/`PAPERLENS_CONFIG` falls back to the code defaults). The
`make` targets pass it for you and take `CONFIG=` to override.

The **project root** — the nearest `pyproject.toml` ancestor of the resolved config file —
is what every relative path in the file is anchored to, so commands work from any working
directory and a config kept in `configs/` still writes data to the repo root (not
`configs/data`). Absolute paths are used as-is.

### 🔧 Interpolation & CLI overrides

Values support OmegaConf `${...}` interpolation — reference another key (`${server.port}`)
or an env var (`${oc.env:HOME}`); resolved at load time. Configs are trusted, user-authored
input, so env interpolation is a convenience, not a new trust boundary.

`paperlens-serve` / `paperlens-ingest` also take **per-field CLI overrides** on top of the
file (`--server.port=9000`, `--llm.chat.model=…`, `--help` to list them all). Overrides are
merged *before* interpolation resolves, so `${...}` sees them too. `paperlens-ingest` also
takes the non-config flag `--retag`.

> **Migration (from the pydantic config):** the LLM selector key `provider` was renamed to
> `type` (uniform with `embedding.type` / `reranker.type`), and the config-file CLI flag
> `--config` became `--config_path`. Both old spellings now fail loudly rather than silently
> defaulting.

Copy-me templates for common setups (local gpt-oss, Anthropic, Gemini, Ollama) live in
[`configs/examples/`](../configs/examples/README.md), alongside the annotated
[`reference.yaml`](../configs/examples/reference.yaml) — copy one into `configs/` and point
`--config_path` / `PAPERLENS_CONFIG` / `make … CONFIG=` at it.

## 📝 `config.yaml` reference

### 📁 `paths`

| Key | Type | Default | Description |
|---|---|---|---|
| `rag_db` | path | `data/rag_db` | Chroma persistent dir + the `papers.json` manifest. |
| `pdf_dir` | path | `data/papers/pdf` | Downloaded PDFs (named `<paper_id>.pdf`). |
| `markdown_dir` | path | `data/papers/text` | Docling-extracted markdown (`<paper_id>.md`). |
| `chat_history` | path | `data/chat_history` | Per-session chat JSON files. |
| `web_dist` | path | `web/dist` | Built frontend SPA served by the backend. |

### 🔝 top level

| Key | Type | Default | Description |
|---|---|---|---|
| `collection` | string | `arxiv_papers` | Chroma collection name for all chunks. |
| `data_path` | string | `data` | Base dir for runtime data; interpolation handle for `paths` (`${data_path}`). |
| `papers` | list | `[]` | The paper list; each entry is `{ name, arxiv_id }`. |

The shipped configs build `paths` from these via interpolation, e.g.
`rag_db: ${data_path}/${collection}/rag_db`, so each collection's data stays isolated and
overriding `--collection` / `--data_path` moves every path at once.

`collection` is the most coupled field in the config — it does **triple duty**: it names the
Chroma collection, it namespaces the data paths (via the interpolation above), and — combined
with the embedder's `name()` — it forms the on-disk index identity. Changing it silently
re-points all three, i.e. it starts a fresh, empty dataset rather than renaming the existing
one. (Changing the embedder alone likewise invalidates the index; see the embedder note.)

`papers` entries:

| Field | Type | Description |
|---|---|---|
| `name` | string | Human name **and** the `paper_id` (filename stem, manifest key, search filter). |
| `arxiv_id` | string | arXiv id used to download the PDF. Quote it (e.g. `"2412.19437"`). |

### 🧬 `embedding`

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | `hf` | Backend variant: `hf` · `openai` · `gemini` · `ollama`. Selects which keys below apply. |
| `model` | string | `BAAI/bge-m3` | HF model id, or the API model name for API types (common to all). |
| `batch_size` | int | `32` | Embedding batch size (common to all). |
| `max_seq_length` | int | `1024` | **`hf` only.** Token cap guarding the MPS `2**32` per-tensor limit on Apple Silicon. |
| `api_base` | string | `""` | **`openai`/`ollama` only.** Base URL (`""` → provider default). |
| `api_key_env` | string | per type | **`openai` (`OPENAI_API_KEY`) / `gemini` (`GEMINI_API_KEY`) only.** Env var holding the key. |

### 🎯 `reranker`

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | `hf` | `hf` (local cross-encoder) or `llm` (reuses the chat model, no extra deps). |
| `model` | string | `BAAI/bge-reranker-v2-m3` | Cross-encoder model id (`hf` type). |
| `enabled` | bool | `true` | Turn the whole rerank stage on/off. |
| `max_length` | int | `512` | **`hf` only.** Cross-encoder input token cap — raise it if `chunking.max_tokens` grows, or passages get silently truncated before scoring. |
| `max_chars` | int | `600` | **`llm` only.** Per-passage excerpt length sent to the judge LLM. |

### 🔀 `sparse`

Opt-in hybrid dense+BM25 retrieval: a lexical (BM25) search runs alongside dense recall and
the two rankings are fused via reciprocal rank fusion (RRF) before reranking. Unproven until
screened — `enabled` defaults to `false`; use `paperlens-eval screen --tier retrieval --hybrid`
to measure it on your pool before flipping it on (see [harness](harness.md)).

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | `bm25` | Sparse backend variant (only `bm25` today). |
| `enabled` | bool | `false` | Turn hybrid fusion on/off. Opt-in, unlike `reranker.enabled` — this needs a harness screen first. |
| `rrf_k` | int | `60` | Reciprocal-rank-fusion constant (`score = Σ 1/(k + rank)`); shared across sparse variants. |
| `fetch_multiplier` | int | `3` | Each system (dense, BM25) over-fetches `fetch_multiplier × retrieval.candidates` before fusing, then truncates to `candidates` — gives RRF margin to promote a hit either side ranked outside its own top-`candidates`. Must be `>= 1`. |
| `k1` | float | `1.5` | **`bm25` only.** BM25 term-frequency saturation constant (`rank_bm25` default). |
| `b` | float | `0.75` | **`bm25` only.** BM25 length-normalization constant (`rank_bm25` default). |

### ✅ `faithfulness`

Opt-in post-generation check: verifies each `[rN]`-cited sentence of the agent's answer
against the passage it cites, using a local consistency-scoring cross-encoder (not a generic
NLI model — see [architecture](architecture.md) for why). Passive labeling only — a citation
that fails the check gets a `faithfulness` verdict attached; nothing about the answer text or
delivery changes. `enabled` defaults to `false`, like `sparse.enabled`.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | `hf` | Faithfulness backend variant (only `hf` today). |
| `enabled` | bool | `false` | Turn the post-generation check on/off. |
| `model` | string | `vectara/hallucination_evaluation_model` | Local consistency-scoring cross-encoder id. |
| `revision` | string | `hhem-1.0-open` | Pinned revision — the current default revision needs `trust_remote_code` and doesn't load against this repo's `transformers` version; this one is a plain checkpoint, no remote code. |
| `max_length` | int | `512` | Cross-encoder input token cap. Scoring is sentence-vs-sentence, so this is rarely a binding constraint. |
| `contradiction_max` | float | `0.05` | Score at/below this → `"contradiction"`. |
| `entailment_min` | float | `0.3` | Score at/above this → `"entailment"`; between the two → `"neutral"`. Both thresholds were calibrated on a small hand-labeled sample (see [architecture](architecture.md)) — a starting point, not a universal constant. |

Citations gain an optional `faithfulness: [{sentence, label, score}, ...]` field (one entry per
citing sentence) when enabled.

### 🤖 `llm`

Two LLM specs share one schema. `llm.tagging` labels papers at ingestion (cheap/fast is
fine); `llm.chat` powers the agent and **must support tool/function calling**.

Each spec is a tagged union on `type` (the `LLMSpec` variants):

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | `anthropic` | `anthropic` · `openai` · `vllm` · `sglang` · `gemini`. `vllm`/`sglang`/`openai` all speak the OpenAI wire format. |
| `model` | string | `claude-opus-4-8` | Model id the provider/endpoint serves. |
| `api_base` | string | `""` | **`openai`/`vllm`/`sglang` only.** Endpoint URL (`""` → OpenAI). Not a valid key for `anthropic`/`gemini`. |
| `api_key_env` | string | `ANTHROPIC_API_KEY` | Env var holding the API key (`OPENAI_API_KEY`/`GEMINI_API_KEY` per type). Local servers ignore it. |
| `max_tokens` | int | `2048` | Max output tokens. |
| `temperature` | float | `0.0` | Sampling temperature. |
| `timeout` | float | `0.0` | Per-request timeout in seconds. `0` → the provider SDK's own default (Anthropic/OpenAI clients default to several minutes). |
| `max_retries` | int | `-1` | Retries *after* the first attempt, on a failed request. `-1` → the provider SDK's own default; `0` → never retry. |

`type` selects the variant; only that variant's keys are valid — an unknown key (e.g. the
old `provider`, or `api_base` on `anthropic`) or an unknown `type` **fails loudly at load**.

Defaults differ by spec: `llm.tagging` defaults to model `claude-haiku-4-5-20251001`;
`llm.chat` defaults to `claude-opus-4-8`. The **shipped `config.yaml`** overrides both to a
local OpenAI-compatible server.

### ✂️ `chunking`

Corpus-dependent: the defaults are tuned for dense ML technical reports and may not fit a
differently-shaped paper list (surveys, shorter position papers, non-ML PDFs, ...).

| Key | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | int | `512` | Target max tokens per chunk. |
| `overlap_tokens` | int | `64` | Overlap between sub-chunks when a section is split. Must be `< max_tokens`. |
| `min_tokens` | int | `24` | Sections shorter than this are dropped (bare titles, author lists, stray captions). |
| `noise_ratio` | float | `0.4` | Fraction of numeric/punctuation tokens in an unnumbered section body that flags it as figure/plot noise. |
| `extra_skip_titles` | list[string] | `[]` | Extra case-insensitive regexes for section titles to always drop, appended to the built-in list (references, TOCs, acknowledgements, ...). |

Changing these doesn't retroactively rechunk already-ingested papers — like the embedder,
it only affects papers indexed after the change (delete `paths.rag_db` and re-ingest to
apply retroactively).

Incoherent combinations (`overlap_tokens >= max_tokens`, `noise_ratio` outside `[0, 1]`,
`retrieval.k > retrieval.candidates`, `tagger.min_tags > tagger.max_tags`) **fail loudly at
load** rather than silently degrading the index.

### 📄 `extraction`

| Key | Type | Default | Description |
|---|---|---|---|
| `ocr_enabled` | bool | `false` | Turn on for scanned/no-text-layer PDFs. arXiv PDFs have a real text layer, so this is off by default — enabling it also triggers a Docling OCR model download. |

### 🔍 `retrieval`

Shapes the agentic search loop. The chat model can still request a different `top_k` per
`search_papers` call.

| Key | Type | Default | Description |
|---|---|---|---|
| `k` | int | `5` | Passages returned per `search_papers` call when the model doesn't specify `top_k`. Must be `<= candidates`. |
| `candidates` | int | `20` | Dense-recall pool size handed to the reranker. This is a **floor**: a larger `top_k` scales the pool to `4 × top_k`, so the reranker always has candidates to discard rather than merely reordering what dense recall returned. |
| `max_rounds` | int | `8` | How many search/answer ReAct cycles the agent gets before it must answer. |

### 🏷️ `tagger`

Shapes the prompt `generate_tags` sends to `llm.tagging` — the model itself is configured
separately under `llm.tagging`.

| Key | Type | Default | Description |
|---|---|---|---|
| `max_tags` | int | `12` | Max tags requested per paper. |
| `min_tags` | int | `5` | Min tags requested per paper. |
| `max_excerpt_chars` | int | `6000` | How much of the paper (title/abstract/section headings) to tag from. |

### 📥 `ingestion`

| Key | Type | Default | Description |
|---|---|---|---|
| `auto_start` | bool | `true` | Start the ingestion worker when the app launches. |

### 🌐 `server`

| Key | Type | Default | Description |
|---|---|---|---|
| `host` | string | `127.0.0.1` | Bind host. |
| `port` | int | `8000` | Bind port. |

## 🔑 Environment variables

Loaded from a local `.env` (via `python-dotenv`) or the shell.

| Variable | Purpose |
|---|---|
| `PAPERLENS_CONFIG` | Path to `config.yaml` (overrides the upward search). |
| `ANTHROPIC_API_KEY` | Default key env for the `anthropic` provider. |
| `GEMINI_API_KEY` | Default key env for the `gemini` provider/embedder. |
| `OPENAI_API_KEY` | Default key env for the `openai` embedder. |
| *(custom)* | Whatever you set a spec's `api_key_env` to (e.g. `LOCAL_LLM_KEY`). |

## 💻 Commands

Console scripts (from `pyproject.toml`); each also runs as `python -m <module>`.

| Command | Equivalent | What it does |
|---|---|---|
| `uv run paperlens-serve` | `python -m server` | Serve the API on `server.host:server.port`; auto-starts the worker if `ingestion.auto_start`. |
| `uv run paperlens-ingest` | `python -m rag.ingest` | Ingest every configured paper not yet in the DB (headless, same pipeline as the worker). |
| `uv run paperlens-ingest --retag` | — | Regenerate tags for already-ingested papers (no re-index). |
| `uv run paperlens-{serve,ingest} --config_path <path>` | — | Use a specific config file. |

### 🎁 Make targets

| Target | Runs |
|---|---|
| `make install` | `uv sync` + `npm --prefix web install`. |
| `make serve` | `uv run paperlens-serve --config_path $(CONFIG)`. |
| `make dev` | Backend + frontend dev server together. |
| `make ingest` | `uv run paperlens-ingest --config_path $(CONFIG)`. |
| `make build` | Production frontend build → `web/dist`. |

`serve`/`dev`/`ingest` default `CONFIG` to `configs/recent-oss-agentic-models.yaml`;
override per run, e.g. `make serve CONFIG=configs/examples/anthropic.yaml`.

### 🖼️ Frontend

| Command | What it does |
|---|---|
| `npm --prefix web run dev` | Dev server on `:5173`, proxies `/api` → backend. |
| `npm --prefix web run build` | Type-check + build to `web/dist` (served by the backend). |

## 🔌 HTTP API

Served under `/api`; any other path falls through to the SPA.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/papers` | List ingested papers with tags. |
| GET | `/api/papers/{paper_id}` | One paper's full markdown. |
| GET | `/api/tags` | Tags with paper counts, excluding any tag present on every paper (no filtering value). |
| GET | `/api/admin/status` | Paper/chunk counts, pending papers, ingestion progress. |
| POST | `/api/admin/rescan` | Re-scan `config.yaml` and ingest new papers. |
| GET | `/api/chats` | List chat sessions. |
| POST | `/api/chats` | Create a chat session. |
| GET | `/api/chats/{chat_id}` | Fetch one chat session. |
| DELETE | `/api/chats/{chat_id}` | Delete a chat session. |
| POST | `/api/chat` | Run the agent; streams the answer + trace over SSE. |
