# Features

A working inventory of what PaperLens does today, organized into four sections:
**App / Product**, **ML / RAG**, **Backend / Infra**, and **DevX**.

## App / Product

### Chat — conversation UX

- **Ask a question (composer)** — textarea with Enter-to-send, Shift+Enter newline, busy/disabled states (`web/src/pages/ChatPage.tsx`)
- **Streaming answer tokens** — assistant text renders incrementally as SSE `token` events arrive, "Thinking…" loader before first token (`web/src/pages/ChatPage.tsx`, `web/src/api.ts`)
- **Edit a previous user message and resend** — inline editable textarea on any user turn, truncates and replays the conversation from that point (`web/src/pages/ChatPage.tsx`)
- **Confirm-before-discard on edit** — warns how many later exchanges will be dropped before an edit truncates them (`web/src/pages/ChatPage.tsx`)
- **New chat** — clears turns/filters, starts a fresh session (`web/src/components/ChatSidebar.tsx`, `web/src/pages/ChatPage.tsx`)
- **Chat history sidebar** — collapsible rail listing all saved chats with active highlight (`web/src/components/ChatSidebar.tsx`)
- **Delete a conversation** — per-row delete, redirects home if it was the active chat (`web/src/components/ChatSidebar.tsx`, `web/src/pages/ChatPage.tsx`)
- **Resume/reload a past conversation** — restores full turn history including citations, trace, feedback, usage (`web/src/pages/ChatPage.tsx`, `web/src/api.ts`)
- **Auto-scroll to latest turn** (`web/src/pages/ChatPage.tsx`)
- **Empty-state hero** — friendly prompt shown for a chat with no turns yet (`web/src/pages/ChatPage.tsx`)
- **Empty-library warning banner** — shown when no papers are indexed, links to Admin (`web/src/pages/ChatPage.tsx`)
- **409 "turn in progress" handling** — friendly error if another turn is already streaming for that chat (`web/src/api.ts`)
- **In-answer error surfacing** — stream `error` events appended inline into the assistant bubble (`web/src/pages/ChatPage.tsx`)

### Chat — filters/scoping

- **Restrict search by paper** — multi-select scoping retrieval, locked once a conversation has started (`web/src/pages/ChatPage.tsx`)
- **Restrict search by tag** — multi-select scoping retrieval, likewise locked mid-conversation (`web/src/pages/ChatPage.tsx`)
- **"Filters locked" tooltip** — explains why filters are disabled mid-conversation and that New Chat changes them (`web/src/pages/ChatPage.tsx`)

### Chat — trace / observability

- **Thought → Action → Observation trace timeline** — collapsible rail rendering the agent's reasoning steps, search-query actions with paper badge, observations (`web/src/components/TraceBox.tsx`)
- **Auto-expand trace while streaming, collapsible after** (`web/src/components/TraceBox.tsx`)
- **Search-count badge** — "N searches" in the trace header (`web/src/components/TraceBox.tsx`)
- **Token usage / latency display** — per-turn footer with total tokens and response latency (`web/src/pages/ChatPage.tsx`, `web/src/api.ts`)

### Chat — citations & faithfulness

- **Clickable inline citation badges** — `[rN]` markers, click navigates to the source paper with the passage highlighted (`web/src/components/Answer.tsx`)
- **Citation hover tooltip** — paper title, section, snippet preview, "click to open" hint (`web/src/components/Answer.tsx`)
- **Faithfulness flag on citations** — colored marker + tooltip on any non-"entailment" citation (`web/src/components/Answer.tsx`, `web/src/faithfulness.ts`)
- **Source cards ("Sources" section)** — cards grouped by paper, citation numbers, faithfulness flags, retrieval-source tooltip (keyword vs semantic match) (`web/src/components/SourceCards.tsx`)
- **Turn-level faithfulness summary badge** — "N/M not clearly supported / may contradict source" (`web/src/components/SourceCards.tsx`, `web/src/faithfulness.ts`)
- **Markdown rendering with GFM/LaTeX/heading anchors** (`web/src/components/Markdown.tsx`)

### Chat — feedback

- **Thumbs up/down per answer** — toggleable vote, persisted via API (`web/src/components/FeedbackControl.tsx`)
- **Optional feedback note** — free-text note attached to a vote, commits on blur (`web/src/components/FeedbackControl.tsx`)
- **Per-chat-scoped feedback state** — keyed by `chatId-index` so switching chats doesn't leak stale state (`web/src/pages/ChatPage.tsx`)

### Chat — export/sharing

- **Copy answer as Markdown** — `[rN]` rewritten to `[^N]` footnotes plus a generated `## References` block (`web/src/components/AnswerActions.tsx`, `web/src/exportAnswer.ts`)
- **Copy as BibTeX** — one `@misc` entry per distinct cited paper (`web/src/components/AnswerActions.tsx`, `web/src/exportAnswer.ts`)
- **Copy-confirmation UI** — icon swaps to a checkmark briefly after a successful copy (`web/src/components/AnswerActions.tsx`)

### Paper library (Papers page)

- **Paper grid/library browser** — responsive card grid with title, id, chunk count, tag badges (`web/src/pages/PapersPage.tsx`)
- **Open a paper** — card links into the paper viewer (`web/src/pages/PapersPage.tsx`)
- **Remove a paper (with confirmation)** — deletes index/cache/config entry, refreshes the grid (`web/src/pages/PapersPage.tsx`, `web/src/api.ts`)
- **Empty-library state** — points to Admin when no papers exist yet (`web/src/pages/PapersPage.tsx`)

### Paper viewer

- **Rendered paper reading view** — full paper markdown with title, arXiv link, tags (`web/src/pages/PaperViewer.tsx`)
- **Link out to arXiv abstract page** (`web/src/pages/PaperViewer.tsx`)
- **Citation deep-link highlighting** — navigating from a citation/source card highlights and scrolls to the cited passage (`web/src/pages/PaperViewer.tsx`, `web/src/highlight.ts`)
- **Text-selection annotation toolbar** — floating "Highlight / Add note" toolbar on selecting ≥20 chars (`web/src/pages/PaperViewer.tsx`, `web/src/components/AnnotationPopover.tsx`)
- **Highlight a passage (no note)** (`web/src/pages/PaperViewer.tsx`)
- **Add a note to a passage** — section-scoped for re-anchoring (`web/src/pages/PaperViewer.tsx`, `web/src/components/AnnotationPopover.tsx`)
- **View/edit/delete an existing annotation** (`web/src/pages/PaperViewer.tsx`, `web/src/components/AnnotationPopover.tsx`)
- **Persistent multi-annotation highlighting** — all saved annotations render as persistent highlights on load (`web/src/highlight.ts`)
- **Annotation re-anchoring failure state** — flags "Not found in current text" if a paper's text changed since annotation (`web/src/pages/PaperViewer.tsx`)
- **Notes rail (drawer)** — side drawer listing all annotations with badge/count indicator (`web/src/pages/PaperViewer.tsx`)
- **Jump to annotation** — smooth-scrolls to the highlighted passage (`web/src/pages/PaperViewer.tsx`)
- **First-time annotation hint** — floating tip until the paper has at least one annotation (`web/src/pages/PaperViewer.tsx`)
- **Loading state** (`web/src/pages/PaperViewer.tsx`)

### Admin / ingestion management

- **Add paper by arXiv id/URL** — inline error display on failure, e.g. "already curated" (`web/src/pages/AdminPage.tsx`, `web/src/api.ts`)
- **Re-scan config** — triggers a re-scan/re-sync of the papers config (`web/src/pages/AdminPage.tsx`, `web/src/api.ts`)
- **Live ingestion progress** — polls every 1.5s, current paper/stage, done/total counts, animated progress bar (`web/src/pages/AdminPage.tsx`)
- **Ingestion state badge** — idle/running/error (`web/src/pages/AdminPage.tsx`)
- **Pending papers list** (`web/src/pages/AdminPage.tsx`)
- **Ingestion error list** (`web/src/pages/AdminPage.tsx`)
- **Library stat tiles** — paper count, chunk count, pending count (`web/src/pages/AdminPage.tsx`)
- **Tag directory with counts** (`web/src/pages/AdminPage.tsx`)

### Global navigation / theming / app shell

- **Top nav (Chat / Papers / Admin)** (`web/src/App.tsx`)
- **Light/dark mode toggle** — starts OS-following, persists once explicitly toggled (`web/src/App.tsx`)
- **Multiple built-in color palettes** — Rosé Pine default, Solarized/Catppuccin/Gruvbox alternatives (`web/src/theme.ts`, `web/src/palettes.ts`)
- **Frosted-glass sticky header** (`web/src/App.tsx`)
- **Responsive layout** — breakpoint-aware grid and content width (`web/src/App.tsx`, `web/src/pages/PapersPage.tsx`)
- **Reduced-motion accessibility support** — respects `prefers-reduced-motion` (`web/src/styles.css`)
- **Custom scrollbar styling** (`web/src/styles.css`)
- **Global error boundary** — catches render/effect crashes app-wide, shows error + Reload button (`web/src/components/ErrorBoundary.tsx`)
- **Custom icon set** — hand-rolled SVG icons used throughout the UI (`web/src/components/Icons.tsx`)

## ML / RAG

### Retrieval

- **Two-stage retrieval (dense recall → rerank)** — embeds the query, pulls `retrieval.candidates` nearest chunks from Chroma, reranks to `retrieval.k` (`src/rag/search.py`)
- **Candidate-pool scaling** — recall pool scales to `max(candidates, top_k * 4)` when the model requests a larger `top_k` (`src/server/agent.py`, `src/rag/config.py`)
- **Graceful degrade on reranker failure** — falls back to pre-rerank order rather than failing the request (`src/rag/search.py`)
- **Asymmetric embedding query/document split** — `embed_query()` diverges from `__call__()` for embedders that treat queries and documents differently (`src/rag/embedders.py`)
- **Hybrid dense+BM25 retrieval (opt-in)** — lexical search alongside dense recall, fused via reciprocal rank fusion before reranking (`src/rag/search.py`, `src/rag/sparse.py`, `src/rag/config.py`)
- **Stale-snapshot-safe BM25** — lazily rebuilds the BM25 index whenever the collection has grown since the last snapshot (`src/rag/search.py`)
- **Result provenance tagging** — each fused result records whether it came from dense/sparse/both (`src/rag/search.py`)
- **Multi-query expansion (opt-in)** — paraphrases the query into `n_paraphrases` variants, RRF-fuses all rankings in one pass (`src/rag/search.py`, `src/rag/query_expansion.py`, `src/rag/config.py`)
- **Paper/tag-scoped search filter** — intersects a tag filter and explicit paper picker into Chroma's `where` clause (`src/rag/search.py`, `src/rag/manifest.py`)
- **Section-aware chunking with breadcrumbs** (upstream of retrieval) — breadcrumb + body is what dense recall matches against (`src/rag/chunking.py`)

### Reranking

- **Pluggable reranker ABC** — uniform `score(query, docs) -> list[float]` interface (`src/rag/reranker.py`)
- **`hf` cross-encoder reranker** — local sentence-transformers cross-encoder, default `BAAI/bge-reranker-v2-m3`, lazy-loaded (`src/rag/reranker.py`)
- **`llm` reranker** — reuses the chat LLM to pointwise-rate candidates 0–10 in one batched call, degrades to zero scores on malformed output (`src/rag/reranker.py`)
- **`build_reranker` registry dispatch** (`src/rag/reranker.py`)
- **`reranker.enabled` kill switch** — turns off the whole rerank stage (`src/rag/config.py`)

### Agent loop

- **`ChatAgent` — ReAct loop over native tool calling** — Thought → Action (`search_papers`) → Observation, repeats until it answers; small talk skips search entirely (`src/server/agent.py`)
- **Single tool: `search_papers`** — `query` (required), optional `paper`, optional `top_k`; decomposes multi-part questions into several focused calls (`src/server/agent.py`)
- **Filter-scoped system prompt** — injected paper catalog restricted to the active tag/paper filter so the model can't bypass it via the prompt prefix (`src/server/agent.py`)
- **Ref/citation registry** — each returned passage gets a `ref` (`r1`, `r2`, ...) the model must cite; numbering continues across turns in the same chat (`src/server/agent.py`)
- **Provider-agnostic tool-use loop** — one neutral tool schema drives every LLM backend's own wire format (`src/rag/llm.py`)
- **Streaming reasoning/trace surfacing** — `on_text`/`on_reasoning` callbacks stream tokens and reasoning (including `<think>` tag stripping) for the live trace (`src/rag/llm.py`, `src/server/agent.py`)
- **`retrieval.max_rounds`** — caps ReAct search/answer cycles before the agent must answer (`src/rag/config.py`)

### Faithfulness

- **Post-generation faithfulness check (opt-in)** — verifies each `[rN]`-cited sentence against the passage it cites; attaches a verdict as a signal, never gates or alters the answer (`src/rag/faithfulness.py`, `src/server/agent.py`)
- **`hf` faithfulness backend** — local consistency-scoring cross-encoder (`vectara/hallucination_evaluation_model`, pinned revision) (`src/rag/faithfulness.py`, `src/rag/config.py`)
- **Sentence-vs-sentence scoring (SummaC-style)** — scores the citing sentence against every sentence of the cited passage, keeps the strongest-supporting pair (`src/rag/faithfulness.py`)
- **Threshold-derived labels** — `contradiction_max`/`entailment_min` turn the raw score into entailment/neutral/contradiction (`src/rag/faithfulness.py`, `src/rag/config.py`)
- **Faithfulness threshold calibration** — standalone script checks thresholds against a hand-labeled golden set, corpus-independent (`scripts/calibrate_faithfulness.py`, `tests/data/faithfulness_pairs.jsonl`)
- **Model warmup for faithfulness** — dummy check at server startup preloads the cross-encoder (`src/server/main.py`)

### Eval / calibration harness (`paperlens-eval`)

- **`gen`** — builds the span-anchored QA eval set from the loaded, ingested pool, idempotent on corpus fingerprint; optional closed-book leakage pre-check (`src/eval/cli.py`, `src/eval/queryset.py`, `src/eval/genfilter.py`, `src/eval/fingerprint.py`)
- **`run`** — scores the current config on the dev split, no re-index (`src/eval/cli.py`, `src/eval/harness.py`)
- **`screen --tier retrieval`** — one-factor-at-a-time screen of `reranker.enabled`/`retrieval.candidates` (plus optional hybrid/multi-query arms), paired vs. default with confidence intervals (`src/eval/optimizer.py`)
- **`screen --tier chunking`** — OFAT screen over `chunking.*` knobs, each arm re-indexed into its own isolated collection (`src/eval/optimizer.py`)
- **`sweep`** — staged grid over `chunking.max_tokens × retrieval.candidates × reranker.enabled` (`src/eval/optimizer.py`)
- **`confirm`** — scores one human-chosen config once on the held-out test split, emits a paste-ready `config.yaml` block (`src/eval/cli.py`)
- **Guard: chunking-sweep index isolation** — every chunking arm re-indexes into its own throwaway collection, asserts chunk counts to catch id collisions (`src/eval/index_isolated.py`)
- **Guard: config-independent gold spans** — gold is a character span over sections, never a chunk id, so no arm can recognize its own handwriting (`src/eval/queryset.py`)
- **Two-stage metrics** — Stage 1 `success@candidates`, Stage 2 `MRR@k` conditioned on stage-1 success (`src/eval/metrics.py`)
- **Resolution reporting (paper-clustered bootstrap)** — ceiling, minimum detectable difference, cluster count; warnings below cluster minimum and near-saturated ceilings (`src/eval/stats.py`)
- **Resumable checkpointing** — `run`/`screen`/`sweep`/`confirm` checkpoint per-item/per-cell, resuming a killed run; invalidated by header/index mismatch, `--fresh` discards (`src/eval/checkpoint.py`)
- **Corpus fingerprinting** — SHA-256 over sorted paper ids + markdown content keys every eval-set file, so a changed pool is detected and regenerated (`src/eval/fingerprint.py`)
- **Known-limits self-documentation** — estimand gap, section-localization-not-answer-localization, no contamination audit, documented inline and in `docs/harness.md`

## Backend / Infra

### Ingestion pipeline

- **`ingest_paper` — six-stage per-paper pipeline** — download → extract (Docling) → chunk → index (Chroma) → tag (LLM) → manifest write; index and tag stages run concurrently (`src/rag/pipeline.py`)
- **PDF download** — arXiv PDF fetch by id, cached (skips if already downloaded) (`src/rag/pipeline.py`)
- **PDF → markdown extraction (Docling)** — single converter instance reused across calls, OCR off by default, cached on disk (`src/rag/extract.py`)
- **Section-aware chunking with breadcrumbs** — rebuilds section-numbering hierarchy into a breadcrumb prepended to embedded text; drops noise sections; packs oversized sections into overlapping sub-chunks; keeps tables atomic (`src/rag/chunking.py`)
- **Chroma indexing** — content-hashing chunk ids so re-ingest under a changed chunking config also deletes stale orphaned chunks (`src/rag/index.py`)
- **LLM tagging** — generates topic tags per paper, degrades to `[]` on failure rather than failing ingestion (`src/rag/tagger.py`)
- **Tag normalization pass** — post-ingestion pass merging near-duplicate tags across the whole library (`src/rag/pipeline.py`, `src/rag/tagger.py`)
- **Pending-paper diffing** — diffs `config.yaml`'s `papers:` list against the manifest to find what's left to ingest (`src/rag/pipeline.py`)
- **Headless ingestion CLI** — `paperlens-ingest`, with `--retag` (regenerate tags only) and `--reindex` (re-chunk/re-embed, tags carried forward) (`src/rag/ingest.py`)
- **Manifest (paper-level metadata store)** — `papers.json`, no in-memory cache, lock-serialized writes, tag counts and filtering (`src/rag/manifest.py`)
- **Comment-preserving config writer** — `ruamel.yaml` read-modify-write over `config.yaml`'s `papers:` list, preserves comments/formatting, write-temp-then-rename, thread-locked (`src/rag/config_writer.py`)

### Serving infra

- **FastAPI app composition** — `create_app(cfg)` wires manifest, chat store, annotation store, ingestion worker, Chroma collection, all HTTP routes; SPA served as catch-all fallback (`src/server/main.py`)
- **In-process ingestion worker** — background thread over pending papers, re-scans for newly-added papers within the same run, single-flight via `trigger()` (`src/server/worker.py`)
- **Lazy, singleton `ChatAgent` build with lock** — embedder/reranker/LLM stack builds once on first use, guarded against double-build by the startup warmer (`src/server/main.py`)
- **Admin add/remove-paper routes** — add normalizes an arXiv id/URL, writes config, triggers ingestion; remove deletes config entry, Chroma chunks, manifest record, annotations, cached PDF/markdown (`src/server/main.py`)
- **Chat session store (file-backed)** — one JSON file per session, parallel messages/citations/traces/feedback/usage arrays, single-flight per-chat guard, write-temp-then-rename (`src/server/chats.py`)
- **Annotation store (file-backed)** — per-paper JSON anchored by text snippet + section slug (not a chunk id/offset), reflow-resilient across re-extraction (`src/server/annotations.py`)
- **Request/response schemas** — pydantic models for chat, feedback, annotations, add-paper requests (`src/server/schemas.py`)
- **HTTP API surface** — papers/tags listing, paper markdown fetch, admin status/rescan/add/remove, chat session CRUD + feedback, streaming `/api/chat` (`src/server/main.py`)

### Startup / model warming

- **Instant startup, background model warmup** — worker starts and a daemon thread eagerly builds the searcher (+ faithfulness checker if enabled) via a dummy search, so routes are servable immediately while the ~20-30s model load happens in the background; warmup failures are non-fatal (`src/server/main.py`)
- **MPS tensor-cap guard** — `embedding.max_seq_length` caps `bge-m3`'s sequence length to avoid overflowing Metal's per-tensor limit on Apple Silicon (`src/rag/embedders.py`)
- **Cloud clients imported lazily** — OpenAI/Gemini/Anthropic SDKs imported inside methods, not at module load (`src/rag/llm.py`, `src/rag/embedders.py`)

### SSE streaming plumbing

- **`/api/chat` SSE endpoint** — agent turn runs on a worker thread, bridges thread-safe callbacks to an asyncio queue, async generator drains it as an `EventSourceResponse` (`src/server/main.py`)
- **Event types emitted** — `token`, `trace`, `citations`, `usage`, `meta`, `error`, `done` (`src/server/main.py`)
- **Latency measurement scope** — timed across retrieval + rerank + LLM + faithfulness check, the wait the user actually feels (`src/server/main.py`)
- **Concurrency guard for streamed turns** — prevents two overlapping `/api/chat` calls on the same chat id from interleaving a truncate-then-append (`src/server/chats.py`, `src/server/main.py`)

### Config system

- **`draccus` dataclass config with `ChoiceRegistry` tagged unions** — `Config` is the single decode target; embedder/reranker/sparse/faithfulness/LLM are tagged-union bases whose `type:` selects a variant (`src/rag/config.py`)
- **OmegaConf `${...}` interpolation** — resolved at load time or after CLI-override merge (`src/rag/config.py`)
- **Per-field CLI overrides** — `--field.subfield=value` flags on top of the config file (`src/rag/config.py`)
- **Config discovery** — `--config_path` → `PAPERLENS_CONFIG` env var → upward search from CWD (`src/rag/config.py`)
- **Project-root anchoring** — every relative path anchored to the nearest `pyproject.toml` ancestor of the resolved config file, not CWD (`src/rag/config.py`)
- **Fail-loud validation** — `__post_init__` checks (e.g. `overlap_tokens < max_tokens`, `k <= candidates`, `min_tags <= max_tags`) so incoherent configs fail at load rather than degrading silently (`src/rag/config.py`)
- **`IngestConfig` — narrow, frozen ingestion projection** — `Config.for_ingest()` aliases sub-objects by reference so a live admin edit is immediately visible to the worker with no reload (`src/rag/config.py`)
- **`.env` loading** — picked up automatically by every entry point (`src/rag/config.py`)
- **Migration guards** — old spellings (`provider` instead of `type`, `--config` instead of `--config_path`) fail loudly rather than silently defaulting (`src/rag/config.py`)

### ChoiceRegistry extensibility pattern

- **Pattern shape** — one `@Base.register_subclass("name")` dataclass + one `match` arm in the corresponding `build_*` function; unknown `type` or stray field fails loudly at decode time, no `if/elif` dispatch (`src/rag/config.py`)
- **Embedder backends** — `hf`, `openai`, `gemini`, `ollama` (`src/rag/embedders.py`, `src/rag/config.py`)
- **Reranker backends** — `hf` cross-encoder, `llm` (`src/rag/reranker.py`, `src/rag/config.py`)
- **LLM backends** — `anthropic`, `openai`/`vllm`/`sglang` (OpenAI-compatible wire format), `gemini` (`src/rag/llm.py`, `src/rag/config.py`)
- **Sparse backend** — `bm25` today, structured for future lexical-backend additions (`src/rag/config.py`)
- **Faithfulness backend** — `hf` today, same pattern (`src/rag/config.py`, `src/rag/faithfulness.py`)
- **Documented add-a-backend workflow** — `docs/how-to.md#add-a-new-llm-backend`, codified as the `add-llm-backend` skill

## DevX

### Quality gate

- **4-command Python gate** — `ruff format --check`, `ruff check`, `ty check`, `pytest`; identical locally and in CI (`CLAUDE.md`, `.github/workflows/ci.yaml`)
- **4-command web gate** — prettier, eslint, tsc, vitest; identical locally and in CI (`web/package.json`, `.github/workflows/ci.yaml`)
- **Auto-fix commands** — `ruff format`/`ruff check --fix` and `prettier --write`/`eslint --fix`
- **ruff config** — line length 100, double quotes, lint rules `E,F,I,UP,B,SIM` (`pyproject.toml`)
- **Node version pin** — `web/.nvmrc` (22)

### CI pipeline

- **GitHub Actions, two parallel jobs** (`python`, `web`) — `uv sync --all-extras` so `ty` resolves lazy imports and backend tests run; web job builds the frontend after its gate (`.github/workflows/ci.yaml`)

### Pre-commit hooks

- **Fast auto-fixing subset runs on every commit** — `ruff-check`/`ruff-format` (pinned via `uv run`), `codespell`, whitespace/EOF/YAML/TOML/AST/merge-conflict/large-file checks, `eslint --fix`/`prettier --write` scoped to `web/` (`.pre-commit-config.yaml`)
- **`ty` and `pytest`/`typecheck`/`test` deliberately excluded** from pre-commit to keep hooks fast — CI-only

### Test fixtures / offline seams

- **Factory fixtures in `tests/conftest.py`** — `make_config`, `make_searcher`, `seed_chunks`, `fake_embedder` (deterministic bag-of-words hashing embedder), `fake_llm` (scripted `FakeLLM`), `fake_faithfulness_checker` (scripted verdicts) — every test runs fully offline, no network or model downloads
- **`tests/unit/` + `tests/integration/` + `tests/data/` split**, no `__init__.py` (`--import-mode=importlib`)
- **Coverage config** — branch coverage over `rag`, `server`, `eval` (`pyproject.toml`)
- **Optional-extra test skipping** — `pytest.importorskip(...)` for anthropic/gemini when extras aren't installed
- **Web test setup** — jsdom environment, shared setup file (`web/vite.config.ts`, `web/src/test/setup.ts`)

### Config-driven design as a DevX feature

- **Copy-me example configs** — `local-gpt-oss.yaml`, `anthropic.yaml`, `gemini.yaml`, `ollama.yaml` (`configs/examples/`)
- **Fully annotated reference config** — every key, default, and accepted value documented in one file (`configs/examples/reference.yaml`)

### Docs structure

- **Diátaxis-style docs hub** — landing page routes by task type (Tutorial/Reference/How-to/Explanation) (`docs/README.md`)
- **Tutorial, Reference, How-to, Explanation docs** — `docs/getting-started.md`, `docs/configuration.md`, `docs/how-to.md`, `docs/architecture.md`, `docs/harness.md`
- **Project glossary** — one term per concept with code pointer and deliberately-avoided synonyms (`CONTEXT.md`)
- **Contributor guide** — dev setup, the gate, layout, import-graph rule, style, test conventions, docs-update-on-change rule (`CONTRIBUTING.md`)
- **AI-agent-facing project instructions** — restates the gate and layout for Claude Code sessions (`CLAUDE.md`)

### Architecture enforcement as DevX

- **Documented one-way import graph** across `rag` → `server`/`eval`, checked by convention, not tooling (`src/rag/__init__.py`, `CONTRIBUTING.md`)

### Dev scripts

- **`scripts/calibrate_faithfulness.py`** — standalone faithfulness-threshold calibration against a hand-labeled golden set; deliberately outside the pytest gate since it loads a real model

### Makefile targets

- **`make install`**, **`make dev`** (backend + frontend together), **`make serve`**, **`make ingest`**, **`make build`** — all of `serve`/`dev`/`ingest` accept `CONFIG=<path>` (`Makefile`)

### CLI entry points

- **`paperlens-serve`**, **`paperlens-ingest`** (`--retag`, `--reindex`), **`paperlens-eval`** (`gen`/`run`/`screen`/`sweep`/`confirm`) — all take `--config_path` and per-field overrides (`pyproject.toml`)

### Optional dependency extras

- **Lazy LLM backend extras** — `anthropic`, `gemini`, `all` — only installed when needed (`pyproject.toml`)
