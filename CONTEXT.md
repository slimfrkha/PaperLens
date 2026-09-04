# 📖 CONTEXT — PaperLens domain glossary

One term per concept. Use these names in code, comments, docs, commit messages, and
when talking to an AI assistant about this repo. Each entry lists the definition, where
it lives in the code, and an `_Avoid_:` list of synonyms we deliberately do **not** use.

---

### 📄 Paper

One arXiv paper tracked by the app. Declared in `config.yaml` under `papers` as
`{ name, arxiv_id }`. Everything downstream keys off it.

- Code: `Paper` model in `src/rag/config.py`.
- `_Avoid_:` document, article, PDF (a PDF is one *artifact* of a paper, not the paper).
- See [Why arXiv-specific](docs/architecture.md#-why-arxiv-specific--what-wont-generalize)
  for why this app doesn't generalize to arbitrary documents.

### 🆔 paper_id

The stable identifier for a paper, equal to its config `name` (e.g. `deepseek-v3`). Used
as the Chroma metadata key, the manifest key, the markdown/PDF filename stem, and the
`paper` filter in search.

- Code: set to `paper.name` in `ingest_paper` (`src/rag/pipeline.py`).
- `_Avoid_:` slug, key, doc_id. It is **not** the `arxiv_id`.

### ✂️ Chunk

A section-sized unit of a paper that gets embedded and indexed. Its embedded text is a
**breadcrumb** prefix + the section body; the body is stored separately for display.

- Code: `Chunk` dataclass and `chunk_markdown` in `src/rag/chunking.py`.
- `_Avoid_:` passage (reserve "passage" for a retrieved chunk shown to the LLM/reader),
  segment, fragment, node.

### 🧭 Breadcrumb

The reconstructed section path prepended to a chunk, e.g. `2.1.1 Multi-Head Latent
Attention`. Docling flattens every heading to `##`, so we rebuild the hierarchy from the
section *numbering* in the heading text. Prepending it gives the embedding context.

- Code: built in `src/rag/chunking.py`; carried as `Result.breadcrumb`.
- `_Avoid_:` heading path, section trail, hierarchy string.

### 📋 Manifest

The paper-level metadata store, `papers.json`, living inside the RAG DB directory. Holds
one record per ingested paper (`paper_id`, `title`, `arxiv_id`, `tags`, `n_chunks`,
`ingested_at`) and answers "is this paper ingested?" and "which papers have these tags?".

- Code: `Manifest` in `src/rag/manifest.py`; file at `<rag_db>/papers.json`.
- `_Avoid_:` catalog, registry (see **registry** below — a different concept), index
  (the Chroma vector store is the "index").

### 🗄️ Index / RAG DB

The Chroma persistent vector store of chunk embeddings, on disk under `paths.rag_db`. One
Chroma **collection** (`collection: arxiv_papers`) holds every chunk. The manifest lives
in the same directory.

- Code: `open_collection` / `index_markdown` in `src/rag/index.py`.
- `_Avoid_:` vector DB *and* embeddings store used interchangeably — say "the index" or
  "the RAG DB" for the on-disk store, "collection" for the Chroma collection.

### 🧬 Embedder

The component that turns text into vectors. Selected by `embedding.type`
(`hf` · `openai` · `gemini` · `voyage` · `ollama`) and built through the **registry**. The same
embedder is used at indexing time and query time — though asymmetric embedders may embed
the same text differently depending on which side it's called from: Gemini's `task_type`,
or `hf`'s optional `query_prefix`/`document_prefix`, via `embed_query()` diverging from
`__call__()`.

- Code: `Embedder` ABC + `build_embedder` in `src/rag/embedders.py`.
- `_Avoid_:` encoder (that is the reranker's cross-encoder), vectorizer, model.

### 🎯 Reranker

The second retrieval stage: rescores candidate chunks against the query. Selected by
`reranker.type` — `hf` (a local cross-encoder), `llm` (reuses the chat model to score), or
`voyage` (a dedicated rerank API). `reranker.enabled` toggles the whole stage.

- Code: `Reranker` ABC + `build_reranker` in `src/rag/reranker.py`.
- `_Avoid_:` re-ranker (no hyphen), scorer, second-pass.

### 🔀 Hybrid retrieval

Opt-in fusion of dense recall with a **BM25** lexical search before reranking, via **RRF**.
Toggled by `sparse.enabled` (off by default — unproven until screened, see
[harness](docs/harness.md)'s `--hybrid` screen). Catches exact lexical tokens (model IDs,
acronyms) a bi-encoder can smear into semantic space.

- Code: the fusion branch in `Searcher.search` (`src/rag/search.py`); `SparseCfg`/`BM25Cfg`
  in `src/rag/config.py`.
- `_Avoid_:` sparse retrieval alone (that's just the BM25 half); hybrid search (ambiguous —
  say "hybrid retrieval").

### 🔡 BM25

The lexical (term-frequency/IDF) retrieval algorithm that supplies the sparse half of
**hybrid retrieval**. Indexes the same chunk text already stored in Chroma — no separate
storage. A **snapshot** of the corpus at build time (its IDF needs global corpus stats), so
`Searcher.sparse` rebuilds it whenever the collection has grown since the snapshot was taken.

- Code: `BM25Index` / `build_sparse_index` in `src/rag/sparse.py` (wraps `rank_bm25.BM25Okapi`).
- `_Avoid_:` keyword search, full-text search, lexical index (say "BM25").

### 🔗 RRF (reciprocal rank fusion)

The formula that merges the dense and BM25 rankings into one: `score(d) = Σ 1/(k + rank_i(d))`
per ranking `i`, summed over every ranking `d` appears in. `sparse.rrf_k` is the constant `k`
(default 60); `sparse.fetch_multiplier` controls how far each side over-fetches before fusing.

- Code: `reciprocal_rank_fusion` / `rrf_scores` in `src/rag/sparse.py`.
- `_Avoid_:` score fusion, rank merging, ensemble (say "RRF").

### 🌐 Multi-query expansion

Opt-in: `llm.chat` paraphrases the query into `multi_query.n_paraphrases` variants, each
variant is searched, and every resulting ranking — dense, plus BM25 per variant when
**hybrid retrieval** is also on — is **RRF**-fused in one flat pass, not a fusion of
per-variant fusions. Toggled by `multi_query.enabled` (off by default — unproven until
screened, see [harness](docs/harness.md)'s `--multi-query` screen). A recall boost against
how a question happens to be phrased; the cost (one extra LLM call plus
`n_paraphrases` extra retrievals per `search_papers` call) is unconditional, not gated on
retrieval confidence.

- Code: the `len(variants) > 1` branch in `Searcher.search` (`src/rag/search.py`);
  `generate_paraphrases` in `src/rag/query_expansion.py`; `MultiQueryCfg` in
  `src/rag/config.py`.
- `_Avoid_:` query rewriting, query expansion alone (say "multi-query expansion"), paper
  voting (that's a different peer repo's design — PaperLens fuses chunks, not papers).

### 🧩 Per-paper retrieval

Opt-in, toggled per chat message (`ChatRequest.per_paper`, not a config knob): recall runs
once per paper in the resolved `paper_ids` — each paper's own dense/sparse/**multi-query
expansion** fusion, scoped via `where`, capped to a per-paper candidate budget
(`clamp(candidates // n_papers, max_k, candidates)`) — instead of once over the whole scope,
then pools every paper's candidates flat before the shared rerank/**elbow cutoff** step.
Purpose: stop one paper with many relevant chunks from crowding the candidate budget out of
other papers before the reranker ever sees them; the final elbow cutoff stays global, with
no per-paper quota — per-paper retrieval only ever shaped recall, never guaranteed every
paper survives into what's returned. That gap — no guarantee every paper in scope actually
gets asked — is what **Compare mode** closes at the agent level instead of the retrieval
level; the two coexist as a primary/secondary pair (see below), not competing toggles. When
no paper/tag filter is active, `ChatAgent` falls back to every paper in the **manifest** —
`Searcher` itself has no manifest awareness and raises if asked for per-paper recall with
no resolved paper list.

- Code: the `per_paper` branch in `Searcher.search` (`src/rag/search.py`); the manifest
  fallback in `ChatAgent.run` (`src/server/agent.py`); `ChatRequest.per_paper`
  (`src/server/schemas.py`).
- Product-facing: the "Broaden recall per paper" knob, shown (and only sendable) in **Ask**
  mode — hidden under **Compare mode**, where it's structurally meaningless (Compare's
  per-paper sub-runs already search one paper at a time).
- `_Avoid_:` per-source retrieval (collides with `Result.source`'s dense/sparse/both
  meaning), per-document retrieval, paper-level round-robin.

### 🆚 Compare mode

An agent-level answer shape, alongside — not replacing — **per-paper retrieval**: the
primary **Auto**/**Ask**/**Compare** switch in the composer (see **Auto mode** below for the
third option). Ask produces one combined answer over a shared retrieval pool, same as
before. Compare runs the question once per paper as an
independent sub-run (`ChatAgent.compare`, reusing `ChatAgent.run` unmodified, scoped to one
paper at a time), guaranteeing every paper in the resolved scope actually gets asked — a
guarantee per-paper retrieval's recall-only fix doesn't give, since the shared rerank/elbow
cutoff after it stays global. Every per-paper answer then feeds back into the model once
more to synthesize one final, genuinely comparative answer that reuses each sub-run's own
`[rN]` **ref**s (`SYNTHESIS_SYSTEM_PROMPT`). The per-paper legwork becomes an optional,
collapsible carousel drill-down above the final answer, one slide per paper, mirroring how
the **Turn**'s trace already sits above a normal answer.

- Code: `ChatAgent.compare`/`SYNTHESIS_SYSTEM_PROMPT`/`_resolve_paper_ids`
  (`src/server/agent.py`); `ChatRequest.compare`, `"compare"`/`"compare_results"` on a
  stored **Turn** (`src/server/schemas.py`, `src/server/chats.py`); `compare_row` SSE event
  (`src/server/chat_turn.py`); `ComparePanel`/`TraceEntries` (`web/src/components/`).
- **Naming — don't confuse with `paperlens-eval comparative`**: that's an unrelated
  eval-harness command (see [docs/harness.md](docs/harness.md)) measuring whether
  retrieval-level per-paper retrieval helps genuinely cross-paper questions — a different
  layer (harness measurement, not a product feature) that happens to use the adjacent word
  "comparative," not "compare."
- `_Avoid_:` comparative mode (see naming note above), side-by-side mode, multi-paper mode.

### 🤖 Auto mode

The default option on the primary **Ask**/**Compare** switch — a meta-decision on that same
axis, not a third peer answer shape. Before the real turn is sent, `ChatAgent.classify_mode`
asks a small, tool-free LLM completion (`CLASSIFY_SYSTEM_PROMPT`, run on a cheap tagging-tier
client — same idiom as `generate_name`'s session titling, not the full chat model) whether the
question needs **Compare**'s per-paper guarantee or a pooled Ask answer is faithful enough. The
classify prompt includes the same papers catalog `_system` injects into the ReAct system
prompt (`_papers_catalog`, shared by both), so it can tell "what's the model size?" needs
Compare from the scope alone (several distinct model papers) even when the question never
says "each" or "compare" — not from wording alone. Below 2 resolved papers, or on any
classifier failure, it resolves to Ask deterministically —
no LLM call needed in the first case, the always-safe default in the second. Classification is
a separate pre-flight HTTP round trip (`POST /api/chat/classify`) before the real turn, not
folded into `/api/chat`'s SSE stream — SSE can't pause mid-stream for the large-Compare confirm
dialog Auto reuses once it knows the resolved mode + scope size. `auto: bool` is a badge-only
persistence field (`ChatRequest.auto`, stored alongside `compare`/`per_paper`) — it never
drives dispatch, only lets a reloaded turn show an "Auto" badge and restore the control to Auto
rather than to whatever mode was resolved.

- Code: `ChatAgent.classify_mode`/`CLASSIFY_SYSTEM_PROMPT` (`src/server/agent.py`);
  `ClassifyModeRequest`/`ClassifyModeResponse`/`ChatRequest.auto` (`src/server/schemas.py`);
  `/api/chat/classify` route (`src/server/main.py`); `classifyMode`/`resolveSendMode`
  (`web/src/api.ts`, `web/src/pages/ChatPage.tsx`).
- `_Avoid_:` automatic mode, smart mode, AI-picked mode — "Auto" alone is the product-facing
  name, matching the segmented control's label exactly.

### 🪜 Elbow cutoff

The third retrieval stage: how many reranked passages a `search_papers` call actually
returns. Not a fixed count — `find_cutoff` looks for the first real drop-off in score (a
MAD-based robust outlier test plus a prominence floor, both self-normalizing per query so
it works whether the reranker's scores are unbounded cross-encoder logits or an `llm`
reranker's 0–10 ratings) and cuts there, bounded to `[retrieval.min_k, retrieval.max_k]`.
Only runs when reranking actually happened — a skipped or failed rerank falls back to plain
`max_k` truncation, same as `retrieval.elbow_enabled: false`.

- Code: `find_cutoff` in `src/rag/search.py`; `SearchOutcome.cutoff_reason` records why
  that many came back (`"elbow"` / `"no_elbow"` / `"pool_exhausted"` / `"no_rerank"` /
  `"disabled"`).
- `_Avoid_:` knee detection, threshold cutoff (say "elbow cutoff") — it's not a fixed score
  threshold; see [harness](docs/harness.md) for why an absolute threshold was rejected.

### 🔎 Searcher

The object that runs three-stage retrieval: dense **embedder** recall of `candidates`, then
optional **reranker** rescoring, then an **elbow cutoff**, returning between `min_k` and
`max_k` **Result**s. Accepts a `paper` / `paper_ids` filter.

- Code: `Searcher` in `src/rag/search.py`.
- `_Avoid_:` retriever, search engine, query engine.

### 📑 Result / passage

One retrieved chunk returned by the Searcher: `score`, `paper_id`, `breadcrumb`,
`section_title`, `text`, `body`, `source`. When a Result is handed to the chat model or shown to
the reader, call it a **passage**. `source` (`"dense"`/`"sparse"`/`"both"`) records which
retrieval pool(s) surfaced it — meaningful only when **hybrid retrieval** is on; otherwise
always `"dense"`. `Searcher.search` doesn't return a bare list of these — it returns a
`SearchOutcome` (`results` + `cutoff_reason`, see **Elbow cutoff**).

- Code: `Result`/`SearchOutcome` dataclasses in `src/rag/search.py`.
- `_Avoid_:` hit, match, document.

### 📎 ref / citation

The `[rN]` marker (`r1`, `r2`, …) the agent assigns to each retrieved passage and threads
into its answer. The frontend turns each `ref` into a clickable **citation** that opens
the paper at the cited passage.

- Code: `ref` registry built in `ChatAgent.run` (`src/server/agent.py`).
- `_Avoid_:` reference number, footnote, source id. The marker is a `ref`; the rendered
  clickable thing is a `citation`.

### 🖍️ Annotation

A user-saved passage in the Paper Viewer, with an optional personal note attached. Anchored
to the passage's **text** (a `snippet`, scoped to its section via `section_slug`), not a
chunk id or character offset — the same reflow-resilient anchoring `ref`/citation jumps use
— so it survives a re-extraction that changes chunk boundaries. Rendered as its own
persistent, multi-range `"annotation"` highlight group, distinct from the single-range,
transient `"citation"` highlight group used for a citation jump.

- Code: `AnnotationStore` (`src/server/annotations.py`); anchor resolution + the
  `"annotation"` highlight group in `web/src/highlight.ts`; UI in
  `web/src/components/AnnotationPopover.tsx` and `web/src/pages/PaperViewer.tsx`.
- `_Avoid_:` bookmark, note (alone — a note is what's *attached* to an annotation, not the
  annotation itself), highlight (alone — ambiguous with the citation highlight group).

### ✅ Faithfulness check / verdict

An opt-in post-generation check (`faithfulness.enabled`, off by default) verifying each
`[rN]`-cited sentence of the agent's answer is supported by the passage it cites, scored
sentence-vs-sentence with a local consistency-scoring cross-encoder (not the whole passage
as one premise). Attaches a `Verdict` (`label`: `entailment`/`neutral`/`contradiction`,
`score`) to the citation — a signal, not a gate; the answer itself never changes.

- Code: `FaithfulnessChecker` / `Verdict` / `attribute_refs` in `src/rag/faithfulness.py`;
  wired into `ChatAgent.run` (`src/server/agent.py`).
- `_Avoid_:` hallucination detector, grounding score (say "faithfulness check"/"verdict").

### 🏷️ Tag

An LLM-generated topic label attached to a paper at ingestion time. Powers the tag filter
that scopes search to matching papers (via `Manifest.paper_ids_for_tags`, OR semantics).

- Code: `generate_tags` in `src/rag/tagger.py`; stored on the manifest record.
- `_Avoid_:` label, category, topic, keyword.

### 🤖 Agentic RAG / the agent

The chat loop that reasons and, only when needed, calls the `search_papers` tool — a ReAct
loop over the model's native tool calling. Each tool call is an **Action**, the returned
passages the **Observation**, the model's reasoning a **Thought**; the UI renders the
Thought → Action → Observation **trace**.

- Code: `ChatAgent` in `src/server/agent.py`.
- `_Avoid_:` chatbot, RAG chain, assistant loop. The single tool is `search_papers`.

### 🔄 Turn

One request/response cycle of the chat loop: get-or-build the **agent**, run it, persist the
result, and stream it. Orchestrated by `run_turn`, called from `/api/chat`'s route handler; the
route itself owns only the single-flight guard (`ChatStore.try_acquire`/`release`) and the SSE
plumbing around it, since both are tied to the HTTP response, not the turn's logic.

- Code: `run_turn` in `src/server/chat_turn.py`.
- `_Avoid_:` request, exchange, round-trip (say "turn" — it's the established term in
  `docs/architecture.md`'s single-flight-guard note).

### 🔧 Pipeline

The per-paper ingestion sequence: **download → extract (Docling) → index → tag →
manifest**. One function runs it; both the CLI and the worker call it.

- Code: `ingest_paper` in `src/rag/pipeline.py`.
- `_Avoid_:` flow, ETL, job. The `on_stage` callback reports named **stages**.

### ⚙️ Worker (ingestion worker)

The in-process background thread that finds **pending papers** and runs each through the
pipeline — on app startup (`ingestion.auto_start`) and on admin **Re-scan**. Exposes
progress for the Admin page.

- Code: `IngestionWorker` in `src/server/worker.py`.
- `_Avoid_:` daemon, background job, task runner, ingester.

### ⏳ Pending paper

A paper listed in `config.yaml` but not yet in the manifest — i.e. still to ingest.
Computed by diffing the config paper list against the manifest.

- Code: `pending_papers` in `src/rag/pipeline.py`.
- `_Avoid_:` queued, unprocessed, new paper.

### 🗂️ Registry

The `draccus.ChoiceRegistry` config bases whose `type` string selects a variant
subclass carrying only that backend's fields, so adding a backend is one
`@Base.register_subclass("name")` dataclass plus a `build_*` match arm. The five registries
are `EmbeddingCfg`, `RerankerCfg`, `SparseCfg`, `FaithfulnessCfg`, and `LLMSpec` (all in
`src/rag/config.py`).

- Code: the `ChoiceRegistry` bases + variants in `src/rag/config.py`; backend builders in
  `embedders.py`, `reranker.py`, `faithfulness.py`, and `llm.py`.
- `_Avoid_:` factory map, plugin table, dispatch dict, decorator registry (the old
  `@register_*` decorators are gone). Not to be confused with the **manifest**.

### 🔌 LLM backend

A provider adapter behind a uniform interface, selected by the LLM spec's `type`
(`anthropic` · `openai` · `vllm` · `sglang` · `gemini`) and built by `build_llm`. Powers
chat, tagging, and the `llm` reranker.

- Code: `LLMBackend` ABC + `build_llm` in `src/rag/llm.py`; the `LLMSpec` variants in
  `src/rag/config.py`.
- `_Avoid_:` model, client, provider (say "backend"; the config *string* that selects it
  is `type`, not `provider` — that key was renamed).

### ✅ The gate

The five commands that define "done", matching CI:
`ruff format --check` · `ruff check` · `ty check` · `check_docs` · `pytest`. See
[CONTRIBUTING.md](CONTRIBUTING.md).

- `_Avoid_:` checks, lint step, CI suite (say "the gate").

### 🌱 Config / project root

`config.yaml` is the single source of truth. The **project root** is the nearest
`pyproject.toml` ancestor of the config file, and every relative path in the config is
anchored there — so a config in `configs/` still writes to the repo root, and commands work
from any working directory. Serve/ingest locate it by explicit `--config_path` →
`PAPERLENS_CONFIG` env → upward search from the CWD. Eval uses `--config` for its explicit
form, then the same fallbacks.

- Code: `load_config` / `_project_root` in `src/rag/config.py`.
- `_Avoid_:` settings, config dir, working dir (the root is the pyproject.toml ancestor, not
  necessarily the CWD or the config file's own directory).
