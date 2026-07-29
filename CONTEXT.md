# 📖 CONTEXT — PaperLens domain glossary

One term per concept. Use these names in code, comments, docs, commit messages, and
when talking to an AI assistant about this repo. Each entry lists the definition, where
it lives in the code, and an `_Avoid_:` list of synonyms we deliberately do **not** use.

---

### 📄 Paper

One arXiv model technical report tracked by the app. Declared in `config.yaml` under
`papers` as `{ name, arxiv_id }`. Everything downstream keys off it.

- Code: `Paper` model in `src/rag/config.py`.
- `_Avoid_:` document, article, PDF (a PDF is one *artifact* of a paper, not the paper).

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
(`hf` · `openai` · `gemini` · `ollama`) and built through the **registry**. The same
embedder is used at indexing time and query time — though asymmetric embedders may embed
the same text differently depending on which side it's called from: Gemini's `task_type`,
or `hf`'s optional `query_prefix`/`document_prefix`, via `embed_query()` diverging from
`__call__()`.

- Code: `Embedder` ABC + `build_embedder` in `src/rag/embedders.py`.
- `_Avoid_:` encoder (that is the reranker's cross-encoder), vectorizer, model.

### 🎯 Reranker

The second retrieval stage: rescores candidate chunks against the query. Selected by
`reranker.type` — `hf` (a local cross-encoder) or `llm` (reuses the chat model to score).
`reranker.enabled` toggles the whole stage.

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

### 🔎 Searcher

The object that runs two-stage retrieval: dense **embedder** recall of `candidates`, then
optional **reranker** rescoring, returning the top `k` **Result**s. Accepts a `paper` /
`paper_ids` filter.

- Code: `Searcher` in `src/rag/search.py`.
- `_Avoid_:` retriever, search engine, query engine.

### 📑 Result / passage

One retrieved chunk returned by the Searcher: `score`, `paper_id`, `breadcrumb`,
`section_title`, `text`, `body`, `source`. When a Result is handed to the chat model or shown to
the reader, call it a **passage**. `source` (`"dense"`/`"sparse"`/`"both"`) records which
retrieval pool(s) surfaced it — meaningful only when **hybrid retrieval** is on; otherwise
always `"dense"`.

- Code: `Result` dataclass in `src/rag/search.py`.
- `_Avoid_:` hit, match, document.

### 📎 ref / citation

The `[rN]` marker (`r1`, `r2`, …) the agent assigns to each retrieved passage and threads
into its answer. The frontend turns each `ref` into a clickable **citation** that opens
the paper at the cited passage.

- Code: `ref` registry built in `ChatAgent.run` (`src/server/agent.py`).
- `_Avoid_:` reference number, footnote, source id. The marker is a `ref`; the rendered
  clickable thing is a `citation`.

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
`@Base.register_subclass("name")` dataclass plus a `build_*` match arm. Three of them:
`EmbeddingCfg`, `RerankerCfg`, `LLMSpec` (all in `src/rag/config.py`).

- Code: the `ChoiceRegistry` bases + variants in `src/rag/config.py`; the `build_embedder`
  / `build_reranker` / `build_llm` match dispatch in `embedders.py` / `reranker.py` / `llm.py`.
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

The four commands that define "done", identical locally and in CI:
`ruff format --check` · `ruff check` · `ty check` · `pytest`. See
[CONTRIBUTING.md](CONTRIBUTING.md).

- `_Avoid_:` checks, lint step, CI suite (say "the gate").

### 🌱 Config / project root

`config.yaml` is the single source of truth. The **project root** is the nearest
`pyproject.toml` ancestor of the config file, and every relative path in the config is
anchored there — so a config in `configs/` still writes to the repo root, and commands work
from any working directory. Located by: explicit `--config_path` → `PAPERLENS_CONFIG` env →
upward search from the CWD.

- Code: `load_config` / `_project_root` in `src/rag/config.py`.
- `_Avoid_:` settings, config dir, working dir (the root is the pyproject.toml ancestor, not
  necessarily the CWD or the config file's own directory).
