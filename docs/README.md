# 📚 PaperLens documentation

PaperLens is a local, config-driven RAG app over arXiv papers:
ingest papers, then chat with an agent that retrieves passages and answers with
clickable, grounded citations. Everything is driven by one `config.yaml`. ⚙️

This hub is organized by what you're trying to do. Start with the project
[README](../README.md) for the one-paragraph pitch and the fastest install; come
here when you want to go deeper.

## 📋 At a glance

- **Config-driven.** A single `config.yaml` (typed dataclasses via `draccus`, with
  `ChoiceRegistry` variants) is the source of truth for both flows.
- **Ingestion.** arXiv PDF → Docling markdown → section-aware chunk → embed → Chroma
  index → LLM tags → manifest.
- **Retrieval.** A FastAPI `ChatAgent` runs a **ReAct loop** with one `search_papers`
  tool over a **two-stage `Searcher`** (dense recall → cross-encoder/LLM rerank), with
  opt-in hybrid dense+BM25 fusion and multi-query paraphrase expansion.
- **Pluggable backends.** Embedders (`hf` · `openai` · `gemini` · `ollama`), rerankers
  (`hf` cross-encoder · `llm`), and LLM providers — each selected by config.
- **Faithfulness (opt-in).** A post-generation check verifies each cited sentence
  against the passage it cites and flags entailment/neutral/contradiction, without
  altering the answer.
- **Streaming UI.** SSE streams the answer and its Thought → Action → Observation trace
  to a React + Mantine frontend.
- **Annotate & export.** Highlight passages and attach notes in the Paper Viewer;
  per-turn 👍/👎 feedback; copy an answer as Markdown footnotes or BibTeX.
- **Eval harness.** `paperlens-eval` tunes chunking/reranker/retrieval config for
  *your* paper pool, with paired statistics and index-isolation guards.
- **Grounded & maintainable.** Clickable citations backed by a reference registry;
  offline tests, docs, and a clean one-way import graph.

## 🧭 Find your path

| I want to… | Go to | Type |
|---|---|---|
| 🚀 Get the app running and ask my first question | [Getting started](getting-started.md) | Tutorial |
| 📋 See everything the app does, feature by feature | [Features](features.md) | Reference |
| ⚙️ Look up a `config.yaml` key, a command, or an API route | [Configuration & commands](configuration.md) | Reference |
| 🧩 Add a paper, swap the LLM/embedder, use a cloud provider, re-tag, tune retrieval config | [How-to guides](how-to.md) | How-to |
| 🏛️ Understand chunking, two-stage retrieval, and the agent loop | [Architecture](architecture.md) | Explanation |
| 🎛️ Optimize config for your pool — why the eval harness works the way it does | [Eval harness](harness.md) | Explanation + Reference |
| 🤝 Set up a dev environment and contribute | [CONTRIBUTING](../CONTRIBUTING.md) | Reference |
| 📖 Learn the project's vocabulary | [CONTEXT (glossary)](../CONTEXT.md) | Reference |

## 🗺️ The shape of the system

```text
config.yaml ─┬─> ingestion worker: download → markdown (Docling) → index (Chroma) → LLM tags
             └─> FastAPI backend ── agentic RAG ──> LLM (Claude | any OpenAI-compatible server)
                       │  tool: search_papers → Searcher (embedder + reranker)
                       └─> React + Vite + Mantine UI: Chat · Papers · Admin
```

Two flows meet at the vector index: **ingestion** fills it, **retrieval** reads it.
See [Architecture](architecture.md) for the full picture. 🏛️
