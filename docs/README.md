# 📚 PaperLens documentation

PaperLens is a local, config-driven RAG app over arXiv model technical reports:
ingest papers, then chat with an agent that retrieves passages and answers with
clickable, grounded citations. Everything is driven by one `config.yaml`. ⚙️

This hub is organized by what you're trying to do. Start with the project
[README](../README.md) for the one-paragraph pitch and the fastest install; come
here when you want to go deeper.

## 🧭 Find your path

| I want to… | Go to | Type |
|---|---|---|
| 🚀 Get the app running and ask my first question | [Getting started](getting-started.md) | Tutorial |
| ⚙️ Look up a `config.yaml` key, a command, or an API route | [Configuration & commands](configuration.md) | Reference |
| 🧩 Add a paper, swap the LLM/embedder, use a cloud provider, re-tag | [How-to guides](how-to.md) | How-to |
| 🏛️ Understand chunking, two-stage retrieval, and the agent loop | [Architecture](architecture.md) | Explanation |
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
