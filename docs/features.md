# Features

This page is for people deciding whether PaperLens fits their workflow. It summarizes the
current product surface without duplicating the configuration and architecture references.

## What PaperLens is for

PaperLens is a local, single-user research app for asking grounded questions across arXiv
papers. It ingests a configured paper library, retrieves relevant passages, and asks a
tool-calling LLM to answer with citations that open the exact source passage.

It is a good fit when you want to:

- work with a curated library of modern arXiv papers;
- run the app and its data locally while choosing local or cloud model backends;
- inspect the agent's searches instead of accepting an opaque answer; or
- compare several papers and keep notes alongside the sources.

It is not an authenticated, multi-user service or a generic upload-and-chat product. See
[architecture: why arXiv-specific](architecture.md#-why-arxiv-specific--what-wont-generalize)
for the structural assumptions.

## Ask and compare

### Grounded chat

- Answers stream into the browser over Server-Sent Events (SSE).
- The agent can make several focused `search_papers` calls before answering. Small talk can
  skip retrieval.
- Each `[rN]` citation opens the cited paper at the highlighted passage.
- Source cards group used citations by paper and identify semantic, keyword, or combined
  retrieval provenance.
- A collapsible Thought → Action → Observation trace shows the searches and passages the
  model used.
- Each completed turn shows token usage when the backend reports it and the total latency.
- You can stop a running turn. PaperLens persists the text already streamed and unlocks the
  conversation for the next message.

### Search scope and answer modes

You can restrict retrieval by paper, tag, or both before the first turn in a conversation.
The selected scope stays fixed for that conversation.

PaperLens offers three answer modes:

- **Ask** searches the selected scope as one pool. You can optionally broaden recall by
  retrieving from each paper separately before the shared rerank step.
- **Compare** runs an independent search-and-answer pass for every paper, then synthesizes
  those answers. A carousel lets you inspect each paper's answer and trace. Large comparisons
  require confirmation because cost and latency grow with the number of papers.
- **Auto** asks the configured tagging LLM to choose Ask or Compare. With fewer than two
  papers in scope, or if classification fails, it uses Ask.

Compare tolerates one paper-level failure and preserves the other results. Compare and Auto
turns are stored with the conversation and restore on reload.

### Conversation controls

- Create, resume, and delete saved conversations.
- Edit an earlier user message and resend from that point. PaperLens confirms before
  discarding later turns; it does not preserve branches.
- Rate an answer with thumbs up or down and attach an optional note.
- Copy an answer as Markdown footnotes or as one BibTeX entry per cited paper. BibTeX export
  omits authors because the manifest does not store them.

## Read and annotate papers

### Paper library and viewer

- Browse ingested papers with their titles, tags, and chunk counts.
- Read rendered Markdown with tables, mathematics, heading links, and figures extracted
  during ingestion.
- Follow a citation directly to its highlighted passage.
- Use the generated contents rail to jump between sections.
- Remove a paper after confirmation; PaperLens removes its config entry, index chunks,
  cached files, annotations, and manifest record.

Extracted figures are display-only. They are not chunked, embedded, or retrieved.

### Notes

- Highlight selected text or attach a note to it in the paper viewer.
- Edit, delete, and jump back to saved annotations.
- Browse annotations across the library on the Notes page, with text and paper filters.
- Export the filtered notes view as Markdown.
- If re-extraction changes the text and an annotation cannot be re-anchored, the viewer marks
  it as not found instead of silently attaching it elsewhere.

## Manage the library

The Admin page lets you add one or more modern arXiv IDs or arXiv URLs. It reports queued,
duplicate, invalid, and failed inputs separately, updates the active config, and triggers one
ingestion run for the batch.

It also shows:

- paper, chunk, and pending-paper counts;
- the tag directory and counts;
- current ingestion stage and progress; and
- ingestion errors.

**Re-scan** asks the worker to recheck pending papers in the configuration already loaded by
the server. It does not reload manual edits made to the config file on disk. Restart the
server or run `paperlens-ingest --config_path ...` after a manual edit.

## Retrieval and grounding

PaperLens combines several configurable stages:

1. **Section-aware chunking** reconstructs heading breadcrumbs from section numbers and
   prepends them to the embedded text.
2. **Dense recall** retrieves candidate chunks from Chroma.
3. **Hybrid retrieval** can add BM25 lexical recall and fuse both rankings with reciprocal
   rank fusion (RRF).
4. **Multi-query expansion** can generate paraphrases and fuse their result rankings.
5. **Reranking** uses a local cross-encoder, the chat LLM, or Voyage's rerank API.
6. **Elbow cutoff** chooses a bounded result count from the reranked score drop-off.

Hybrid retrieval and multi-query expansion are off by default. The
[`paperlens-eval` harness](harness.md) can measure them and tune retrieval settings against
your own paper pool.

An optional post-generation faithfulness check scores each cited sentence against its cited
passage and labels it as entailment, neutral, or contradiction. It is a diagnostic signal;
it does not edit or block the answer. See
[architecture: faithfulness](architecture.md#-post-generation-faithfulness-check) for its
limits and calibration model.

## Ingestion and model backends

Ingestion downloads each configured arXiv PDF, extracts Markdown with Docling, creates and
embeds chunks, generates tags, and writes the manifest. Indexing and tag generation overlap,
and cached artifacts avoid repeated work. You can re-tag without re-indexing or re-index
without replacing tags.

Backends are selected in `config.yaml`:

| Component | Supported backends |
|---|---|
| Embedding | Hugging Face, OpenAI-compatible, Gemini, Voyage, Ollama |
| Reranking | Hugging Face cross-encoder, chat LLM, Voyage rerank API |
| Chat and tagging LLMs | Anthropic, Gemini, OpenAI, or OpenAI-compatible vLLM/SGLang servers |
| Sparse retrieval | BM25 |
| Faithfulness | Hugging Face consistency scorer |

The base `uv sync` installs the supported provider dependencies. Local model weights still
download on first use; API-backed configurations avoid loading local models they do not use.

## Where to go next

- Install and ask a first question: [Getting started](getting-started.md).
- Look up keys, commands, and routes: [Configuration and commands](configuration.md).
- Complete a specific task: [How-to guides](how-to.md).
- Understand the design: [Architecture](architecture.md).
- Tune retrieval for a paper pool: [Eval harness](harness.md).
- Change the code: [Contributing](../CONTRIBUTING.md).
