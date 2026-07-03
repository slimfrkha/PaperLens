# 🎓 Getting started

> 👤 **For:** someone who just cloned PaperLens and wants it running.
> 🎯 **You'll finish with:** the app serving locally, a few papers ingested, and a
> grounded, cited answer to your first question.

This is a guaranteed-success path: the happy path only. For every option and alternative,
see [Configuration & commands](configuration.md).

## 📋 Prerequisites

- 🐍 **Python 3.14+** (the project pins it in `.python-version`).
- 📦 **[uv](https://docs.astral.sh/uv/)** — manages the virtualenv, dependencies, and scripts.
- 🟢 **Node.js** and **npm** — for the frontend.
- 🤖 An **LLM endpoint**. The default config (`configs/recent-oss-agentic-models.yaml`)
  points at a local OpenAI-compatible server (LM Studio at `http://127.0.0.1:1234/v1`). A
  cloud provider works too — see [step 3](#3-point-it-at-an-llm).

## 1. Install

```bash
uv sync
npm --prefix web install
```

`uv sync` creates a `.venv`, installs the locked dependencies, and puts the `paperlens-serve`
and `paperlens-ingest` commands on the path. Run project commands with `uv run <cmd>`.

Verify:

```bash
uv run paperlens-serve --help
```

✅ You should see the command run without an import error. Press `Ctrl-C` if it starts
serving.

## 2. Look at the configuration

Open `configs/recent-oss-agentic-models.yaml` — the default run config. A config file is
the single source of truth for paths, models, the server, and the **paper list**. You
don't need to change anything yet — it ingests a set of recent OSS model reports. For every
key and its options, see [`configs/examples/reference.yaml`](../configs/examples/reference.yaml).

## 3. Point it at an LLM

The chat model must support tool/function calling (the agent calls a `search_papers`
tool). Pick one:

- 🏠 **Local server (default).** Start LM Studio (or any OpenAI-compatible server) on
  `http://127.0.0.1:1234/v1` and load a tool-calling model. No API key needed.
- ☁️ **A cloud provider (Anthropic/Gemini) or a different local port.** See
  [How-to: use a cloud or different LLM](how-to.md#switch-the-chat-or-tagging-llm). Come
  back here when it's configured.

## 4. Start the app

```bash
uv run paperlens-serve --config configs/recent-oss-agentic-models.yaml
```

This serves `http://127.0.0.1:8000` **and** auto-starts the ingestion worker, which begins
downloading and indexing the papers from the config. On first run the database is empty,
so the UI says so while papers ingest.

> 💡 `make serve` runs exactly this; it defaults `CONFIG` to the same file. Override with
> `make serve CONFIG=configs/examples/anthropic.yaml`.

✅ You should see startup logs and, as papers process, ingestion progress.

## 5. Start the frontend

In a second terminal:

```bash
npm --prefix web run dev
```

Open `http://localhost:5173`. The dev server proxies `/api` to the backend on port 8000.

✅ Go to the **Admin** page. You should see paper and chunk counts climbing and a live
progress bar while ingestion runs. Wait until at least one paper finishes.

> 💡 **Tip:** to run backend and frontend together, use `make dev` instead of steps 4–5.

## 6. Ask your first question

Open the **Chat** page and ask something answerable from the papers, for example:

> How does DeepSeek-V3 reduce the KV cache?

✅ You should see the agent's **trace** (Thought → Action → Observation) as it searches,
then a streamed answer with `[r1]`, `[r2]` citations. Click a citation — it opens the paper
on the **Papers** page with the cited passage highlighted.

> ⚠️ If the agent answers with no citations for a paper question, retrieval returned
> nothing — confirm on the Admin page that ingestion actually finished.

## 🎉 What you just did

You installed with uv, picked one config, ran the server (which ingested papers), and got
a grounded, cited answer from the agent.

## 👉 Where next

- ⚙️ Change models, paths, or the paper list → [Configuration & commands](configuration.md).
- 🧩 Add a paper or swap a backend → [How-to guides](how-to.md).
- 🏛️ Understand how retrieval and the agent work → [Architecture](architecture.md).
