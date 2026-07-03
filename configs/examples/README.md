# Example run configs

Copy-me `config.yaml` templates for common PaperLens setups. Each is a **complete,
standalone** config (PaperLens loads one file whole — there is no merge/overlay), so pick
the one closest to your setup, copy it into `configs/`, and edit the `papers` list.

[`reference.yaml`](reference.yaml) is the annotated master: **every key, its default, and
every accepted value**, commented. Read it to see what's tunable; start from a focused
template below to actually run.

```bash
cp configs/examples/anthropic.yaml configs/my-setup.yaml   # then edit `papers:`
```

Point any command at a config with `--config_path` (or the `PAPERLENS_CONFIG` env var); the
`make` targets take `CONFIG=`:

```bash
uv run paperlens-serve  --config_path configs/my-setup.yaml
uv run paperlens-ingest --config_path configs/my-setup.yaml
make serve CONFIG=configs/my-setup.yaml
# or:  export PAPERLENS_CONFIG=configs/my-setup.yaml
```

Cloud backends read their key from the env var named in `api_key_env` (put it in `.env`),
and install as extras: `uv sync --extra anthropic` / `--extra gemini`.

| Template | Embedder / reranker | Tagging + chat LLM | Needs |
|---|---|---|---|
| [`local-gpt-oss.yaml`](local-gpt-oss.yaml) | local HF (`bge-m3` / `bge-reranker`) | OpenAI-compatible server (LM Studio, vLLM, sglang, llama.cpp…) | a running local endpoint |
| [`anthropic.yaml`](anthropic.yaml) | local HF | Anthropic (Claude) | `--extra anthropic`, `ANTHROPIC_API_KEY` |
| [`gemini.yaml`](gemini.yaml) | Google Gemini | Gemini | `--extra gemini`, `GEMINI_API_KEY` |
| [`ollama.yaml`](ollama.yaml) | Ollama (`/api/embed`) | Ollama (`/v1`) | a running `ollama serve` |

Every key is documented in [`reference.yaml`](reference.yaml) and, in prose, in
[docs/configuration.md](../../docs/configuration.md). Paths are anchored to the config
file's directory, so relative paths keep working wherever the file lives.
