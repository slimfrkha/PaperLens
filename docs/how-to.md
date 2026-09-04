# 🧩 How-to guides

> 👤 **For:** someone with PaperLens already running who has a specific task. Each recipe is
> self-contained. For the meaning of every key, see [Configuration](configuration.md); for
> why the pieces fit together, see [Architecture](architecture.md).

- 📄 [Add papers](#add-papers)
- 🏷️ [Re-tag papers](#re-tag-papers)
- 📋 [Export an answer](#export-an-answer)
- 🎛️ [Tune retrieval config for your pool (the eval harness)](#tune-retrieval-config-for-your-pool)
- 🧪 [Calibrate the faithfulness checker](#calibrate-the-faithfulness-checker)
- 🤖 [Switch the chat or tagging LLM](#switch-the-chat-or-tagging-llm)
- 🧬 [Switch the embedder](#switch-the-embedder)
- 🎯 [Use the LLM reranker (no extra model)](#use-the-llm-reranker)
- 🔌 [Add a new LLM backend (code)](#add-a-new-llm-backend)
- ➕ [Add a new embedder backend (code)](#add-a-new-embedder-backend)

---

## Add papers

The Admin page is the fastest path. Paste one or more modern arXiv IDs or arXiv URLs,
then select **Add**. PaperLens writes them to the active config and starts ingestion without
a restart.

To choose a human-readable `paper_id`, edit `papers` in your config instead:

1. Add a line to the active config (for example, `configs/my-setup.yaml`):

   ```yaml
   papers:
     - { name: my-model-report, arxiv_id: "2501.01234" }
   ```

   `name` becomes the `paper_id` (filename stem, manifest key, search filter). Keep it
   short and unique. Quote the `arxiv_id`.

2. Load the edited file. Either restart the server (which starts ingestion when
   `ingestion.auto_start` is on), or run ingestion headlessly:

   ```bash
   uv run paperlens-ingest --config_path configs/my-setup.yaml
   ```

   Only papers not already in the manifest are processed. The Admin **Re-scan** action
   rechecks the configuration already loaded by the running server; it does not reload a
   config file edited on disk.

3. ✅ Verify: the paper appears on the **Papers** page with tags, and the Admin chunk count
   rises. Ask a question the new paper should answer and check the citations point to it.

---

## Re-tag papers

Regenerate tags for already-ingested papers without re-indexing (e.g. after changing the
tagging model):

```bash
uv run paperlens-ingest --config_path configs/my-setup.yaml --retag
```

Both re-tagging and a normal ingest end with a **tag normalization** pass: the LLM sees the
whole tag vocabulary and merges near-duplicates (e.g. `moe` → `mixture-of-experts`) across
all papers, so the tag filter isn't fragmented by spelling variants.

✅ Verify: the printed tags per paper change, the `X -> Y` merges are listed under
`== Normalizing tags ==`, and the **Papers** page / tag filter reflect them.

---

## Export an answer

Every assistant answer has two copy actions below it, in their own row underneath the
👍/👎 feedback control:

- **Copy as Markdown** — the answer text with each `[rN]` marker turned into a `[^N]`
  footnote, plus a `## References` block linking each cited passage's paper on arXiv. Works
  on any answer, cited or not (a small-talk reply just copies as plain text).
- **Copy BibTeX** — one `@misc` entry per distinct paper actually cited in the answer
  (`title` + `eprint`/`archivePrefix` + a `year` derived from the arXiv ID's `YYMM` prefix).
  Disabled when the answer cited nothing.

Both are client-side only — no network call, nothing persisted. **Known limitation:** the
BibTeX entries have no `author` field. The manifest doesn't store paper authors (only
`paper_id`/`title`/`arxiv_id`/`tags`), so this is a fully-offline stub rather than a
complete citation — paste it in and fill in authors by hand, or resolve them from the
`arxiv_id` yourself.

✅ Verify: ask a question that triggers at least one search, click both buttons, and paste
the clipboard contents somewhere — the footnote numbers should match what's shown on screen,
and the BibTeX should have one entry per distinct cited paper.

---

## Tune retrieval config for your pool

`paperlens-eval` (`src/eval/`) is a **per-pool config optimizer**: it generates an eval set from
*your* ingested papers, sweeps `chunking` / `embedding` / `reranker` / `retrieval.candidates`
against it, and prints a paste-ready `config.yaml` block. It never invents a corpus to study —
whatever pool your `config.yaml` has ingested is what gets tuned; swap the pool and re-run `gen`
to retune. See [Eval harness](harness.md) for why it's built this way (the two correctness
guards, the resolution/MDD statistics, the cost model) — this section is just the steps.

1. **Generate the eval set** (once per pool — regenerates automatically if the pool changes):

   ```bash
   uv run paperlens-eval gen --config configs/my-setup.yaml
   ```

   Writes `evals/<fingerprint>.dev.jsonl` / `.test.jsonl` — one question per section, gold is a
   character span in the paper's markdown (not a chunk id), so it survives a re-chunk.

2. **See what the current config gets you:**

   ```bash
   uv run paperlens-eval run --config configs/my-setup.yaml
   ```

   Prints the stage-1 recall ceiling and stage-2 `MRR@k` on the dev split, leading with the
   pool's *resolution* (`n_clusters`, the minimum detectable difference) — read every later
   number against that.

3. **Screen which knobs matter for this pool:**

   ```bash
   uv run paperlens-eval screen --tier retrieval --config configs/my-setup.yaml  # reranker/candidates, no re-index
   uv run paperlens-eval screen --tier chunking --config configs/my-setup.yaml   # chunking knobs, isolated re-index per cell
   ```

   Each knob is screened one-factor-at-a-time against the default, paired, with a CI. A knob
   whose CI straddles zero isn't worth grid-searching *for this pool* — a different pool may
   screen differently.

4. **Grid-search the survivors:**

   ```bash
   uv run paperlens-eval sweep --config configs/my-setup.yaml
   ```

   Stages the `chunking.max_tokens × retrieval.candidates × reranker.enabled` grid, re-indexing
   into a throwaway collection per cell — your ingested collection is never touched or mutated.

5. **Confirm the winner once, on data never used to pick it:**

   ```bash
   uv run paperlens-eval confirm --config configs/my-setup.yaml \
     --max-tokens 256 --candidates 50 --rerank
   ```

   Read `screen`/`sweep`'s report yourself and pass the config you want validated — `confirm`
   doesn't auto-select a winner (the retrieval screen's own success/MRR trade-off needs a human
   call). Omit a flag to keep the current config's value for that knob; running with no flags at
   all confirms the as-shipped default as a baseline. Prints the held-out score, then a
   `config.yaml`-ready block — paste the `chunking`/`embedding`/`reranker` sections directly, and
   just the `candidates:` line under your existing `retrieval:` section (`min_k`/`max_k` are a
   product choice the harness deliberately leaves alone; if `screen --tier elbow` flagged
   `elbow_mad_multiplier`/`elbow_prominence` as worth tuning, paste those two too).

**Stopping rule:** if a report says no delta clears the MDD, that's the honest answer — the
default config is already fine for this pool, and tuning further won't measurably help. Don't
chase noise.

Before trusting a recommendation, read [Eval harness § Known limits](harness.md#️-known-limits-read-before-trusting-a-recommendation) —
in short: the test split is meant to be touched once per pool (`confirm` warns, doesn't block,
on a repeat), every number is scored on whole questions rather than the sub-queries production
actually retrieves on, and `confirm` only covers the axes `sweep`'s grid enumerates
(`max_tokens`/`candidates`/`rerank` — not `overlap_tokens`/`min_tokens`/`noise_ratio`).

---

## Calibrate the faithfulness checker

`faithfulness.enabled` (off by default) attaches an entailment/neutral/contradiction verdict
to each `[rN]`-cited sentence, derived from two thresholds — `contradiction_max` and
`entailment_min` — on the checker's raw `[0, 1]` consistency score (see
[Architecture § Post-generation faithfulness check](architecture.md#-post-generation-faithfulness-check)).
`scripts/calibrate_faithfulness.py` checks whether those thresholds are well-placed, against
a hand-labeled golden set. It's a standalone tool, not part of the `pytest` gate — it loads
the real checker model, and this repo's tests stay offline.

1. Add or extend labeled pairs in `tests/data/faithfulness_pairs.jsonl` — one JSON object per
   line:

   ```json
   {"premise": "<passage sentence>", "hypothesis": "<citing sentence>", "label": "entailment"}
   ```

   `label` is your judgment of whether `hypothesis` is supported by `premise`: one of
   `entailment` / `neutral` / `contradiction`.

2. Run it:

   ```bash
   uv run python scripts/calibrate_faithfulness.py
   ```

   `--config path/to/config.yaml` calibrates against a non-default config's `faithfulness`
   section; `--fixture path/to/pairs.jsonl` points at a different golden set.

3. Read the report: a confusion matrix and precision/recall/F1 per label at the *currently
   configured* thresholds, then the top combos by macro F1 from a coarse threshold sweep
   (computed over the same scores — no extra model calls) — compare against the checked-in
   `contradiction_max`/`entailment_min` in `HFFaithfulnessCfg` (`src/rag/config.py`) to see
   whether they're still well-placed.

✅ Verify: the printed golden-set size matches what you added, and macro F1 at the current
thresholds is in a range you trust before relying on the faithfulness verdict in the UI. The
golden set here is a proxy, not ground truth on real hallucinations — see the caveat in
[Architecture](architecture.md#-post-generation-faithfulness-check).

---

## Switch the chat or tagging LLM

Edit `llm.chat` (the agent) or `llm.tagging` (ingestion tags) in your config. The chat
model **must support tool/function calling**.

☁️ **A cloud provider (Anthropic):**

```yaml
llm:
  chat:
    type: anthropic
    model: claude-opus-4-8
    api_key_env: ANTHROPIC_API_KEY
```

Put the key in `.env`: `ANTHROPIC_API_KEY=sk-...`. Use `type: gemini` (key
`GEMINI_API_KEY`) for Google — every provider goes through LiteLLM, so there's no
extra to install either way.

🏠 **Any OpenAI-compatible server** (LM Studio, Ollama `/v1`, vLLM, SGLang, llama.cpp):

```yaml
llm:
  chat:
    type: openai            # or vllm / sglang — all speak the OpenAI wire format
    model: <model-the-endpoint-serves>
    api_base: http://127.0.0.1:1234/v1
    api_key_env: LOCAL_LLM_KEY   # local servers ignore the key
```

Restart `paperlens-serve`. ✅ Verify: ask a question and confirm the agent still searches and
cites. If it never calls the tool, the served model likely lacks tool-calling support.

---

## Switch the embedder

Set `embedding.type` in your config to `hf`, `openai`, `gemini`, `voyage`, or `ollama`,
and set `model` (plus `api_base`/`api_key_env` for API types). Examples:

```yaml
embedding:
  type: ollama
  model: bge-m3
  api_base: http://localhost:11434
```

```yaml
embedding:
  type: gemini
  model: text-embedding-004
  api_key_env: GEMINI_API_KEY
```

```yaml
embedding:
  type: voyage       # Anthropic's recommended embedding partner — no embeddings API of its own
  model: voyage-3.5
  api_key_env: VOYAGE_API_KEY
```

> ⚠️ **Changing the embedder changes the vectors.** The embedder name is baked into the Chroma
> collection, so switching means re-indexing. Delete the RAG DB directory (`paths.rag_db`)
> and re-ingest, or use a fresh `collection` name.

✅ Verify: re-ingest one paper, then search — you should get sensible passages.

---

## Use the LLM reranker

Rerank with the chat model instead of loading the local cross-encoder (no extra model or
download):

```yaml
reranker:
  type: llm
  enabled: true
```

The `llm` reranker reuses `llm.chat`. ✅ Verify: search still returns results; if the model's
scoring response can't be parsed, it degrades to the dense-retrieval order rather than
erroring.

A dedicated rerank API is usually a better fit than `llm` — purpose-built for this instead
of a prompt-and-parse workaround:

```yaml
reranker:
  type: voyage
  enabled: true
  model: rerank-2.5       # default
  api_key_env: VOYAGE_API_KEY
```

---

## Add a new LLM backend

Every provider goes through one `LiteLLMBackend`, backed by
[LiteLLM](https://docs.litellm.ai/)'s multi-provider `completion()` — adding a
LiteLLM-supported provider (Bedrock, Vertex, Cohere, ...) is a config variant + one
`_litellm_provider()` arm, not a new backend class:

1. In `src/rag/config.py`, register an `LLMSpec` variant carrying that provider's fields:

   ```python
   @LLMSpec.register_subclass("myprovider")
   @dataclass
   class MySpec(LLMSpec):
       api_base: str = ""          # only the keys this provider needs
   ```

2. In `src/rag/llm.py`, add one arm to `_litellm_provider()` mapping the spec to
   LiteLLM's provider prefix (the `provider` half of its `"provider/model"` model
   string — see [LiteLLM's provider docs](https://docs.litellm.ai/docs/providers)):

   ```python
   # in _litellm_provider(spec):
   #     case MySpec(): return "myprovider"
   ```

   If the provider speaks the OpenAI wire format already, subclass `OpenAISpec`
   instead (see `VLLMSpec`/`SGLangSpec`) — it already routes through
   `_litellm_provider`'s `OpenAISpec()` arm, no new arm needed.

3. Set `type: myprovider` in a config LLM spec. `build_llm` dispatches on the variant.

4. ✅ Verify: add a unit test alongside `tests/unit/test_llm.py` (mock
   `litellm.completion` — see the existing `_FakeCompletionSeq` pattern there). Run the
   [gate](../CONTRIBUTING.md).

---

## Add a new embedder backend

Same ChoiceRegistry pattern across `config.py` + `embedders.py`. If the provider is one
LiteLLM already supports and doesn't need a provider-only param LiteLLM drops (see the
Voyage caveat below), route through `litellm.embedding()` like `OpenAIEmbedder`/
`GeminiEmbedder`/`OllamaEmbedder` do — otherwise call the API directly, like
`VoyageEmbedder`.

1. In `src/rag/config.py`, register an `EmbeddingCfg` variant with the fields it needs:

   ```python
   @EmbeddingCfg.register_subclass("myembedder")
   @dataclass
   class MyEmbeddingCfg(EmbeddingCfg):
       api_base: str = ""
   ```

2. In `src/rag/embedders.py`, subclass `Embedder` and add a `build_embedder` arm:

   ```python
   class MyEmbedder(Embedder):
       def name(self) -> str: ...                       # namespaces the Chroma collection
       def __call__(self, input: list[str]) -> list[list[float]]: ...

   # in build_embedder(cfg):
   #     case MyEmbeddingCfg(): return MyEmbedder(cfg.model, batch_size=cfg.batch_size)
   ```

   Implement `embed_query` too if queries embed differently from documents (see the Gemini
   embedder). Before wiring a provider through `litellm.embedding()`, check its
   `map_openai_params` in the installed litellm's source for any param (like Gemini/Voyage's
   asymmetric `input_type`) your embedder needs — some providers only forward
   OpenAI-standard params (`dimensions`/`encoding_format`/`user`), silently dropping
   provider-specific ones. If it would drop a param you need, call the API directly instead
   (see `VoyageEmbedder`).

3. Set `embedding.type: myembedder`. `build_embedder` dispatches on the variant.

4. ✅ Verify: add a test near `tests/unit/test_embedders.py`; run the gate. Remember a new
   embedder means a fresh index (see the note above).
