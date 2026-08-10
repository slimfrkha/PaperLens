---
name: calibrate-retrieval-config
description: Run PaperLens's eval harness (`paperlens-eval`) to calibrate a loaded config's retrieval knobs — chunking, reranker, retrieval.candidates — against whatever pool of papers that config points at, and to test whether the `per_paper` retrieval flag (no config.yaml field of its own) is worth turning on for that pool, on single-paper or genuinely cross-paper questions. Use when the user asks to calibrate, tune, optimize, or "dial in" a config for a pool, wants a config.yaml recommendation for their papers, says a config "feels off" for their pool, asks whether `per_paper`/per-paper-scoped retrieval helps, or asks to run/interpret any `paperlens-eval` command — `gen/run/screen/sweep/confirm`, `per-paper sweep/confirm`, or `comparative gen/sweep/confirm`. Works on any config path — never assume a specific config file or paper set. Do NOT use for: explaining how the harness itself is built or why the guards exist (that's docs/harness.md, read it directly), general RAG/agent code changes unrelated to config values, one-off metric questions with no intent to change the config, or a `paperlens-eval` command that crashed/errored — that's a bug to debug, not a calibration run to launch.
---

# Calibrate a PaperLens config with the eval harness

`paperlens-eval` turns a config + its ingested pool into a tuned config block. **The goal
is not to grid-search everything and report the argmax** — it's to find which knob
settings measurably beat the default for this specific pool, stop when nothing does, and
let a human resolve any trade-off. This skill is the operational playbook for running that
loop well — sequencing, reading the reports, and knowing when to stop. It assumes the
harness's design (the two guards, why relevance is section-identity, why estimand gap
exists) — see [docs/harness.md](../../../docs/harness.md) for that; don't re-derive it
here, don't re-explain it to the user unless asked.

Every command below takes `--config <path>`, resolved the same way `paperlens-serve` does.
Never hardcode a config path or assume which papers are loaded — read them from the user's
request or the working directory's config.

## The four laws

**1. Cheapest decision first — screen before sweep, retrieval before chunking.**
`run` and `screen --tier retrieval` are near-instant (cached embeddings, in-memory
rerank-score slicing) — just run them. `screen --tier chunking` and `sweep` **re-index**
and can take tens of minutes to ~2 hours depending on pool size. Never invoke a
re-indexing command synchronously and block the conversation on it: launch it with
`run_in_background: true`, tell the user in one sentence what's running and why (pool
size, cell count if known), and wait for the completion notification — don't poll. Only
reach for `screen --tier chunking`/`sweep` once the retrieval tier has been read; if
retrieval tuning alone already answers the trade-off the user cares about, say so before
paying for a re-index. **Default is inform-and-background, not ask-and-wait** — one
sentence stating what's running and why is enough. **Exception:** when Law 3(b)'s
ceiling-bound specifically fires (default ceiling already within ~2 points of 1.0), that's
a concrete reason the re-index may not be worth its cost — pause and ask there instead of
just backgrounding it. Absent that specific signal, don't manufacture other reasons to ask.
*Overcorrection to avoid:* cost-consciousness is not an excuse to skip stages the user
asked for — warn and background them, don't silently downgrade the ask or refuse to run
them.

**2. Resolution before ranking.** Every report leads with ceiling, MDD, and `n_clusters` —
relay those three in your own summary *before* stating which arm won anything. If the
ceiling is within ~2 points of 1.0, say the ranking's real signal is in the paired metric
delta, not the ceiling. If `n_clusters` is small (the harness warns below 25), label every
ranking from that run "directional" in the same sentence you give it — once, not as a
disclaimer paragraph repeated per arm.
*Overcorrection to avoid:* this is one line of context, not a statistics tutorial. State
the three numbers and the one caveat they imply, then move on.

**3. Stop when a stage is moot — not only when it's null.** There are two distinct reasons
to skip a later command, and both must be named explicitly rather than run reflexively:
   - *Statistical mootness (the easy case):* a screen/sweep's own CI straddles zero or
     misses MDD → report "the default is fine for this pool; tuning this knob won't
     measurably help," not the largest raw (noise) delta dressed up as a win. Only declare
     this once that command has actually run — never preempt a stage by assuming it won't
     find anything.
   - *Structural mootness (the one that's easy to miss):* a later, expensive command's
     value is *conditioned* on an earlier one's outcome, so running it anyway burns cost
     for no new decision-relevant information. Two concrete triggers: (a) `sweep` grids
     retrieval against chunking *survivors* — if `screen --tier chunking` found none, `sweep`
     would only re-derive the retrieval screen's already-known ranking at full re-index
     cost, so skip it and say so, naming the dependency (e.g. "chunking screen found no
     survivor, so sweep can't add anything beyond the retrieval screen — skipping it").
     (b) a ceiling within ~2 points of 1.0 bounds chunking's *upside* — no arm can gain
     more than that headroom on recall — but it does **not** bound chunking's *downside*:
     chunking can move stage-2 `MRR` independently of the ceiling (a live run saw
     `max_tokens=1024` drop MRR while `success` stayed flat — a regression the ceiling
     number alone would never surface). So a saturated ceiling is a reason to ask whether
     the re-index cost is worth it for *more upside*, not a reason to skip the screen
     outright — the screen is still the only way to check the current chunking config
     isn't already quietly costing MRR. Surface both angles and let the user decide.
*Overcorrection to avoid:* don't let either form become reflexive defeatism — a delta that
clears MDD, or a stage whose precondition genuinely holds, is a real result and must be run
and reported, not skipped for looking expensive.

**4. Confirm is a one-shot human decision, never an auto-pick.** `screen`/`sweep` can
report metrics pulling in different directions (e.g. `+success` / `-MRR`). Characterize
the trade-off in plain language, then ask the user which values to `confirm` — use
`AskUserQuestion` when the choice isn't obvious from the conversation. Before running
`confirm`, check whether `<fingerprint>.confirm.json` already exists for this pool; if it
does, surface that as a loud but non-blocking note (the test split has already been
touched) instead of silently re-running it. Never re-run `confirm` for a fingerprint that
already has a `confirm.json` without the user explicitly asking to re-touch it — this is a
per-fingerprint guard, not a per-session one, so a `confirm.json` from a prior session
blocks a silent re-run today exactly as it would minutes later in the same session.
*Overcorrection to avoid:* don't turn the ask into an undigested data dump — name which
value each metric favors and by how much before handing over the choice.

## Command reference (all under `paperlens-eval`, all take `--config <path>`)

| Step | Command | Re-indexes? | Notes |
|---|---|---|---|
| 1 (hard dep) | `gen [--limit N] [--genfilter] [--genfilter-threshold F]` | No | Idempotent; regenerates only if the pool's fingerprint changed. It's one LLM call per section, fanned out over every paper — decide sync-vs-background from the paper count *before* invoking it (you can't downgrade a call already in flight): background it unconditionally for anything beyond a handful of papers, same as the chunking-tier commands. |
| 2 | `run [--limit N]` | No | Baseline on the current config, dev split. |
| 3 | `screen --tier retrieval [--candidates ...]` | No | `reranker.enabled` + `retrieval.candidates`, cheapest signal. |
| 4 | `screen --tier chunking [--max-tokens ...]` | Yes, per cell | Only after step 3; background it (unless Law 3(b)'s ceiling-bound fires — then ask first). |
| 5 | `sweep [--max-tokens ...] [--candidates ...]` | Yes, per `max_tokens` | Only if chunking screen found a survivor worth gridding against retrieval. Skip it outright and say so if nothing cleared MDD in step 4. |
| 6 | `confirm [--max-tokens N] [--candidates N] [--rerank/--no-rerank]` | Only if `--max-tokens` differs from the config's value | Touches the **test** split; touched-once (Law 4). Prints the paste-ready `config.yaml` block. |

Only step 1 is a hard dependency of the others (docs/harness.md). Running 1→6 in order is
the recommendation this skill encodes, not something the CLI enforces — a user can ask to
jump straight to `confirm`, which is allowed, but say plainly that it's confirming blind
if screen/sweep haven't run yet.

**Embedding is out of scope for the default flow.** `embedding.*` is plumbed into the
isolated-index machinery but not exercised by `screen`/`sweep` by default — they vary
chunking against the config's own embedder. An alternate-embedder arm needs the user to
supply a second, already-downloaded model (the harness won't fabricate a model download).
If asked to calibrate embedding, say this plainly rather than silently doing nothing.

## 🔬 `per_paper` and `comparative` — a different question, no config block

`per_paper` (`Searcher.search(per_paper=True)`) has no `config.yaml` field — it's a per-call
flag, not a retrieval knob — so these two command families answer *"turn it on or not,"* not
*"what value."* Two separate, standalone tests exist because `per_paper`'s effect can only be
measured against a question shape `gen`'s dev split doesn't contain — see docs/harness.md's
`per-paper` and `comparative` sections for why:

| | Tests | Eval set | First command |
|---|---|---|---|
| `per-paper sweep`/`confirm` | `per_paper` on single-paper-lookup questions crowded into a random multi-paper scope | Reuses `gen`'s dev split — no separate `gen` step | `per-paper sweep --per-paper-n 4 --candidates 10,20,30,50` |
| `comparative gen`/`sweep`/`confirm` | `per_paper` on questions whose gold genuinely spans 2+ papers | Its own `comparative gen`, separate dev/test split | `comparative gen --target-p 5 --max-trials 20` (**pilot only** — see below) |

**Same cost discipline as chunking, for a different reason.** `per-paper sweep` is **the most
expensive command in the harness** (`n_questions × up to 3 arms × |candidates grid|`, each
on-arm point running `n_papers` separate retrievals) — background it exactly like a chunking
screen, state the pool size and grid up front. `comparative gen` costs real LLM calls, not just
extra retrieval, and its prompt is genuinely unvalidated on any new pool — **always run the
cheap pilot first** (`--target-p 5 --max-trials 20`), read a sample of the generated questions
yourself (`sections[].body` is kept in `dev.jsonl` exactly so you don't have to re-open the
papers) to confirm they genuinely need every paper shown rather than being answerable from just
one, *then* background the full run at the defaults. Don't skip straight to `target_p=40,
max_trials=200` on a pool this hasn't been tried on.

**`comparative confirm`'s hard floor.** `comparative confirm` refuses to print `"Confirmed"`
below `n_clusters < COMPARATIVE_MIN_CLUSTERS` (`8`) regardless of what the delta's CI says —
stricter than the generic `n_clusters < 25` warning from Law 2, because a small pool's
cross-paper item count clusters even thinner than its single-paper one. State this plainly
("too few clusters to confirm anything, read as anecdotal") rather than reporting the raw delta
as a finding.

**Split-seed gotcha in `comparative gen`.** `--seed` drives both the trial loop's paper sampling
and the dev/test coin-flip split — a bad seed can produce a badly skewed split (seen in
practice: seed `0` gave 2 test items out of 40 where 10 was expected) purely by chance, not a
bug. Check `<fingerprint>.comparative.meta.json`'s `n_test` against `n_items × test_frac` after
any `comparative gen` run; if it's off by more than a couple of items, try a different `--seed`
before trusting `comparative confirm` on that split.

**Deliverable is a recommendation, not a block.** Neither path ends in a `config.yaml` snippet —
there's no field to set. End with a plain-language answer ("turn `per_paper` on for this pool" /
"leave it off, no measurable benefit") plus the same ceiling/MDD/`n_clusters`-style resolution
caveat Law 2 requires elsewhere.

## Delivering the result

End with the emitted `config.yaml` block from `confirm` (paste-ready as-is — don't
hand-edit it) plus a short caveat line carrying forward the harness's own known limits
(estimand gap: scored on whole questions, production retrieves on decomposed sub-queries)
— one line, not the full limits section from the docs. (For `per_paper`/`comparative`, there's
no block — see above instead.)

## Before delivering — a silent self-check, not a printed checklist

Run these five against your own conduct before presenting the final config block. This
is for you, not the user — don't print it as a checklist; if something fails, go fix the
gap, don't disclaim it.

1. **(Law 1)** Did every retrieval-tier command run synchronously, and every chunking-tier
   command either background silently or pause for a go/no-go — never block the
   conversation on a re-index without saying so first?
2. **(Law 2)** Did the summary state ceiling, MDD, and `n_clusters` — with the saturation
   or small-cluster caveat if either applies — *before* naming any arm's ranking?
3. **(Law 3)** For every stage skipped, is there a named reason on record — statistical
   (CI straddled zero / missed MDD) or structural (a dependency didn't hold) — rather than
   a silent omission or a run "for completeness" that added no decision-relevant signal?
4. **(Law 4)** Was `<fingerprint>.confirm.json` checked before invoking `confirm`, and is
   the confirmed config a value the user actually chose (stated directly or via
   `AskUserQuestion`) rather than one picked unilaterally?
5. **(Delivering the result)** Is the `config.yaml` block copied verbatim from `confirm`'s
   own output — not retyped or paraphrased — with the estimand-gap caveat included as one
   line?
6. **(`per_paper`/`comparative` paths)** If a `per-paper`/`comparative` run happened, was its
   cost stated up front (background for `per-paper sweep`, pilot-first for `comparative gen`),
   was the split checked (`n_test` vs. expected `test_frac` share) before trusting
   `comparative confirm`, and does the final answer respect `COMPARATIVE_MIN_CLUSTERS` rather
   than reporting a raw delta as decision-grade?

Any "no" means the flow isn't done — go back and close that gap before presenting a result.

## Acceptance tests

- **Full run request.** "Calibrate this config for its pool" (user gives some `--config`
  path) → `gen` (check freshness) → `run` → `screen --tier retrieval` reported with
  ceiling/MDD/n_clusters → ask before/background the chunking screen → summarize
  trade-offs → ask which config to `confirm` → deliver the block. Does not auto-run
  `confirm` unprompted, and the reported config/pool are whatever the user actually
  pointed at — never a filename assumed from a prior session.
- **Trade-off surfaced, not resolved.** Retrieval screen shows `candidates=50`:
  `+0.028 success`, `-0.002 MRR` → both numbers stated in plain language, then the user is
  asked which to confirm — the skill does not pick `candidates=50` on its own.
- **Genuine null result (statistical mootness).** Chunking screen: every knob's CI
  straddles zero → reported as "chunking doesn't move this pool; keep the default" — no
  fabricated "best" `max_tokens`, and `sweep` is skipped with a one-line reason.
- **Structural mootness.** `screen --tier chunking` found no survivor → `sweep` is skipped
  even though its own CI was never computed, with the reason named ("no chunking survivor
  to grid against retrieval, sweep would just re-derive the retrieval screen's ranking at
  re-index cost") — not run "for completeness," and not silently omitted without saying why.
- **Ceiling-bounded stage.** `run`'s ceiling is 0.99 under the default → before launching
  the (expensive) chunking screen, the skill states both angles — upside capped at ~1
  point of recall, but the screen is still the only way to catch a chunking-driven MRR
  regression the ceiling wouldn't show — and asks whether the re-index cost is worth
  paying, rather than either launching unprompted or skipping the screen outright because
  the ceiling looks saturated.
- **Skip-ahead request.** User says "just confirm with candidates=30" → allowed, but
  `<fingerprint>.confirm.json` is checked first; if present, the skill surfaces the
  touched-once warning before re-running rather than silently proceeding twice.
- **Small pool.** `n_clusters` below 25 on every stage → each ranking gets one
  "directional, not conclusive" clause inline, not a repeated multi-paragraph disclaimer.
- **Out-of-scope: "why does MRR ignore chunk count?"** → answered by pointing at
  docs/harness.md's Metrics section, no tuning run launched.
- **Out-of-scope: unrelated agent/RAG code change** → skill stays out of the way; that's
  a separate task.
- **`per_paper` question.** "Does turning on per-paper retrieval help this pool?" → routes to
  `per-paper sweep` (background, cost stated up front — the most expensive command in the
  harness) → `per-paper confirm` with a fresh required seed on the point the user picks →
  plain-language recommendation, no config block.
- **Cross-paper question.** "Does per_paper help with genuinely multi-paper questions?" →
  `comparative gen` pilot first (`--target-p 5 --max-trials 20`), the generated questions
  read by eye for genuine cross-paper dependence, before the full run at defaults →
  `comparative sweep` → `comparative confirm`. A result with `n_clusters <
  COMPARATIVE_MIN_CLUSTERS` is reported as anecdotal, never as "confirmed."
- **Split sanity check.** After any `comparative gen` run, `n_test` in `meta.json` is compared
  against the expected `test_frac` share before trusting `comparative confirm` — a skewed
  split (e.g., 2 of 40) is treated as a reason to retry with a different `--seed`, not ignored.
