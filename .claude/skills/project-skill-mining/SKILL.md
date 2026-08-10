---
name: project-skill-mining
description: Mine the current project's session history for repeated manual workflows worth packaging, and produce evidenced skill briefs. Use whenever the user asks what they should automate or turn into skills — "what should I automate?", "mine this project's history", "what workflows keep coming up", "review my recent sessions for patterns", "I feel like I keep doing the same thing every session" — even if the word "skill" never appears. An unscoped "what should I automate?" belongs to this skill, scoped to the current project. Do NOT use when the user already knows what skill they want ("create a skill for X" is direct creation), for ordinary progress retrospectives, for applying an existing skill, or for anything spanning multiple projects or the user's whole history — decline those in one line as outside this skill's scope.
---

# Project Skill Mining

Mine this project's session history for workflows worth packaging. The deliverable is
**skill briefs** — evidenced proposals — never the skills themselves. Diagnose now;
create later, one item per follow-up, after the user signs off on that item.

## Laws

These govern everything below. When any instruction seems to conflict, reason from the
laws; when prose conflicts with the acceptance tests at the end, the tests win.

1. **Friction over frequency.** Repetition alone never qualifies a candidate — a
   workflow the default behavior already handles well needs no skill, however often it
   recurs. Require at least one quotable instance of friction: the user corrected the
   same default behavior, re-pasted the same context, re-explained the same conventions,
   or redid output that came back in the wrong shape. "Recurs, but no delta" is a
   mandatory skip reason.
2. **Evidence or it didn't happen.** Every claim cites sessions and dates. Track
   provenance: the user's repeated behavior counts; your own repeated *suggestions* do
   not — a pattern you proposed three times and the user never adopted is not their
   workflow. Zero candidates is a success state. Prediction ("bound to come up again")
   never substitutes for evidence; it routes to `needs-more-evidence`, never to a brief.
   You will feel pressure to find candidates because you were asked to — thin evidence
   reported as thin is the correct output.
3. **Briefs, not skills.** Hard stop after the report. Never create or modify any skill,
   agent, or automation in the mining run — not even an "obvious win." Creation happens
   only after per-item sign-off, one item per follow-up turn, built to that brief's
   acceptance scenarios.
4. **New skills are a cost.** Every skill added consumes context, competes with other
   descriptions for triggering, and accrues maintenance. At most 3 briefs per run. Check
   every candidate against skills already visible in this project — repo-level and
   user-level; within trigger distance, recommend `extend` instead. `extend` is the
   default that `create new` must beat with a stated reason.
5. **Stay in the project.** Probe which history sources exist for *this project* —
   conversation/session search, project session files, transcript directories — and use
   what's there; a missing source is a coverage note, not a failure. Never hardcode
   source or product names. Even when wider history is reachable, do not mine it: the
   one-line generality note (below) is the ceiling for cross-project observations.

## Procedure

1. **Probe** this project's history sources; note what's reachable and what isn't.
2. **Bound the scan** — default: last 30 days or last ~50 sessions of this project,
   whichever is smaller. State the bound; the user can widen it.
3. **Mine in two passes:** first an undirected chronological skim of recent sessions,
   then targeted queries for suspected patterns. Targeted search only finds what you
   already suspect; the undirected pass is what discovers.
4. **Build a candidate table:** workflow, dated occurrences, friction instances (quoted
   or tightly paraphrased), provenance (user behavior vs. your suggestion).
5. **Apply the gates.** Every failing candidate lands in the skip list or
   `needs-more-evidence` with a one-line reason — nothing is dropped silently, so the
   user can overrule.
6. **Report** in this order: briefs (≤3), skip list, `needs-more-evidence` list,
   coverage statement (≤3 lines: sources scanned, window, sources unavailable).
7. **Stop and wait.** On sign-off for an item, create that one item in the follow-up
   turn, to its brief's acceptance scenarios.

## Qualification gates (all pass/fail)

- **Friction:** ≥1 quotable friction instance. No exceptions.
- **Recurrence:** ≥3 occurrences in this project, or ≥2 when both carry friction.
- **Stability:** stable inputs, a repeatable procedure, a checkable output or stopping
  condition.
- **Collision:** no visible skill within trigger distance — otherwise recommend
  `extend`, framed as a diff to that skill.
- **Sensitivity:** workflows centering on credentials, personal, or confidential
  material → skip, with a reason, without reproducing the material anywhere.

## Brief format

- Name + one-line job.
- Observed trigger phrasings — verbatim from sessions, never invented.
- Delta statement: what the default behavior did wrong, slowly, or repeatedly needed.
- Evidence: dates, session references, minimal quotes; provenance marked.
- Recommended form with a concrete output: `skill` (SKILL.md at a repo path) · `extend`
  (change to a named existing skill) · `subagent` / `automation` (only where this
  environment supports them) · `skip` · `needs-more-evidence`. Install target is always
  this project.
- Generality note (only when warranted, one line): "pattern looks project-agnostic —
  likely recurs elsewhere; not assessed here." A flag, not a claim; never gather
  evidence outside the project to back it.
- 2–3 acceptance scenarios, at least one inverse (an input where the future skill must
  do nothing).
- Cost note: what it could collide or overlap with.

## Guardrails on this skill's own biases

- *Friction hunting:* Law 1 taken to its extreme reads every minor correction as a
  candidate. The recurrence gate, the 3-brief cap, and the null-result norm exist to
  stop that.
- *Scope creep:* reachable global history tempts a "bonus" cross-project analysis. The
  generality note is the ceiling.
- *Coverage theater:* methodology belongs in ≤3 lines at the end, never a preamble.

## Calibration

A passing brief looks like this (format illustration, not a template to pad):

> **api-error-conventions** — apply this repo's error-envelope and logging conventions
> when writing handlers.
> Observed: "wrap it in the standard error envelope", "no — use AppError like the other
> handlers".
> Delta: default output used bare exceptions and ad-hoc JSON errors; corrected in 3
> sessions.
> Evidence: Jun 3, Jun 11, Jun 24; user re-pasted `errors.md` twice (user corrections,
> not my suggestions).
> Form: `skill` at `.claude/skills/api-error-conventions/`. Install target: this project.
> Acceptance: new-handler request → envelope + logger without prompting; unrelated
> script request → skill silent.
> Cost: nearest existing skill is a review persona; no trigger overlap.

A null result looks like this:

> No candidates. Scanned project `acme-api`: 42 sessions, May 28–Jun 27, via session
> search and project files; no transcript archive available. Nothing recurred with
> friction — the defaults are holding, which is the good outcome.

## Acceptance tests

Tests outrank prose.

1. Sparse, varied one-off history → "no candidates," coverage statement, zero briefs.
2. Workflow repeated 5×, handled perfectly by default → skip: "recurs, no delta."
3. Same schema doc pasted across 4 sessions + output format corrected twice → brief with
   dated citations, delta statement, acceptance scenarios.
4. Pattern overlapping an existing repo or user-level skill → `extend <skill>`, not new.
5. "Create a skill for code review" → this skill stays out; direct creation flow.
6. "Review my skills across all my projects" → decline in one line as out of scope; no
   partial attempt.
7. 40 sessions, 9 plausible candidates → 3 briefs; the rest in skip /
   `needs-more-evidence` with one-line reasons.
8. You proposed the same refactor in 3 sessions, user never adopted it → not a candidate.
9. Lint/test scaffolding set up once in this project, looks obviously reusable → fails
   recurrence; `needs-more-evidence` with a generality note — not a brief.
10. Candidate workflow centers on credential handling → skip: sensitive, no material
    reproduced.
