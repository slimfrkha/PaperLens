"""Agentic RAG chat: a ReAct loop built on the model's native tool calling.

The model reasons, then (only when needed) calls the `search_papers` tool — its
tool call is the parseable Action, the returned passages are the Observation, and
the loop repeats until it answers. Small talk (e.g. "hi", "thanks") is answered
directly with no search. Each search is surfaced as a step so the UI can show the
Thought → Action → Observation trace.
"""

from __future__ import annotations

import string
from collections.abc import Callable
from typing import Literal

from rag.config import Config
from rag.faithfulness import (
    FaithfulnessChecker,
    attribute_refs,
    best_support,
    build_faithfulness_checker,
    split_sentences,
)
from rag.llm import Usage, build_llm
from rag.manifest import Manifest
from rag.search import Searcher


def _search_tool() -> dict:
    return {
        "name": "search_papers",
        "description": (
            "Search the library of arXiv papers and return the most "
            "relevant passages. Call it once per focused sub-question (call it several "
            "times to decompose a multi-part question). Each result carries a unique "
            "`ref` (e.g. r7) — numbered once across the whole conversation, not "
            "restarted per call — you must cite exactly as shown, and `paper=<paper_id>` "
            "for attribution. The result also tells you how many searches you have left "
            "this turn. Do NOT call this for greetings or small talk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused natural-language search query.",
                },
                "paper": {
                    "type": "string",
                    "description": (
                        "Optional paper_id to restrict the search. Only set this when the "
                        "user explicitly names a paper or the question is unambiguously "
                        "about one already-established paper — don't infer it from which "
                        "paper happened to dominate an earlier answer's citations."
                    ),
                },
            },
            "required": ["query"],
        },
    }


SYSTEM_PROMPT = """You are a research assistant for a library of arXiv papers. \
Treat these papers as unknown to you, even ones you recognize by name — you have \
no reliable memory of what any of them actually say. Ground every claim about \
the papers in a passage a search actually returned; don't answer from \
pretraining.

Decide how to answer before doing anything else — if a message mixes more than \
one of these, take the branch that needs the most evidence:
- Greetings, thanks, or simple conversational messages: answer directly, do NOT \
call any tool.
- "What papers do you have?": answer from the papers in scope below, no \
search needed — but the topic hints are routing aids, not evidence: they \
carry no ref, so never state one as a finding, and a tag's absence doesn't \
mean the paper doesn't cover it, only that it wasn't a top tag. Anything more \
specific than titles and hints belongs to the next branch.
- A follow-up on your own prior answer in this conversation ("say more about \
that", "what did you mean by X?"): only your prior text persists across turns, \
not the passages behind it. Restating a claim your last answer already made \
may reuse that ref; anything beyond what you already wrote needs a fresh \
search — don't extend a ref past what you actually retrieved.
- Anything else about the papers or concepts in them — or plausibly in them, \
including questions outside the papers and "does the library cover X?": search first \
rather than guessing it's out of scope; use `search_papers`. If the question \
decomposes into more than one sub-question needing distinct evidence, say in \
one short line which you'll search for before your first call — that line \
becomes the start of your visible answer, so keep it brief, and skip it for a \
single-search question. Comparing two papers is one filtered search per \
paper, not one unfiltered search — the paper argument narrows within scope, \
it never overrides it — and every sub-question needs its own search, even one \
that turns up a passage you already had. You have at most {search_budget} \
searches this turn and must answer within the same turn; a reformulation \
counts against that same budget. Each result tells you how many searches are \
left — stop before you hit zero. If you end up skipping a sub-question or \
answering from a weaker passage than you wanted, say so at the end of your \
answer, not the start — the plan you opened with may not match what you \
actually got to.

Once a search returns:
- If a result doesn't actually answer the sub-question it was for (the common \
case — retrieval rarely returns nothing, it returns plausible-looking chunks \
that miss): don't treat it as coverage. You may reformulate that one \
sub-question's query once; if it's still empty, say in your final answer that \
the papers in scope don't appear to cover this rather than filling the gap \
from memory.
- If nothing you retrieved answers a question that's outside the papers, you \
may answer it from general knowledge instead — label that clearly as not \
from the library.

When writing a query, resolve pronouns/references against the conversation \
first. Include the user's own term alongside the paper's likely technical \
vocabulary, not instead of it — if the user uses an acronym, search for the \
acronym itself, not just its expansion, since the passage may only ever use \
the acronym.

Every result carries paper=<paper_id> — that field is the paper's identity, \
not its content; attribute claims from it, never by guessing which paper a \
passage sounds like. Cite retrieved passages inline with the [rN] markers \
exactly as they appear in the results, and name the paper in prose, not just \
the marker — e.g. "<paper name> reports X [r3]," not "X [r3]." If a title looks \
wrong (garbled, or just an author/org name), use the paper_id instead — the \
id is always the paper's real identity. If \
papers disagree, present both attributed positions rather than reconciling \
them, and say when a passage only partially answers. Only cite refs you \
actually received. Be concise and technical.

{filter_note}
Papers in scope: {papers}"""


CLASSIFY_SYSTEM_PROMPT = """Decide how a research-assistant chat should answer the user's \
latest question over a library of arXiv papers: with one combined answer over the whole \
pool ("ask"), or with a guaranteed independent search+answer for every paper in scope, \
synthesized into one comparative answer ("compare").

Choose "compare" only when the question genuinely needs every paper checked on its own — a \
per-paper fact, or an explicit cross-paper comparison, where one paper dominating a pooled \
search could silently drop another paper's answer. This includes a question that never says \
"each" or "compare" but is clearly asking about a property every paper in scope has its own \
answer to — use the paper list below to judge that, not just the question's wording: a \
question phrased for one paper still needs "compare" when the scope holds several distinct \
papers it could equally be about. Choose "ask" whenever a single answer drawn from the shared \
pool is a faithful answer, including questions that are about the papers collectively rather \
than paper-by-paper.

Examples (illustrative only — never assume any specific paper, topic, or field; the pool in \
scope could be about anything):
- "What's the model size in each paper?" -> COMPARE (a per-paper fact; a pooled search could \
miss one paper's number entirely)
- "What's the model size?" over a scope of several distinct model papers -> COMPARE (no \
"each," but the scope makes it a per-paper fact all the same)
- "What's the model size?" over a scope of one survey paper and one position paper (neither \
proposes a model) -> ASK (the scope doesn't make this a per-paper fact)
- "How do these papers differ in their approach?" -> COMPARE (an explicit cross-paper \
comparison)
- "What's the main idea of the first paper?" -> ASK (one paper, no comparison)
- "Summarize what this library covers on a given topic" -> ASK (one synthesized view over \
the pool is a faithful answer; no per-paper guarantee needed)
- "Thanks, that's helpful" -> ASK (not a research question at all)
- A follow-up like "and their results?" after a per-paper comparison -> COMPARE (resolve it \
against the conversation: it continues the same per-paper question)

Papers in scope for this question:
{papers}

Reply with exactly one word, nothing else: COMPARE or ASK."""


SYNTHESIS_SYSTEM_PROMPT = """You already ran a separate, complete search over each paper \
below and got one answer per paper to the same question. Do not search again — synthesize \
what you already have.

Write ONE answer that directly addresses the question by drawing on every per-paper answer \
given. Where papers agree, say so once instead of repeating it per paper. Where they \
differ, name the papers and state the difference plainly. If a paper's own answer says the \
library doesn't cover something, say that too rather than silently dropping that paper.

Every claim must come from the per-paper answers below, not general knowledge. Reuse each \
[rN] marker exactly as it appears in the source answer you're drawing from — never invent \
a new ref, renumber one, or drop the marker when restating a claim it supports. Name the \
paper in prose alongside the marker, the same convention the source answers already use.

A table is often the clearest way to present a per-paper comparison (e.g. "what's the \
model size in each paper?") — use one when it fits. A table does not exempt you from \
citing: every cell that states a fact still needs its [rN] marker in that same cell, \
e.g. "7B [r3]", exactly like a prose sentence would. A table with no markers at all is \
not an acceptable answer.

A search tool is available but you should not need it — everything required is already in \
the per-paper answers below. Only search again if a per-paper answer is genuinely missing \
information you can't do without.

Per-paper answers:
{per_paper_answers}"""


class InsufficientScopeError(ValueError):
    """Raised by `ChatAgent.compare` when fewer than 2 papers resolve in scope. A narrow
    subclass (not a bare `ValueError`) so `chat_turn.run_turn`'s classify-then-send race
    fallback can catch exactly this guard — not any other `ValueError` that might
    legitimately propagate out of `compare()` after rows/tokens have already streamed
    (e.g. a synthesis retry that fails twice), which would otherwise get silently
    reinterpreted as this race and rerun as a fresh turn on top of what already streamed."""


def _flatten_compare_rows(rows: list[dict]) -> tuple[str, list[dict]]:
    """Shared fallback for a Compare turn that never reaches synthesis — used both by
    `ChatAgent.compare` itself (stopped mid-loop) and by `chat_turn._run_agent_compare`'s
    outer abandon-timeout path (a different, coarser stop race — see there). Flattens
    whatever per-paper rows did complete into one concatenated text, same "persist partial
    results" spirit a normal turn's abandoned-thread fallback already has."""
    text = "".join(f"## {r['title']}\n\n{r['text']}\n\n" for r in rows)
    citations = [c for r in rows for c in r["citations"]]
    return text, citations


class ChatAgent:
    def __init__(
        self,
        cfg: Config,
        searcher: Searcher,
        manifest: Manifest,
        client=None,
        faithfulness: FaithfulnessChecker | None = None,
    ):
        self.cfg = cfg
        self.searcher = searcher
        self.manifest = manifest
        # Inject an LLM client to run offline (tests); default uses the configured chat model.
        self.client = client or build_llm(cfg.llm.chat)
        # Inject a faithfulness checker to run offline (tests); default per cfg.faithfulness,
        # gated at call time by cfg.faithfulness.enabled — same pattern as reranker.enabled.
        self.faithfulness = faithfulness or build_faithfulness_checker(cfg.faithfulness)
        self.search_tool = _search_tool()

    def _papers_catalog(self, paper_ids: list[str] | None) -> str:
        """Formats the papers in `paper_ids` (or every manifest paper if `None`) as
        "paper_id (Title; rarest-first tag hints)", comma-separated — shared by `_system`
        (the ReAct system prompt) and `classify_mode` (the Auto-mode classifier), so both
        see the same paper identities rather than duplicating this sorting logic."""
        # Scope the injected catalog to the active filter — otherwise the model can
        # read every paper off the prompt prefix and answer catalog questions
        # ("which models?") without searching, bypassing the filter entirely.
        recs = self.manifest.papers()
        if paper_ids is not None:
            allowed = set(paper_ids)
            recs = [p for p in recs if p["paper_id"] in allowed]
        # Rarest-first tags per paper as a routing hint — e.g. distinguishing two
        # near-duplicate titles in the same paper family — since sorting by raw tag
        # order would surface whatever generic tags nearly every paper in the
        # library shares.
        tag_counts: dict[str, int] = {}
        for p in recs:
            for t in p.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        def topical(p: dict) -> str:
            tags = sorted(p.get("tags", []), key=lambda t: tag_counts[t])[:3]
            return f"; {', '.join(tags)}" if tags else ""

        return ", ".join(f"{p['paper_id']} ({p['title']}{topical(p)})" for p in recs) or "(none)"

    def _system(self, paper_ids: list[str] | None, search_budget: int) -> str:
        papers = self._papers_catalog(paper_ids)
        # "papers in scope" is the one phrase the prompt uses everywhere it needs to talk
        # about what's searchable — this note is the only place that defines it, so a
        # branch can say "papers in scope" and be correct whether or not a filter is
        # active, instead of every branch separately hardcoding "the library".
        if paper_ids is not None:
            note = (
                "The user restricted this conversation's scope to the papers listed "
                "below — those, and only those, are 'papers in scope' throughout this "
                "prompt; treat them as the only papers that exist. Do not mention, "
                "list, or search any other paper."
            )
        else:
            note = "No paper/tag filter is active; 'papers in scope' means the entire library."
        return SYSTEM_PROMPT.format(filter_note=note, papers=papers, search_budget=search_budget)

    def _resolve_paper_ids(
        self, tags: list[str], papers: list[str], fallback_to_manifest: bool = False
    ) -> list[str] | None:
        """Intersects the tag filter and paper filter into one scope: `None` means no
        filter is active. `fallback_to_manifest` turns that `None` into an explicit list
        of every paper in the manifest — `run`'s `per_paper` and `compare` both need a
        concrete list to loop/scope over (`Searcher` has no manifest to fall back to
        "every paper" itself), while a normal Ask turn is fine describing an inactive
        filter as "the entire library" with no explicit list."""
        tag_ids = self.manifest.paper_ids_for_tags(tags) if tags else None
        selected = list(papers) if papers else None
        if tag_ids is None:
            paper_ids = selected
        elif selected is None:
            paper_ids = tag_ids
        else:
            wanted = set(selected)
            paper_ids = [p for p in tag_ids if p in wanted]
        if fallback_to_manifest and paper_ids is None:
            paper_ids = [p["paper_id"] for p in self.manifest.papers()]
        return paper_ids

    def classify_mode(
        self, messages: list[dict], tags: list[str], papers: list[str]
    ) -> tuple[Literal["ask", "compare"], int]:
        """Auto mode's pre-flight decision: returns (mode, scope_size).

        `scope_size` is the resolved paper count (same resolution `compare()` uses) — the
        caller needs it either way, to size the large-Compare confirm dialog.

        Below 2 resolved papers, Compare is structurally impossible (`compare()` raises
        otherwise) — skip the LLM call entirely and return "ask", no question worth a
        completion.

        Otherwise asks a fresh tagging-tier client (not `self.client`, the full chat model —
        this is a cheap, best-effort classification, exactly what the tagging tier is for,
        same as `generate_name`) to pick one word. The prompt includes the same papers
        catalog `_system` injects into the ReAct system prompt (`_papers_catalog`), so the
        classifier can tell "what's the model size?" needs Compare from the scope alone
        (several distinct model papers) even when the question never says "each" or
        "compare" — not just from the question's own wording. On any parse or backend
        failure, defaults to "ask" — the always-safe, always-cheaper path.
        """
        paper_ids = self._resolve_paper_ids(tags, papers, fallback_to_manifest=True)
        scope_size = len(paper_ids) if paper_ids else 0
        if scope_size < 2:
            return "ask", scope_size
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        system = CLASSIFY_SYSTEM_PROMPT.format(papers=self._papers_catalog(paper_ids))
        mode: Literal["ask", "compare"] = "ask"
        try:
            # Same 256-token budget generate_name uses: reasoning models spend tokens on a
            # hidden reasoning channel before the visible answer and return empty content if
            # capped too low — and since empty/malformed output already parses to "ask"
            # below, under-budgeting this call wouldn't just misclassify occasionally, it
            # would make Auto mode always resolve to Ask on a reasoning-model backend.
            raw = build_llm(self.cfg.llm.tagging).complete(
                system=system, user=transcript, max_tokens=256
            )
            # rstrip punctuation, not substring-match: an otherwise-compliant "COMPARE."
            # (trailing period is a common habit even from models that mostly follow
            # "reply with exactly one word") shouldn't fall through to "ask" just because
            # of trailing punctuation — but a sentence merely containing "compare" still
            # must not match, which is why this stays an exact-token comparison.
            first_word = (
                raw.strip().split()[0].upper().rstrip(string.punctuation) if raw.strip() else ""
            )
            if first_word == "COMPARE":
                mode = "compare"
        except Exception:
            mode = "ask"
        return mode, scope_size

    def run(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_trace=None,
        ref_start: int = 0,
        per_paper: bool = False,
        stop_check: Callable[[], bool] | None = None,
    ) -> tuple[str, list[dict], Usage]:
        """Returns (answer_text, citations[], usage).

        `tags` and `papers` are two optional scoping filters the user can set: a
        tag filter (papers carrying any of the tags) and an explicit paper picker.
        Whichever are active intersect to form the searchable set.

        `on_trace(entry)` fires for each reasoning/tool step, where entry.type is
        "thought" (model reasoning), "action" (a search call), or "observation"
        (its results) — enough to render the full Thought→Action→Observation trace.

        `ref_start` offsets the r1, r2, ... ref numbering — the caller passes the
        count of refs already used earlier in the same chat, so a follow-up
        question continues the numbering (r4, r5, ...) instead of restarting at
        r1 and colliding with refs already shown for a different paper.

        `per_paper` (see `Searcher.search`) applies uniformly to every `search_papers`
        call made during this turn's ReAct loop — the user toggles it per message, not
        the model per call.

        `stop_check`, if given, is polled by the LLM backend between/within streaming
        rounds; once it returns True the backend returns whatever text has streamed so
        far instead of continuing — the caller (chat_turn.run_turn) still persists that
        partial text as a normal answer, same as if the model had finished on its own.
        """
        paper_ids = self._resolve_paper_ids(tags, papers, fallback_to_manifest=per_paper)
        registry: dict[str, dict] = {}
        bodies: dict[str, str] = {}
        counter = {"n": ref_start}
        # max_rounds counts ReAct rounds, not searches, and the harness doesn't reserve
        # a final round to answer in — if the model spends every round on tool calls it
        # returns with no real answer. Budget one fewer than the true round cap so the
        # model treats the last round as the one it must spend answering, not searching.
        search_budget = max(self.cfg.retrieval.max_rounds - 1, 1)
        searches_used = {"n": 0}

        def trace(entry: dict):
            if on_trace:
                on_trace(entry)

        def execute(name: str, args: dict) -> str:
            if name != "search_papers":
                return f"Unknown tool: {name}"
            query = (args.get("query") or "").strip()
            if not query:
                return "Error: empty query."
            searches_used["n"] += 1
            # Told to the model in the tool result (not just the system prompt) so it
            # doesn't have to self-count tool calls across a long interleaved trace —
            # a read, not a memory task.
            remaining = max(search_budget - searches_used["n"], 0)
            trace(
                {
                    "type": "action",
                    "query": query,
                    "paper": args.get("paper") or None,
                    "per_paper": per_paper,
                }
            )
            outcome = self.searcher.search(
                query,
                min_k=self.cfg.retrieval.min_k,
                max_k=self.cfg.retrieval.max_k,
                candidates=max(self.cfg.retrieval.candidates, self.cfg.retrieval.max_k * 4),
                paper=args.get("paper") or None,
                paper_ids=paper_ids,
                rerank=self.cfg.reranker.enabled,
                per_paper=per_paper,
                elbow_enabled=self.cfg.retrieval.elbow_enabled,
                elbow_mad_multiplier=self.cfg.retrieval.elbow_mad_multiplier,
                elbow_prominence=self.cfg.retrieval.elbow_prominence,
            )
            results = outcome.results
            if not results:
                trace({"type": "observation", "text": "(no results)"})
                return (
                    "No results for that query (within the active filter). "
                    f"[{remaining} searches remaining this turn]"
                )
            blocks = []
            for r in results:
                counter["n"] += 1
                ref = f"r{counter['n']}"
                rec = self.manifest.get(r.paper_id)
                registry[ref] = {
                    "ref": ref,
                    "paper_id": r.paper_id,
                    "title": rec["title"] if rec else r.paper_id,
                    "arxiv_id": rec.get("arxiv_id") if rec else None,
                    "breadcrumb": r.breadcrumb,
                    "section_title": r.section_title,
                    "section_number": r.section_number,
                    "source": r.source,
                    "snippet": r.body[:500],
                    "body": r.body,
                }
                # Full passage, kept for the faithfulness check (split into sentences there).
                bodies[ref] = r.body
                # Full passage — exactly what the model receives, shown verbatim in the trace.
                blocks.append(f'[{ref}] paper={r.paper_id}  section="{r.breadcrumb}"\n{r.body}')
            observation = "\n\n".join(blocks)
            if outcome.cutoff_reason != "no_elbow":
                observation = (
                    f"[{len(results)} of up to {self.cfg.retrieval.max_k} returned "
                    f"— {outcome.cutoff_reason}]\n\n"
                ) + observation
            observation = f"[{remaining} searches remaining this turn]\n\n" + observation
            trace({"type": "observation", "text": observation})
            return observation

        text, usage = self.client.run_tools(
            system=self._system(paper_ids, search_budget),
            messages=[dict(m) for m in messages],
            tools=[self.search_tool],
            execute=execute,
            on_text=on_text,
            on_reasoning=lambda t: trace({"type": "thought", "text": t}),
            max_rounds=self.cfg.retrieval.max_rounds,
            stop_check=stop_check,
        )
        if self.cfg.faithfulness.enabled and registry:
            self._check_faithfulness(text, registry, bodies)
        return text, list(registry.values()), usage

    def compare(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_row,
        on_trace=None,
        ref_start: int = 0,
        stop_check: Callable[[], bool] | None = None,
    ) -> tuple[str, list[dict], list[dict], Usage]:
        """Returns (text, compare_results, citations, usage).

        Guarantees every paper in scope gets its own dedicated search+answer — runs `run`
        once per paper (reusing it unmodified, scoped via `papers=[paper_id]`), then feeds
        every per-paper answer back into the model once more to synthesize one final,
        genuinely comparative `text` that reuses the same [rN] markers the sub-runs already
        produced (see `SYNTHESIS_SYSTEM_PROMPT`). `citations` is the union of every
        sub-run's citations, structurally identical to what `run` returns for a normal
        turn. `on_row(row)` fires once per completed paper with that paper's own
        `{paper_id, title, arxiv_id, text, citations, trace}` — the carousel's drill-down
        data; `compare_results` (the second return value) is the accumulated list of every
        row, in scope order.

        `ref_start`/`stop_check` mean the same as in `run`. Raises `InsufficientScopeError`
        (a `ValueError` subclass) if fewer than 2 papers resolve in scope — the UI already
        disables Compare below 2, but this is a contract the backend enforces on its own
        rather than trusting that alone.
        """
        paper_ids = self._resolve_paper_ids(tags, papers, fallback_to_manifest=True)
        if paper_ids is None or len(paper_ids) < 2:
            raise InsufficientScopeError("Compare mode needs at least 2 papers in scope.")

        rows: list[dict] = []
        citations: list[dict] = []
        running_ref = ref_start
        total_in = 0
        total_out = 0
        any_usage = False

        for paper_id in paper_ids:
            if stop_check and stop_check():
                break
            rec = self.manifest.get(paper_id) or {}
            row_trace: list[dict] = []
            try:
                row_text, row_citations, row_usage = self.run(
                    messages,
                    tags=[],
                    papers=[paper_id],
                    # This sub-run's own tokens are never streamed to the user — only the
                    # final synthesized answer is; `row_text` (the return value) is what
                    # feeds both the carousel and the synthesis prompt below.
                    on_text=lambda _t: None,
                    on_trace=row_trace.append,
                    ref_start=running_ref,
                    per_paper=False,
                    stop_check=stop_check,
                )
            except Exception as e:
                # Don't let one paper's failure drop the whole comparison — a placeholder
                # row still participates in synthesis, so the final answer can say "no
                # answer available for <paper>" instead of silently omitting it. Logged,
                # not swallowed silently — this also catches a genuine bug in `run()`
                # (not just an expected search/LLM failure), which would otherwise be
                # indistinguishable from a normal placeholder row.
                print(f"[compare] sub-run failed for {paper_id}: {e}")
                row_text = "_(search failed for this paper)_"
                row_citations = []
                row_usage = Usage(None, None)
            running_ref += len(row_citations)
            if row_usage.input_tokens is not None:
                total_in += row_usage.input_tokens
                any_usage = True
            if row_usage.output_tokens is not None:
                total_out += row_usage.output_tokens
                any_usage = True
            row = {
                "paper_id": paper_id,
                "title": rec.get("title", paper_id),
                "arxiv_id": rec.get("arxiv_id"),
                "text": row_text,
                "citations": row_citations,
                "trace": row_trace,
            }
            rows.append(row)
            citations.extend(row_citations)
            on_row(row)

        if stop_check and stop_check():
            text, flat_citations = _flatten_compare_rows(rows)
            return text, rows, flat_citations, Usage(None, None)

        # Not bracketed like `[r['paper_id']]` — SYNTHESIS_SYSTEM_PROMPT spends a paragraph
        # telling the model `[rN]` is reserved for citation markers; a per-paper header in
        # the same bracket syntax is a needless, avoidable way to confuse that instruction.
        per_paper_answers = "\n".join(
            f"{r['title']} ({r['paper_id']}):\n{r['text']}\n" for r in rows
        )
        synth_system = SYNTHESIS_SYSTEM_PROMPT.format(per_paper_answers=per_paper_answers)

        def synth_trace(entry: dict) -> None:
            if on_trace:
                on_trace(entry)

        # Defensive fallback, not the expected path (the prompt tells the model not to
        # search) — but if it fires anyway, new citations must continue this turn's ref
        # numbering rather than colliding with any row's [rN], so `counter` starts where
        # the per-paper loop left off.
        counter = {"n": running_ref}

        def synth_execute(name: str, args: dict) -> str:
            if name != "search_papers":
                return f"Unknown tool: {name}"
            query = (args.get("query") or "").strip()
            if not query:
                return "Error: empty query."
            synth_trace({"type": "action", "query": query, "paper": args.get("paper") or None})
            outcome = self.searcher.search(
                query,
                min_k=self.cfg.retrieval.min_k,
                max_k=self.cfg.retrieval.max_k,
                candidates=max(self.cfg.retrieval.candidates, self.cfg.retrieval.max_k * 4),
                paper=args.get("paper") or None,
                paper_ids=paper_ids,
                rerank=self.cfg.reranker.enabled,
                per_paper=False,
                elbow_enabled=self.cfg.retrieval.elbow_enabled,
                elbow_mad_multiplier=self.cfg.retrieval.elbow_mad_multiplier,
                elbow_prominence=self.cfg.retrieval.elbow_prominence,
            )
            results = outcome.results
            if not results:
                synth_trace({"type": "observation", "text": "(no results)"})
                return "No results for that query (within the active filter)."
            blocks = []
            for r in results:
                counter["n"] += 1
                ref = f"r{counter['n']}"
                rec2 = self.manifest.get(r.paper_id)
                citation = {
                    "ref": ref,
                    "paper_id": r.paper_id,
                    "title": rec2["title"] if rec2 else r.paper_id,
                    "arxiv_id": rec2.get("arxiv_id") if rec2 else None,
                    "breadcrumb": r.breadcrumb,
                    "section_title": r.section_title,
                    "section_number": r.section_number,
                    "source": r.source,
                    "snippet": r.body[:500],
                    "body": r.body,
                }
                citations.append(citation)
                blocks.append(f'[{ref}] paper={r.paper_id}  section="{r.breadcrumb}"\n{r.body}')
            observation = "\n\n".join(blocks)
            synth_trace({"type": "observation", "text": observation})
            return observation

        # Full conversation history, same as every per-paper sub-run above got — a
        # multi-turn follow-up ("and their inference cost?") needs the same context to
        # resolve pronouns/references that each sub-run already had. This is also the
        # largest-context call in the whole feature (history + every per-paper answer), so
        # on a context-length failure, retry once with just the current question rather
        # than failing the whole comparison after every sub-run above already succeeded.
        # Context-length errors are a request-validation failure that providers surface
        # before any streaming starts, so this assumes `on_text` hasn't emitted partial
        # text from the failed attempt by the time the retry runs.
        try:
            final_text, synth_usage = self.client.run_tools(
                system=synth_system,
                messages=messages,
                tools=[self.search_tool],
                execute=synth_execute,
                on_text=on_text,
                on_reasoning=lambda t: synth_trace({"type": "thought", "text": t}),
                max_rounds=2,
                stop_check=stop_check,
            )
        except Exception:
            final_text, synth_usage = self.client.run_tools(
                system=synth_system,
                messages=[messages[-1]],
                tools=[self.search_tool],
                execute=synth_execute,
                on_text=on_text,
                on_reasoning=lambda t: synth_trace({"type": "thought", "text": t}),
                max_rounds=2,
                stop_check=stop_check,
            )
            # Softened wording deliberately: the except above catches any failure on the
            # first attempt, not specifically a context-length error (no cross-SDK way to
            # detect that specifically without importing every provider's SDK) — naming
            # "too long" as the cause would assert a diagnosis this code hasn't verified.
            final_text = (
                "_(retried with just your current question after the first attempt "
                "failed)_\n\n" + final_text
            )

        if synth_usage.input_tokens is not None:
            total_in += synth_usage.input_tokens
            any_usage = True
        if synth_usage.output_tokens is not None:
            total_out += synth_usage.output_tokens
            any_usage = True
        usage = Usage(total_in if any_usage else None, total_out if any_usage else None)

        if self.cfg.faithfulness.enabled and citations:
            registry = {c["ref"]: c for c in citations}
            bodies = {c["ref"]: c["body"] for c in citations}
            try:
                self._check_faithfulness(final_text, registry, bodies)
            except Exception as e:
                # Don't lose an already-completed N-paper comparison (N sub-runs + a
                # synthesis pass) to a crash in this last, purely-annotative step — the
                # synthesized answer and every row are already good; just ship them
                # without the faithfulness flags. `run()` has this same fragility on its
                # own single-call faithfulness check; the blast radius is much bigger
                # here, which is what makes catching it worth the extra two lines.
                print(f"[compare] faithfulness check failed, shipping without it: {e}")

        return final_text, rows, citations, usage

    def _check_faithfulness(
        self, text: str, registry: dict[str, dict], bodies: dict[str, str]
    ) -> None:
        """Adds a `faithfulness` list to every ref the answer actually cites via an
        [rN] marker: one {sentence, label, score} entry per citing sentence (not
        collapsed to a single verdict — a ref cited several times keeps one entry
        per citation, so "2 of 3 claims entailed" stays visible instead of being
        flattened into one worst-of-3 label). Each entry is scored sentence-vs-
        sentence against the passage (SummaC-style max-pooling over the passage's
        own sentences), not against the whole passage body. Refs retrieved but
        never cited are left unchecked — there's no hypothesis span to test.

        Cost is O(refs x citing_sentences x passage_sentences) NLI pairs, batched
        into one check_batch call — bounded in practice by retrieval.max_rounds /
        retrieval.max_k (how many refs a run can accumulate) and chunking.max_tokens
        (how many sentences one passage has), but there's no explicit cap here."""
        spans = attribute_refs(text, set(registry))
        if not spans:
            return
        pair_owners: list[tuple[str, str]] = []  # (ref, citing sentence) per pair below
        pairs: list[tuple[str, str]] = []
        passage_sentences: dict[str, list[str]] = {}
        for ref, sentences in spans.items():
            if ref not in passage_sentences:
                passage_sentences[ref] = split_sentences(bodies[ref])
            for sentence in sentences:
                for passage_sentence in passage_sentences[ref]:
                    pair_owners.append((ref, sentence))
                    pairs.append((passage_sentence, sentence))
        if not pairs:
            return
        verdicts = self.faithfulness.check_batch(pairs)

        grouped: dict[tuple[str, str], list] = {}
        for (ref, sentence), verdict in zip(pair_owners, verdicts, strict=True):
            grouped.setdefault((ref, sentence), []).append(verdict)

        by_ref: dict[str, list[dict]] = {}
        for (ref, sentence), group in grouped.items():
            v = best_support(group)
            by_ref.setdefault(ref, []).append(
                {"sentence": sentence, "label": v.label, "score": v.score}
            )
        for ref, claims in by_ref.items():
            registry[ref]["faithfulness"] = claims
            if any(c["label"] == "contradiction" for c in claims):
                labels = [c["label"] for c in claims]
                print(f"[faithfulness] {ref} ({registry[ref]['paper_id']}): {labels}")
