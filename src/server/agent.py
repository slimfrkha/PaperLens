"""Agentic RAG chat: a ReAct loop built on the model's native tool calling.

The model reasons, then (only when needed) calls the `search_papers` tool — its
tool call is the parseable Action, the returned passages are the Observation, and
the loop repeats until it answers. Small talk (e.g. "hi", "thanks") is answered
directly with no search. Each search is surfaced as a step so the UI can show the
Thought → Action → Observation trace.
"""

from __future__ import annotations

from collections.abc import Callable

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

    def _system(self, paper_ids: list[str] | None, search_budget: int) -> str:
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

        papers = ", ".join(f"{p['paper_id']} ({p['title']}{topical(p)})" for p in recs) or "(none)"
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
        tag_ids = self.manifest.paper_ids_for_tags(tags) if tags else None
        selected = list(papers) if papers else None
        if tag_ids is None:
            paper_ids = selected
        elif selected is None:
            paper_ids = tag_ids
        else:
            wanted = set(selected)
            paper_ids = [p for p in tag_ids if p in wanted]
        # per_paper needs a concrete paper list to loop over — Searcher has no manifest to
        # fall back to "every paper" itself, so this is the one place that fallback happens.
        if per_paper and paper_ids is None:
            paper_ids = [p["paper_id"] for p in self.manifest.papers()]
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
