"""Agentic RAG chat: a ReAct loop built on the model's native tool calling.

The model reasons, then (only when needed) calls the `search_papers` tool — its
tool call is the parseable Action, the returned passages are the Observation, and
the loop repeats until it answers. Small talk (e.g. "hi", "thanks") is answered
directly with no search. Each search is surfaced as a step so the UI can show the
Thought → Action → Observation trace.
"""

from __future__ import annotations

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


def _search_tool(default_k: int) -> dict:
    return {
        "name": "search_papers",
        "description": (
            "Search the library of arXiv model technical reports and return the most "
            "relevant passages. Call it once per focused sub-question (call it several "
            "times to decompose a multi-part question). Each result carries a `ref` "
            "(r1, r2, ...) you must use to cite it. Do NOT call this for greetings or "
            "small talk."
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
                    "description": "Optional paper_id to restrict the search.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"How many passages to return (default {default_k}).",
                },
            },
            "required": ["query"],
        },
    }


SYSTEM_PROMPT = """You are a research assistant for a library of arXiv model \
technical reports (LLM papers).

First decide whether answering actually needs the papers:
- Greetings, thanks, or simple conversational messages: answer directly, do NOT \
call any tool.
- Questions about the papers or the concepts in them: use the `search_papers` \
tool. Decompose a multi-part question and call the tool once per focused \
sub-question (you may call it several times). Read the returned passages, and if \
you still need more, search again — keep going until you have enough, then answer.

Ground every factual claim in retrieved passages and cite them inline with the \
[rN] markers exactly as they appear in the results (e.g. "MLA shrinks the KV cache \
[r2]."). Only cite refs you actually received. Be concise and technical.

{filter_note}
Papers in the library: {papers}"""


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
        self.search_tool = _search_tool(cfg.retrieval.k)

    def _system(self, paper_ids: list[str] | None) -> str:
        # Scope the injected catalog to the active filter — otherwise the model can
        # read every paper off the prompt prefix and answer catalog questions
        # ("which models?") without searching, bypassing the filter entirely.
        recs = self.manifest.papers()
        if paper_ids is not None:
            allowed = set(paper_ids)
            recs = [p for p in recs if p["paper_id"] in allowed]
        papers = ", ".join(f"{p['paper_id']} ({p['title']})" for p in recs) or "(none)"
        if paper_ids is not None:
            note = (
                "The user restricted this conversation to the papers listed below — "
                "treat those as the only papers that exist. Do not mention, list, or "
                "search any other paper."
            )
        else:
            note = "No paper/tag filter is active; all papers are searchable."
        return SYSTEM_PROMPT.format(filter_note=note, papers=papers)

    def run(
        self, messages, tags, papers, on_text, on_trace=None, ref_start: int = 0
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
        registry: dict[str, dict] = {}
        bodies: dict[str, str] = {}
        counter = {"n": ref_start}

        def trace(entry: dict):
            if on_trace:
                on_trace(entry)

        def execute(name: str, args: dict) -> str:
            if name != "search_papers":
                return f"Unknown tool: {name}"
            query = (args.get("query") or "").strip()
            if not query:
                return "Error: empty query."
            trace({"type": "action", "query": query, "paper": args.get("paper") or None})
            k = int(args.get("top_k") or self.cfg.retrieval.k)
            results = self.searcher.search(
                query,
                k=k,
                candidates=max(self.cfg.retrieval.candidates, k * 4),
                paper=args.get("paper") or None,
                paper_ids=paper_ids,
                rerank=self.cfg.reranker.enabled,
            )
            if not results:
                trace({"type": "observation", "text": "(no results)"})
                return "No results for that query (within the active filter)."
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
            trace({"type": "observation", "text": observation})
            return observation

        text, usage = self.client.run_tools(
            system=self._system(paper_ids),
            messages=[dict(m) for m in messages],
            tools=[self.search_tool],
            execute=execute,
            on_text=on_text,
            on_reasoning=lambda t: trace({"type": "thought", "text": t}),
            max_rounds=self.cfg.retrieval.max_rounds,
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
        retrieval.k (how many refs a run can accumulate) and chunking.max_tokens
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
