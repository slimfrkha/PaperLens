"""Agentic RAG chat: a ReAct loop built on the model's native tool calling.

The model reasons, then (only when needed) calls the `search_papers` tool — its
tool call is the parseable Action, the returned passages are the Observation, and
the loop repeats until it answers. Small talk (e.g. "hi", "thanks") is answered
directly with no search. Each search is surfaced as a step so the UI can show the
Thought → Action → Observation trace.
"""

from __future__ import annotations

from rag.config import Config
from rag.llm import build_llm
from rag.manifest import Manifest
from rag.search import Searcher

SEARCH_TOOL = {
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
            "query": {"type": "string", "description": "A focused natural-language search query."},
            "paper": {"type": "string", "description": "Optional paper_id to restrict the search."},
            "top_k": {"type": "integer", "description": "How many passages to return (default 5)."},
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
    def __init__(self, cfg: Config, searcher: Searcher, manifest: Manifest):
        self.cfg = cfg
        self.searcher = searcher
        self.manifest = manifest
        self.client = build_llm(cfg.llm.chat)

    def _system(self, paper_ids: list[str] | None) -> str:
        papers = (
            ", ".join(f"{p['paper_id']} ({p['title']})" for p in self.manifest.papers())
            or "(none yet)"
        )
        if paper_ids is not None:
            note = (
                "The user restricted this conversation to these papers: "
                f"{', '.join(paper_ids) or '(none — no paper matches the active tags)'}. "
                "Only these are searchable."
            )
        else:
            note = "No paper/tag filter is active; all papers are searchable."
        return SYSTEM_PROMPT.format(filter_note=note, papers=papers)

    def run(self, messages, tags, paper, on_text, on_trace=None):
        """Returns (answer_text, citations[]).

        `on_trace(entry)` fires for each reasoning/tool step, where entry.type is
        "thought" (model reasoning), "action" (a search call), or "observation"
        (its results) — enough to render the full Thought→Action→Observation trace.
        """
        paper_ids = self.manifest.paper_ids_for_tags(tags) if tags else None
        registry: dict[str, dict] = {}
        counter = {"n": 0}

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
            results = self.searcher.search(
                query,
                k=int(args.get("top_k") or 5),
                candidates=max(20, int(args.get("top_k") or 5) * 4),
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
                    "breadcrumb": r.breadcrumb,
                    "section_title": r.section_title,
                    "snippet": r.body[:500],
                }
                # Full passage — exactly what the model receives, shown verbatim in the trace.
                blocks.append(f'[{ref}] paper={r.paper_id}  section="{r.breadcrumb}"\n{r.body}')
            observation = "\n\n".join(blocks)
            trace({"type": "observation", "text": observation})
            return observation

        text = self.client.run_tools(
            system=self._system(paper_ids),
            messages=[dict(m) for m in messages],
            tools=[SEARCH_TOOL],
            execute=execute,
            on_text=on_text,
            on_reasoning=lambda t: trace({"type": "thought", "text": t}),
        )
        return text, list(registry.values())
