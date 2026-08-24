"""Agentic-RAG chat loop with a scripted LLM over a real retrieval seam."""

from __future__ import annotations

import pytest

from rag.config import HFFaithfulnessCfg
from rag.llm import Usage
from rag.manifest import Manifest
from server.agent import CLASSIFY_SYSTEM_PROMPT, ChatAgent


@pytest.fixture
def make_agent(make_searcher, seed_chunks):
    """Factory: a ChatAgent wired to a seeded searcher, manifest, and FakeLLM."""

    def _make(llm, faithfulness=None):
        docs = [
            seed_chunks("paper-a", "Attention", "multi head latent attention kv cache"),
            seed_chunks("paper-b", "Training", "reinforcement learning recipe"),
        ]
        ctx = make_searcher(docs)
        if faithfulness is not None:
            ctx.cfg.faithfulness = HFFaithfulnessCfg(enabled=True)
        manifest = Manifest(ctx.cfg.paths.rag_db)
        manifest.upsert(
            {"paper_id": "paper-a", "title": "Paper A", "tags": ["moe"], "arxiv_id": "2412.19437"}
        )
        manifest.upsert({"paper_id": "paper-b", "title": "Paper B", "tags": ["rl"]})
        agent = ChatAgent(ctx.cfg, ctx.searcher, manifest, client=llm, faithfulness=faithfulness)
        return agent

    return _make


def test_search_call_builds_citations_and_trace(make_agent, fake_llm):
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
        reasoning="I should search.",
    )
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1  # pin a single deterministic result
    agent.cfg.retrieval.max_k = 1

    trace = []
    texts = []
    text, citations, _usage = agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=texts.append,
        on_trace=trace.append,
    )

    assert text == llm.answer
    assert "".join(texts) == llm.answer
    assert len(citations) == 1
    assert citations[0]["ref"] == "r1"
    assert citations[0]["paper_id"] == "paper-a"
    assert citations[0]["title"] == "Paper A"
    assert citations[0]["arxiv_id"] == "2412.19437"
    assert citations[0]["source"] == "dense"
    assert citations[0]["section_number"] == "1"
    assert citations[0]["body"] == "multi head latent attention kv cache"

    kinds = [e["type"] for e in trace]
    assert kinds == ["thought", "action", "observation"]


def test_observation_trace_carries_a_cutoff_diagnostic_when_not_no_elbow(make_agent, fake_llm):
    # reranker.enabled=False by default (make_config) -> cutoff_reason is always "no_rerank",
    # never "no_elbow" -> the diagnostic line must be prepended to the observation text.
    llm = fake_llm(
        answer="See [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 1

    trace = []
    agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_trace=trace.append,
    )

    observation = next(e for e in trace if e["type"] == "observation")
    # Remaining-searches countdown prefixes the observation ahead of the cutoff
    # diagnostic, which itself still comes ahead of the passage blocks.
    assert observation["text"].startswith("[")
    assert "[1 of up to 1 returned — no_rerank]" in observation["text"]


def test_run_returns_usage_from_the_llm_backend(make_agent, fake_llm):
    llm = fake_llm(answer="Hello!", tool_calls=[], usage=Usage(input_tokens=123, output_tokens=45))
    agent = make_agent(llm)

    _text, _citations, usage = agent.run(
        [{"role": "user", "content": "hi"}], tags=[], papers=[], on_text=lambda _t: None
    )
    assert usage.input_tokens == 123
    assert usage.output_tokens == 45


def test_small_talk_answers_without_searching(make_agent, fake_llm):
    llm = fake_llm(answer="Hello!", tool_calls=[])
    agent = make_agent(llm)

    text, citations, _usage = agent.run(
        [{"role": "user", "content": "hi"}], tags=[], papers=[], on_text=lambda _t: None
    )
    assert text == "Hello!"
    assert citations == []
    assert llm.executed == []


def test_ref_start_continues_numbering_across_turns(make_agent, fake_llm):
    # A second /api/chat call in the same conversation must not restart ref
    # numbering at r1 — main.py passes the count of already-used refs as ref_start.
    llm = fake_llm(
        answer="See [r2].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1  # pin a single deterministic result
    agent.cfg.retrieval.max_k = 1

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "follow-up question"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        ref_start=1,
    )
    assert len(citations) == 1
    assert citations[0]["ref"] == "r2"


def test_tag_filter_scopes_search_to_matching_papers(make_agent, fake_llm):
    # tags=["rl"] -> only paper-b; a query is restricted to that paper.
    llm = fake_llm(
        answer="See [r1].",
        tool_calls=[("search_papers", {"query": "reinforcement learning"})],
    )
    agent = make_agent(llm)

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "training recipe?"}],
        tags=["rl"],
        papers=[],
        on_text=lambda _t: None,
    )
    assert citations
    assert {c["paper_id"] for c in citations} == {"paper-b"}


def test_paper_filter_scopes_search_to_selected_papers(make_agent, fake_llm):
    # papers=["paper-a"] -> search is restricted to that paper regardless of query.
    llm = fake_llm(
        answer="See [r1].",
        tool_calls=[("search_papers", {"query": "reinforcement learning recipe"})],
    )
    agent = make_agent(llm)

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "training?"}],
        tags=[],
        papers=["paper-a"],
        on_text=lambda _t: None,
    )
    assert citations
    assert {c["paper_id"] for c in citations} == {"paper-a"}


def test_tag_and_paper_filters_intersect(make_agent, fake_llm):
    # tags -> paper-b, papers -> paper-a: disjoint, so nothing is searchable.
    llm = fake_llm(answer="none", tool_calls=[("search_papers", {"query": "anything"})])
    agent = make_agent(llm)

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "x"}],
        tags=["rl"],
        papers=["paper-a"],
        on_text=lambda _t: None,
    )
    assert citations == []


def test_filter_scopes_the_paper_catalog_in_the_prompt(make_agent, fake_llm):
    # A filter must scope the paper list injected into the system prompt, or the
    # model can list the whole catalog from its prefix without ever searching.
    # Assert through the public seam — the system prompt the LLM actually receives —
    # so renaming the private _system helper doesn't break the test.
    scoped_llm = fake_llm(answer="x", tool_calls=[])
    make_agent(scoped_llm).run(
        [{"role": "user", "content": "x"}], tags=[], papers=["paper-b"], on_text=lambda _t: None
    )
    scoped = scoped_llm.run_tools_calls[0]["system"]
    assert "Paper B" in scoped
    assert "Paper A" not in scoped

    full_llm = fake_llm(answer="x", tool_calls=[])
    make_agent(full_llm).run(
        [{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None
    )
    full = full_llm.run_tools_calls[0]["system"]
    assert "Paper B" in full and "Paper A" in full


def test_empty_query_is_rejected(make_agent, fake_llm):
    llm = fake_llm(answer="done", tool_calls=[("search_papers", {"query": "   "})])
    agent = make_agent(llm)
    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None
    )
    # Blank query -> no search performed, no citations.
    assert citations == []


def test_search_uses_configured_min_max_k(make_agent, fake_llm):
    # The model no longer requests a per-call count (top_k was removed) — every
    # search_papers call gets the config's own min_k/max_k, unconditionally.
    llm = fake_llm(answer="ok", tool_calls=[("search_papers", {"query": "latent attention"})])
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 2
    agent.cfg.retrieval.candidates = 9

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run([{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None)

    assert calls[0]["min_k"] == 1
    assert calls[0]["max_k"] == 2
    assert calls[0]["candidates"] == 9


def test_candidate_pool_scales_with_configured_max_k(make_agent, fake_llm):
    # A large max_k must still leave the reranker something to discard, otherwise
    # it just reorders exactly what dense recall returned.
    llm = fake_llm(answer="ok", tool_calls=[("search_papers", {"query": "mla"})])
    agent = make_agent(llm)
    agent.cfg.retrieval.max_k = 50
    agent.cfg.retrieval.candidates = 20  # below max_k * 4 -> the scaling must win

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run([{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None)

    assert calls[0]["max_k"] == 50
    assert calls[0]["candidates"] == 200


def test_max_rounds_flows_from_retrieval_config(make_agent, fake_llm):
    llm = fake_llm(answer="ok", tool_calls=[])
    agent = make_agent(llm)
    agent.cfg.retrieval.max_rounds = 3
    agent.run([{"role": "user", "content": "hi"}], tags=[], papers=[], on_text=lambda _t: None)
    assert llm.run_tools_calls[0]["max_rounds"] == 3


def test_per_paper_trace_records_the_flag(make_agent, fake_llm):
    llm = fake_llm(
        answer="See [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm)

    trace = []
    agent.run(
        [{"role": "user", "content": "x"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_trace=trace.append,
        per_paper=True,
    )

    action = next(e for e in trace if e["type"] == "action")
    assert action["per_paper"] is True


def test_per_paper_trace_defaults_to_false(make_agent, fake_llm):
    llm = fake_llm(
        answer="See [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm)

    trace = []
    agent.run(
        [{"role": "user", "content": "x"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_trace=trace.append,
    )

    action = next(e for e in trace if e["type"] == "action")
    assert action["per_paper"] is False


def test_per_paper_falls_back_to_every_manifest_paper_when_no_filter(make_agent, fake_llm):
    llm = fake_llm(answer="ok", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm)

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run(
        [{"role": "user", "content": "x"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        per_paper=True,
    )

    assert sorted(calls[0]["paper_ids"]) == ["paper-a", "paper-b"]
    assert calls[0]["per_paper"] is True


def test_per_paper_does_not_override_an_already_resolved_filter(make_agent, fake_llm):
    llm = fake_llm(answer="ok", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm)

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run(
        [{"role": "user", "content": "x"}],
        tags=[],
        papers=["paper-a"],
        on_text=lambda _t: None,
        per_paper=True,
    )

    assert calls[0]["paper_ids"] == ["paper-a"]


def test_per_paper_false_leaves_paper_ids_none_when_no_filter(make_agent, fake_llm):
    llm = fake_llm(answer="ok", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm)

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run([{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None)

    assert calls[0]["paper_ids"] is None
    assert calls[0]["per_paper"] is False


def test_faithfulness_disabled_by_default_no_verdict_and_checker_not_called(
    make_agent, fake_llm, fake_faithfulness_checker
):
    checker = fake_faithfulness_checker()
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    # Not passed to make_agent as the active faithfulness checker -> cfg.faithfulness
    # stays at its default (enabled=False), so ChatAgent builds its own (unused) checker.
    agent = make_agent(llm)
    agent.faithfulness = checker  # would be called if enabled — assert it isn't

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert "faithfulness" not in citations[0]
    assert checker.calls == []


def test_faithfulness_check_attaches_verdict_to_cited_ref(
    make_agent, fake_llm, fake_faithfulness_checker
):
    checker = fake_faithfulness_checker()
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm, faithfulness=checker)
    agent.cfg.retrieval.min_k = 1  # pin a single deterministic result
    agent.cfg.retrieval.max_k = 1

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert len(citations) == 1
    faithfulness = citations[0]["faithfulness"]
    assert len(faithfulness) == 1
    assert faithfulness[0]["sentence"] == "Latent attention shrinks the cache [r1]."
    assert faithfulness[0]["label"] == "entailment"

    # Scored sentence-vs-sentence against the passage, not the whole body.
    assert checker.calls
    for premise, hypothesis in checker.calls:
        assert hypothesis == "Latent attention shrinks the cache [r1]."
        assert premise == "multi head latent attention kv cache"


def test_faithfulness_skips_refs_never_cited_in_text(
    make_agent, fake_llm, fake_faithfulness_checker
):
    checker = fake_faithfulness_checker()
    # Cites only r1 even though the answer text never mentions r2 at all — r2 (if
    # ever retrieved) would have no hypothesis span to test.
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm, faithfulness=checker)

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert citations[0]["ref"] == "r1"
    assert "faithfulness" in citations[0]


def test_faithfulness_ref_cited_twice_gets_two_item_list(
    make_agent, fake_llm, fake_faithfulness_checker
):
    checker = fake_faithfulness_checker()
    llm = fake_llm(
        answer="MLA shrinks the cache [r1]. It also speeds up decoding [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm, faithfulness=checker)

    _text, citations, _usage = agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert len(citations[0]["faithfulness"]) == 2


def test_faithfulness_contradiction_triggers_log_line(
    make_agent, fake_llm, fake_faithfulness_checker, capsys
):
    from rag.faithfulness import Verdict

    checker = fake_faithfulness_checker(verdict=Verdict(label="contradiction", score=0.0))
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm, faithfulness=checker)
    agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert "[faithfulness]" in capsys.readouterr().out


class _FlakyOnFullHistorySynthesisLLM:
    """Duck-typed LLM client: any per-paper sub-run call answers directly (no search
    scripted); the synthesis call (identified by its distinct system prompt) raises on its
    first attempt — as if the full-history request overflowed context — then succeeds on
    the retry, so the test can assert compare()'s degrade-and-retry path end to end."""

    def __init__(self) -> None:
        self.run_tools_calls: list[dict] = []
        self.synthesis_attempts = 0

    def complete(self, system, user, max_tokens=None):
        return "title"

    def run_tools(
        self,
        system,
        messages,
        tools,
        execute,
        on_text=None,
        on_reasoning=None,
        max_rounds=8,
        stop_check=None,
    ):
        self.run_tools_calls.append(
            {"system": system, "messages": messages, "tools": tools, "max_rounds": max_rounds}
        )
        if system.startswith("You already ran"):
            self.synthesis_attempts += 1
            if self.synthesis_attempts == 1:
                raise RuntimeError("context length exceeded")
            if on_text:
                on_text("Synthesized answer.")
            return "Synthesized answer.", Usage(input_tokens=5, output_tokens=5)
        if on_text:
            on_text("Row answer.")
        return "Row answer.", Usage(input_tokens=1, output_tokens=1)


def test_compare_raises_when_fewer_than_two_papers_resolve(make_agent, fake_llm):
    llm = fake_llm(answer="ok", tool_calls=[])
    agent = make_agent(llm)
    with pytest.raises(ValueError):
        agent.compare(
            [{"role": "user", "content": "x"}],
            tags=[],
            papers=["paper-a"],
            on_text=lambda _t: None,
            on_row=lambda _r: None,
        )


def test_compare_builds_one_row_per_paper_and_continues_ref_numbering_into_synthesis(
    make_agent, fake_llm
):
    # tool_calls fires on every run_tools call the FakeLLM sees, including the synthesis
    # round — this exercises the synthesis-round executor's defensive search path (built
    # by ChatAgent._build_search_executor with search_budget=None; see agent.compare's
    # "A search tool is available but you should not need it") for free, on top of the
    # two per-paper searches, so 3 citations are expected, not 2.
    llm = fake_llm(answer="See [r1].", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 1

    rows_seen: list[dict] = []
    _text, compare_results, citations, _usage = agent.compare(
        [{"role": "user", "content": "compare the two papers"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_row=rows_seen.append,
    )

    assert {r["paper_id"] for r in compare_results} == {"paper-a", "paper-b"}
    assert rows_seen == compare_results  # on_row fired once per completed paper, in order
    for row in compare_results:
        assert len(row["citations"]) == 1
        assert row["citations"][0]["paper_id"] == row["paper_id"]

    # 2 per-paper citations + 1 from the synthesis round's defensive search.
    assert [c["ref"] for c in citations] == ["r1", "r2", "r3"]  # no collisions


def test_synthesis_call_gets_full_history_and_the_defensive_search_tool(make_agent, fake_llm):
    llm = fake_llm(answer="Combined answer.", tool_calls=[])
    agent = make_agent(llm)

    messages = [
        {"role": "user", "content": "what's the model size in each?"},
        {"role": "assistant", "content": "Paper A uses 7B params [r1]."},
        {"role": "user", "content": "and their training data?"},
    ]
    text, _compare_results, _citations, _usage = agent.compare(
        messages, tags=[], papers=[], on_text=lambda _t: None, on_row=lambda _r: None
    )

    assert text == "Combined answer."
    synthesis_call = llm.run_tools_calls[-1]
    assert synthesis_call["messages"] == messages  # full history, not just the last message
    assert synthesis_call["tools"] == [agent.search_tool]
    assert synthesis_call["max_rounds"] == 2


def test_synthesis_failure_retries_with_trimmed_history(make_agent):
    # The retry fires on any exception from the first attempt (context overflow included,
    # but not asserted as the cause here or in the caveat text — see agent.py's comment on
    # why the wording doesn't name a specific diagnosis it hasn't verified).
    llm = _FlakyOnFullHistorySynthesisLLM()
    agent = make_agent(llm)

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    text, _compare_results, _citations, _usage = agent.compare(
        messages, tags=[], papers=[], on_text=lambda _t: None, on_row=lambda _r: None
    )

    assert llm.synthesis_attempts == 2
    synthesis_calls = [c for c in llm.run_tools_calls if c["system"].startswith("You already ran")]
    assert len(synthesis_calls) == 2
    assert synthesis_calls[0]["messages"] == messages  # first attempt: full history
    assert synthesis_calls[1]["messages"] == [messages[-1]]  # retry: trimmed
    assert text.startswith("_(retried with just your current question")
    assert "Synthesized answer." in text


def test_subrun_failure_produces_placeholder_row_and_continues(make_agent, fake_llm):
    llm = fake_llm(answer="See [r1].", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 1

    original_run = agent.run

    def flaky_run(messages, tags, papers, **kwargs):
        if papers == ["paper-b"]:
            raise RuntimeError("boom")
        return original_run(messages, tags, papers, **kwargs)

    agent.run = flaky_run

    _text, compare_results, _citations, _usage = agent.compare(
        [{"role": "user", "content": "compare"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_row=lambda _r: None,
    )

    failed = next(r for r in compare_results if r["paper_id"] == "paper-b")
    assert failed["text"] == "_(search failed for this paper)_"
    assert failed["citations"] == []
    ok = next(r for r in compare_results if r["paper_id"] == "paper-a")
    assert ok["citations"]
    # the placeholder still feeds into synthesis instead of being silently dropped.
    synthesis_call = llm.run_tools_calls[-1]
    assert "_(search failed for this paper)_" in synthesis_call["system"]


def test_compare_stopped_before_synthesis_falls_back_to_flattened_rows(make_agent, fake_llm):
    llm = fake_llm(answer="Paper answer.", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 1

    calls = {"n": 0}

    def stop_check():
        # False for the check before paper 1 (lets it run); True from then on, so paper 2
        # never starts and the after-loop check skips synthesis.
        calls["n"] += 1
        return calls["n"] > 1

    text, compare_results, citations, usage = agent.compare(
        [{"role": "user", "content": "compare"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_row=lambda _r: None,
        stop_check=stop_check,
    )

    assert len(compare_results) == 1  # only the first paper's sub-run ran
    row = compare_results[0]
    assert text == f"## {row['title']}\n\n{row['text']}\n\n"
    assert citations == row["citations"]
    assert usage.input_tokens is None and usage.output_tokens is None


def test_faithfulness_attaches_to_the_synthesized_text(
    make_agent, fake_llm, fake_faithfulness_checker
):
    checker = fake_faithfulness_checker()
    llm = fake_llm(answer="Combined finding [r1].", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm, faithfulness=checker)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 1

    _text, _compare_results, citations, _usage = agent.compare(
        [{"role": "user", "content": "compare"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_row=lambda _r: None,
    )

    cited = next(c for c in citations if c["ref"] == "r1")
    assert "faithfulness" in cited


def test_faithfulness_attaches_to_fullwidth_bracket_citations_in_synthesis(
    make_agent, fake_llm, fake_faithfulness_checker
):
    # Regression for a real observed case: a local model synthesized an answer citing
    # 【r1】 (fullwidth CJK brackets) instead of [r1] — the top-level faithfulness check
    # on the synthesized text must still find and attach it, end to end through
    # ChatAgent.compare, not just at the attribute_refs unit-test level.
    checker = fake_faithfulness_checker()
    llm = fake_llm(answer="Combined finding【r1】.", tool_calls=[("search_papers", {"query": "x"})])
    agent = make_agent(llm, faithfulness=checker)
    agent.cfg.retrieval.min_k = 1
    agent.cfg.retrieval.max_k = 1

    _text, _compare_results, citations, _usage = agent.compare(
        [{"role": "user", "content": "compare"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
        on_row=lambda _r: None,
    )

    cited = next(c for c in citations if c["ref"] == "r1")
    assert "faithfulness" in cited


def test_faithfulness_all_entailment_does_not_log(
    make_agent, fake_llm, fake_faithfulness_checker, capsys
):
    checker = fake_faithfulness_checker()  # default: entailment
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache"})],
    )
    agent = make_agent(llm, faithfulness=checker)
    agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert "[faithfulness]" not in capsys.readouterr().out


def test_classify_mode_short_circuits_to_ask_below_2_papers_without_an_llm_call(
    make_agent, fake_llm, monkeypatch
):
    agent = make_agent(fake_llm(answer="ok"))
    build_calls: list = []
    monkeypatch.setattr(
        "server.agent.build_llm",
        lambda spec: build_calls.append(spec) or fake_llm(answer="COMPARE"),
    )
    mode, scope_size = agent.classify_mode(
        [{"role": "user", "content": "x"}], tags=[], papers=["paper-a"]
    )
    assert (mode, scope_size) == ("ask", 1)
    assert build_calls == []


def test_classify_mode_returns_compare_when_llm_says_compare(make_agent, fake_llm, monkeypatch):
    agent = make_agent(fake_llm(answer="ok"))
    classifier = fake_llm(answer="COMPARE")
    monkeypatch.setattr("server.agent.build_llm", lambda spec: classifier)
    mode, scope_size = agent.classify_mode(
        [{"role": "user", "content": "compare them"}], tags=[], papers=[]
    )
    assert (mode, scope_size) == ("compare", 2)
    assert len(classifier.complete_calls) == 1
    call = classifier.complete_calls[0]
    # CLASSIFY_SYSTEM_PROMPT is a template (`{papers}`) filled in per call, not sent verbatim.
    assert call["system"] == CLASSIFY_SYSTEM_PROMPT.format(
        papers="paper-a (Paper A; moe), paper-b (Paper B; rl)"
    )
    assert call["max_tokens"] == 256
    assert "compare them" in call["user"]


def test_classify_mode_lists_the_resolved_papers_in_the_classify_prompt(
    make_agent, fake_llm, monkeypatch
):
    # So the classifier can tell a question needs Compare from the scope alone (e.g. "what's
    # the model size?" over several distinct model papers) even when the question itself
    # never says "each" or "compare".
    agent = make_agent(fake_llm(answer="ok"))
    classifier = fake_llm(answer="ASK")
    monkeypatch.setattr("server.agent.build_llm", lambda spec: classifier)
    agent.classify_mode([{"role": "user", "content": "what's the model size?"}], tags=[], papers=[])
    system = classifier.complete_calls[0]["system"]
    assert "Paper A" in system
    assert "Paper B" in system


def test_classify_mode_returns_ask_when_llm_says_ask(make_agent, fake_llm, monkeypatch):
    agent = make_agent(fake_llm(answer="ok"))
    classifier = fake_llm(answer="ASK")
    monkeypatch.setattr("server.agent.build_llm", lambda spec: classifier)
    mode, scope_size = agent.classify_mode(
        [{"role": "user", "content": "what is MLA?"}], tags=[], papers=[]
    )
    assert (mode, scope_size) == ("ask", 2)


def test_classify_mode_defaults_to_ask_on_malformed_llm_output(make_agent, fake_llm, monkeypatch):
    agent = make_agent(fake_llm(answer="ok"))
    classifier = fake_llm(answer="I'm not sure, maybe compare them?")
    monkeypatch.setattr("server.agent.build_llm", lambda spec: classifier)
    mode, _scope = agent.classify_mode([{"role": "user", "content": "x"}], tags=[], papers=[])
    assert mode == "ask"


def test_classify_mode_tolerates_trailing_punctuation_on_compare(make_agent, fake_llm, monkeypatch):
    # An otherwise-compliant "COMPARE." (trailing period) must still classify as compare —
    # only substring-style false positives (a sentence merely containing "compare") should
    # fall through to ask, not trivial trailing punctuation on an exact one-word reply.
    agent = make_agent(fake_llm(answer="ok"))
    classifier = fake_llm(answer="COMPARE.")
    monkeypatch.setattr("server.agent.build_llm", lambda spec: classifier)
    mode, _scope = agent.classify_mode([{"role": "user", "content": "x"}], tags=[], papers=[])
    assert mode == "compare"


def test_classify_mode_defaults_to_ask_on_llm_exception(make_agent, fake_llm, monkeypatch):
    agent = make_agent(fake_llm(answer="ok"))

    class _RaisingLLM:
        def complete(self, system, user, max_tokens=None):
            raise RuntimeError("boom")

    monkeypatch.setattr("server.agent.build_llm", lambda spec: _RaisingLLM())
    mode, scope_size = agent.classify_mode([{"role": "user", "content": "x"}], tags=[], papers=[])
    assert (mode, scope_size) == ("ask", 2)


def test_classify_mode_sees_full_conversation_history(make_agent, fake_llm, monkeypatch):
    agent = make_agent(fake_llm(answer="ok"))
    classifier = fake_llm(answer="ASK")
    monkeypatch.setattr("server.agent.build_llm", lambda spec: classifier)
    messages = [
        {"role": "user", "content": "What optimizer does paper-a use?"},
        {"role": "assistant", "content": "AdamW [r1]."},
        {"role": "user", "content": "and paper-b?"},
    ]
    agent.classify_mode(messages, tags=[], papers=[])
    transcript = classifier.complete_calls[0]["user"]
    for m in messages:
        assert m["content"] in transcript
