"""Agentic-RAG chat loop with a scripted LLM over a real retrieval seam."""

from __future__ import annotations

import pytest

from rag.config import HFFaithfulnessCfg
from rag.llm import Usage
from rag.manifest import Manifest
from server.agent import ChatAgent


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
