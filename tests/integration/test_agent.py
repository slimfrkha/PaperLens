"""Agentic-RAG chat loop with a scripted LLM over a real retrieval seam."""

from __future__ import annotations

import pytest

from rag.config import HFFaithfulnessCfg
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
        manifest.upsert({"paper_id": "paper-a", "title": "Paper A", "tags": ["moe"]})
        manifest.upsert({"paper_id": "paper-b", "title": "Paper B", "tags": ["rl"]})
        agent = ChatAgent(ctx.cfg, ctx.searcher, manifest, client=llm, faithfulness=faithfulness)
        return agent

    return _make


def test_search_call_builds_citations_and_trace(make_agent, fake_llm):
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
        reasoning="I should search.",
    )
    agent = make_agent(llm)

    trace = []
    texts = []
    text, citations = agent.run(
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
    assert citations[0]["source"] == "dense"
    assert citations[0]["section_number"] == "1"
    assert citations[0]["body"] == "multi head latent attention kv cache"

    kinds = [e["type"] for e in trace]
    assert kinds == ["thought", "action", "observation"]


def test_small_talk_answers_without_searching(make_agent, fake_llm):
    llm = fake_llm(answer="Hello!", tool_calls=[])
    agent = make_agent(llm)

    text, citations = agent.run(
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
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
    )
    agent = make_agent(llm)

    _text, citations = agent.run(
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

    _text, citations = agent.run(
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

    _text, citations = agent.run(
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

    _text, citations = agent.run(
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
    _text, citations = agent.run(
        [{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None
    )
    # Blank query -> no search performed, no citations.
    assert citations == []


def test_uses_configured_retrieval_defaults_when_top_k_omitted(make_agent, fake_llm):
    llm = fake_llm(
        answer="ok", tool_calls=[("search_papers", {"query": "latent attention"})]
    )  # no top_k -> falls back to retrieval.k
    agent = make_agent(llm)
    agent.cfg.retrieval.k = 2
    agent.cfg.retrieval.candidates = 9

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run([{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None)

    assert calls[0]["k"] == 2
    assert calls[0]["candidates"] == 9


def test_candidate_pool_scales_with_requested_top_k(make_agent, fake_llm):
    # A large top_k must still leave the reranker something to discard, otherwise
    # it just reorders exactly what dense recall returned.
    llm = fake_llm(answer="ok", tool_calls=[("search_papers", {"query": "mla", "top_k": 50})])
    agent = make_agent(llm)

    calls: list[dict] = []
    real_search = agent.searcher.search

    def spy(query, **kw):
        calls.append(kw)
        return real_search(query, **kw)

    agent.searcher.search = spy
    agent.run([{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None)

    assert calls[0]["k"] == 50
    assert calls[0]["candidates"] == 200


def test_max_rounds_flows_from_retrieval_config(make_agent, fake_llm):
    llm = fake_llm(answer="ok", tool_calls=[])
    agent = make_agent(llm)
    agent.cfg.retrieval.max_rounds = 3
    agent.run([{"role": "user", "content": "hi"}], tags=[], papers=[], on_text=lambda _t: None)
    assert llm.run_tools_calls[0]["max_rounds"] == 3


def test_faithfulness_disabled_by_default_no_verdict_and_checker_not_called(
    make_agent, fake_llm, fake_faithfulness_checker
):
    checker = fake_faithfulness_checker()
    llm = fake_llm(
        answer="Latent attention shrinks the cache [r1].",
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
    )
    # Not passed to make_agent as the active faithfulness checker -> cfg.faithfulness
    # stays at its default (enabled=False), so ChatAgent builds its own (unused) checker.
    agent = make_agent(llm)
    agent.faithfulness = checker  # would be called if enabled — assert it isn't

    _text, citations = agent.run(
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
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
    )
    agent = make_agent(llm, faithfulness=checker)

    _text, citations = agent.run(
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
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
    )
    agent = make_agent(llm, faithfulness=checker)

    _text, citations = agent.run(
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
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
    )
    agent = make_agent(llm, faithfulness=checker)

    _text, citations = agent.run(
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
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
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
        tool_calls=[("search_papers", {"query": "latent attention kv cache", "top_k": 1})],
    )
    agent = make_agent(llm, faithfulness=checker)
    agent.run(
        [{"role": "user", "content": "How does MLA help?"}],
        tags=[],
        papers=[],
        on_text=lambda _t: None,
    )
    assert "[faithfulness]" not in capsys.readouterr().out
