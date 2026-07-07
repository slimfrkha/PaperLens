"""Agentic-RAG chat loop with a scripted LLM over a real retrieval seam."""

from __future__ import annotations

import pytest

from rag.manifest import Manifest
from server.agent import ChatAgent


@pytest.fixture
def make_agent(make_searcher, seed_chunks):
    """Factory: a ChatAgent wired to a seeded searcher, manifest, and FakeLLM."""

    def _make(llm):
        docs = [
            seed_chunks("deepseek-v3", "Attention", "multi head latent attention kv cache"),
            seed_chunks("glm-4.5", "Training", "reinforcement learning recipe"),
        ]
        ctx = make_searcher(docs)
        manifest = Manifest(ctx.cfg.paths.rag_db)
        manifest.upsert({"paper_id": "deepseek-v3", "title": "DeepSeek-V3", "tags": ["moe"]})
        manifest.upsert({"paper_id": "glm-4.5", "title": "GLM-4.5", "tags": ["rl"]})
        agent = ChatAgent(ctx.cfg, ctx.searcher, manifest, client=llm)
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
    assert citations[0]["paper_id"] == "deepseek-v3"
    assert citations[0]["title"] == "DeepSeek-V3"

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


def test_tag_filter_scopes_search_to_matching_papers(make_agent, fake_llm):
    # tags=["rl"] -> only glm-4.5; a query is restricted to that paper.
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
    assert {c["paper_id"] for c in citations} == {"glm-4.5"}


def test_paper_filter_scopes_search_to_selected_papers(make_agent, fake_llm):
    # papers=["deepseek-v3"] -> search is restricted to that paper regardless of query.
    llm = fake_llm(
        answer="See [r1].",
        tool_calls=[("search_papers", {"query": "reinforcement learning recipe"})],
    )
    agent = make_agent(llm)

    _text, citations = agent.run(
        [{"role": "user", "content": "training?"}],
        tags=[],
        papers=["deepseek-v3"],
        on_text=lambda _t: None,
    )
    assert citations
    assert {c["paper_id"] for c in citations} == {"deepseek-v3"}


def test_tag_and_paper_filters_intersect(make_agent, fake_llm):
    # tags -> glm-4.5, papers -> deepseek-v3: disjoint, so nothing is searchable.
    llm = fake_llm(answer="none", tool_calls=[("search_papers", {"query": "anything"})])
    agent = make_agent(llm)

    _text, citations = agent.run(
        [{"role": "user", "content": "x"}],
        tags=["rl"],
        papers=["deepseek-v3"],
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
        [{"role": "user", "content": "x"}], tags=[], papers=["glm-4.5"], on_text=lambda _t: None
    )
    scoped = scoped_llm.run_tools_calls[0]["system"]
    assert "GLM-4.5" in scoped
    assert "DeepSeek-V3" not in scoped

    full_llm = fake_llm(answer="x", tool_calls=[])
    make_agent(full_llm).run(
        [{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None
    )
    full = full_llm.run_tools_calls[0]["system"]
    assert "GLM-4.5" in full and "DeepSeek-V3" in full


def test_empty_query_is_rejected(make_agent, fake_llm):
    llm = fake_llm(answer="done", tool_calls=[("search_papers", {"query": "   "})])
    agent = make_agent(llm)
    _text, citations = agent.run(
        [{"role": "user", "content": "x"}], tags=[], papers=[], on_text=lambda _t: None
    )
    # Blank query -> no search performed, no citations.
    assert citations == []
