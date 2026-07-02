"""Retrieval over a real temp Chroma DB with a fake embedder (rerank disabled)."""

from __future__ import annotations


def _docs(seed_chunks):
    return [
        seed_chunks("deepseek-v3", "Attention", "multi head latent attention shrinks kv cache"),
        seed_chunks("glm-4.5", "Training", "reinforcement learning from human feedback recipe"),
    ]


def test_retrieves_the_relevant_passage_first(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    results = ctx.searcher.search("latent attention kv cache", k=2, candidates=10, rerank=False)
    assert results
    assert results[0].paper_id == "deepseek-v3"
    assert "latent attention" in results[0].body


def test_paper_filter_restricts_to_one_paper(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    results = ctx.searcher.search(
        "reinforcement learning", k=5, candidates=10, paper="glm-4.5", rerank=False
    )
    assert results
    assert {r.paper_id for r in results} == {"glm-4.5"}


def test_empty_paper_ids_matches_nothing(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    assert ctx.searcher.search("anything", paper_ids=[], rerank=False) == []


def test_paper_and_paper_ids_intersect(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    # `paper` not in the tag-derived `paper_ids` -> empty intersection -> no hits.
    out = ctx.searcher.search("attention", paper="deepseek-v3", paper_ids=["glm-4.5"], rerank=False)
    assert out == []


def test_result_body_strips_breadcrumb_prefix(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    r = ctx.searcher.search("latent attention", candidates=10, rerank=False)[0]
    assert not r.body.startswith("Paper >")
    assert r.breadcrumb.startswith("Paper >")
