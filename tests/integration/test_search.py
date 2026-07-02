"""Retrieval over a real temp Chroma DB with a fake embedder (rerank disabled)."""

from __future__ import annotations

from rag.reranker import Reranker
from rag.search import Searcher


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


class _KeywordReranker(Reranker):
    """Scores a passage 1.0 iff it contains ``needle``, else 0.0."""

    def __init__(self, needle: str):
        self.needle = needle

    def score(self, query, docs):
        return [1.0 if self.needle in d else 0.0 for d in docs]

    @classmethod
    def build(cls, model, *, device, llm):
        return cls("")


def test_rerank_uses_injected_reranker(make_searcher, fake_embedder, seed_chunks):
    # An injected reranker reorders results (and proves the seam runs offline —
    # no cross-encoder model is loaded). The passage carrying the needle wins.
    ctx = make_searcher(
        [
            seed_chunks("deepseek-v3", "Attention", "latent attention over the kv cache"),
            seed_chunks("glm-4.5", "Rewards", "reinforcement learning UNIQUENEEDLE recipe"),
        ]
    )
    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=fake_embedder,
        reranker=_KeywordReranker("UNIQUENEEDLE"),
    )
    results = searcher.search("learning", k=2, candidates=10, rerank=True)
    assert results[0].paper_id == "glm-4.5"
    assert results[0].score == 1.0
    assert results[0].score >= results[-1].score
