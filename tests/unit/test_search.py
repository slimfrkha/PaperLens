"""Searcher construction wiring and lazy-init failure handling (offline: no local model, no
network)."""

from __future__ import annotations

from rag.index import open_collection
from rag.search import Searcher


class _SpyHFEmbedder:
    """Records the kwargs Searcher's default-embedder branch constructs it with."""

    captured: dict = {}

    def __init__(self, model_name, device=None, query_prefix=""):
        _SpyHFEmbedder.captured = {
            "model_name": model_name,
            "device": device,
            "query_prefix": query_prefix,
        }

    def name(self) -> str:
        return "spy"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.0] for _ in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return [[0.0] for _ in input]


def test_searcher_threads_query_prefix_to_default_embedder(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.search.HFEmbedder", _SpyHFEmbedder)
    db_dir = str(tmp_path / "rag_db")
    open_collection(db_dir, "test_papers")

    Searcher(db_dir=db_dir, collection="test_papers", query_prefix="query: ")

    assert _SpyHFEmbedder.captured["query_prefix"] == "query: "


class _RaisingCrossEncoderReranker:
    """Stands in for CrossEncoderReranker; fails at construction (a lazy model-load error)."""

    def __init__(self, *args, **kwargs):
        raise OSError("model not found")


def test_rerank_lazy_reranker_load_failure_falls_back(monkeypatch, make_searcher, seed_chunks):
    # The default reranker is built lazily, on first access to Searcher.reranker — this proves
    # the fallback also covers that construction, not just an already-injected reranker's score().
    monkeypatch.setattr("rag.search.CrossEncoderReranker", _RaisingCrossEncoderReranker)
    ctx = make_searcher([seed_chunks("paper-a", "Attention", "latent attention over the kv cache")])

    dense = ctx.searcher.search("attention", k=1, candidates=10, rerank=False)
    results = ctx.searcher.search("attention", k=1, candidates=10, rerank=True)

    assert [r.paper_id for r in results] == [r.paper_id for r in dense]
    assert [r.score for r in results] == [r.score for r in dense]
