"""Searcher construction wiring (offline: no local model, no network)."""

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
