"""Searcher construction wiring and lazy-init failure handling (offline: no local model, no
network)."""

from __future__ import annotations

import pytest

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


def test_result_carries_section_number(make_searcher, seed_chunks):
    # section_number is stored on every chunk's metadata but was previously dropped on the
    # way back into Result — needed downstream for section-identity scoring (paperlens-eval)
    # and for a mined feedback record to be usable for that purpose.
    ctx = make_searcher([seed_chunks("paper-a", "Attention", "latent attention over the kv cache")])
    results = ctx.searcher.search("attention", k=1, candidates=10, rerank=False)
    assert results[0].section_number == "1"


def test_rerank_lazy_reranker_load_failure_falls_back(monkeypatch, make_searcher, seed_chunks):
    # The default reranker is built lazily, on first access to Searcher.reranker — this proves
    # the fallback also covers that construction, not just an already-injected reranker's score().
    monkeypatch.setattr("rag.search.CrossEncoderReranker", _RaisingCrossEncoderReranker)
    ctx = make_searcher([seed_chunks("paper-a", "Attention", "latent attention over the kv cache")])

    dense = ctx.searcher.search("attention", k=1, candidates=10, rerank=False)
    results = ctx.searcher.search("attention", k=1, candidates=10, rerank=True)

    assert [r.paper_id for r in results] == [r.paper_id for r in dense]
    assert [r.score for r in results] == [r.score for r in dense]


class _VecEmbedder:
    """Deterministic embedder keyed by exact text, for hand-picked cosine similarities.

    Unlike the hash-based ``fake_embedder`` fixture, this gives full control over which
    doc is closest to which query string — needed to assert multi-query fusion actually
    changes what's retrieved, not just that it runs.
    """

    _VECS = {
        "orig": [10.0, 1.0, 0.0, 0.0],
        "doc_a_text": [10.0, 0.0, 0.0, 0.0],  # cos(orig) ~= 0.995 -> always rank 1 for "orig"
        "doc_c_text": [1.0, 1.0, 0.0, 0.0],  # cos(orig) ~= 0.774 -> always rank 2 for "orig"
        "doc_b_text": [0.0, 0.0, 10.0, 0.0],  # cos(orig) = 0 -> always rank 3 for "orig"
        "paraphrase_text": [0.0, 0.0, 10.0, 1.0],  # cos(doc_b_text) ~= 0.995 -> rank 1 for it
    }

    def name(self) -> str:
        return "vec"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def _seed_vec_docs(cfg, embedder):
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [
        ("a", "doc_a_text", "Alpha"),
        ("b", "doc_b_text", "Beta"),
        ("c", "doc_c_text", "Gamma"),
    ]
    metas = [
        {
            "paper_id": f"paper-{doc_id}",
            "breadcrumb": f"Paper > {section}",
            "section_title": section,
            "section_number": "1",
            "body": text,
        }
        for doc_id, text, section in docs
    ]
    collection.upsert(
        ids=[doc_id for doc_id, _, _ in docs],
        embeddings=embedder([text for _, text, _ in docs]),
        documents=[text for _, text, _ in docs],
        metadatas=metas,
    )
    return collection


def test_multi_query_surfaces_chunk_single_query_misses(make_config, fake_llm):
    cfg = make_config()
    embedder = _VecEmbedder()
    _seed_vec_docs(cfg, embedder)
    llm = fake_llm(answer='["paraphrase_text"]')
    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        multi_query_enabled=True,
        multi_query_n=1,
        llm=llm,
    )

    single = searcher.search("orig", k=1, candidates=1, rerank=False)
    assert [r.paper_id for r in single] == ["paper-a"]  # doc_b never even fetched

    fused = searcher.search("orig", k=2, candidates=2, rerank=False)
    assert "paper-b" in [r.paper_id for r in fused]  # surfaced via the paraphrase


def test_multi_query_with_sparse_returns_sane_fused_set(make_config, fake_llm):
    cfg = make_config()
    embedder = _VecEmbedder()
    _seed_vec_docs(cfg, embedder)
    llm = fake_llm(answer='["paraphrase_text"]')
    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        sparse_enabled=True,
        multi_query_enabled=True,
        multi_query_n=1,
        llm=llm,
    )

    results = searcher.search("orig", k=3, candidates=3, rerank=False)

    assert results  # both fusion layers (hybrid + multi-query) ran without crashing
    assert {r.paper_id for r in results} <= {"paper-a", "paper-b", "paper-c"}


def test_multi_query_llm_failure_falls_back_to_single_query(make_config, fake_llm):
    cfg = make_config()
    embedder = _VecEmbedder()
    _seed_vec_docs(cfg, embedder)

    class RaisingLLM(fake_llm):
        def complete(self, system, user, max_tokens=None):
            raise RuntimeError("backend down")

    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        multi_query_enabled=True,
        llm=RaisingLLM(),
    )

    results = searcher.search("orig", k=1, candidates=1, rerank=False)

    assert [r.paper_id for r in results] == ["paper-a"]  # no crash, no paraphrase applied


def test_multi_query_widens_fetch_independent_of_sparse(make_config, fake_llm, monkeypatch):
    cfg = make_config()
    embedder = _VecEmbedder()
    _seed_vec_docs(cfg, embedder)
    llm = fake_llm(answer='["paraphrase_text"]')
    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        sparse_enabled=False,
        multi_query_enabled=True,
        multi_query_n=1,
        multi_query_fetch_multiplier=3,
        llm=llm,
    )
    seen_n_results: list[int] = []
    original_query = searcher.collection.query

    def spy_query(*args, **kwargs):
        seen_n_results.append(kwargs["n_results"])
        return original_query(*args, **kwargs)

    monkeypatch.setattr(searcher.collection, "query", spy_query)

    searcher.search("orig", k=1, candidates=1, rerank=False)

    # candidates(1) * multi_query_fetch_multiplier(3), not candidates(1) alone — the
    # fetch-headroom fix: without it, each variant would fetch depth 1 and the
    # across-variant fuse would have no margin to promote anything.
    assert seen_n_results == [3, 3]


class _FlatEmbedder:
    """Every text (query or doc) embeds to the same vector — with a small corpus, dense
    recall always returns everyone regardless of query text, so only BM25 discriminates.
    Isolates the `.source` tagging test below from RRF/truncation edge cases entirely."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in input]


def test_multi_query_source_unions_dense_and_sparse_across_variants(
    make_config, seed_chunks, fake_llm
):
    # Result.source must union dense/sparse membership across every variant, not just the one
    # variant that happens to surface a given id — exercises the dense_ids_all/sparse_ids_all
    # accumulation in the multi-query branch (search.py), untested until now.
    from rag.index import open_collection

    embedder = _FlatEmbedder()
    cfg = make_config()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name="flat")
    docs = [
        seed_chunks("p-dense", "S", "neutral filler alpha", doc_id="dense-tag"),
        seed_chunks("p-both-orig", "S", "orig appears in this passage", doc_id="both-orig"),
        seed_chunks("p-both-para", "S", "zzflorble appears in this passage", doc_id="both-para"),
    ]
    texts = [text for _, text, _ in docs]
    collection.upsert(
        ids=[doc_id for doc_id, _, _ in docs],
        embeddings=embedder(texts),
        documents=texts,
        metadatas=[meta for _, _, meta in docs],
    )
    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        sparse_enabled=True,
        multi_query_enabled=True,
        multi_query_n=1,
        llm=fake_llm(answer='["zzflorble"]'),
    )

    results = searcher.search("orig", k=3, candidates=3, rerank=False)
    by_paper = {r.paper_id: r.source for r in results}

    assert by_paper["p-dense"] == "dense"  # dense (flat embedder) only — no lexical overlap
    assert by_paper["p-both-orig"] == "both"  # dense + BM25 on the original query "orig"
    assert by_paper["p-both-para"] == "both"  # dense + BM25 on the paraphrase "zzflorble"


def test_multi_query_enabled_without_llm_raises(tmp_path):
    with pytest.raises(ValueError, match="multi_query_enabled"):
        Searcher(
            db_dir=str(tmp_path / "rag_db"),
            collection="test_papers",
            multi_query_enabled=True,
        )
