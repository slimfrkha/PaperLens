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


def test_dense_recall_matches_search_dense_only_pool(make_searcher, seed_chunks):
    # dense_recall is now a public primitive eval's harness/optimizer call directly (instead of
    # reimplementing dense recall by hand) — this pins its contract against search()'s own
    # dense-only, no-rerank path, which is built from the exact same by_id values.
    docs = [
        seed_chunks("paper-a", "Attention", "latent attention over the kv cache"),
        seed_chunks("paper-b", "Training", "large batch training run details"),
    ]
    ctx = make_searcher(docs)

    dense_ids, by_id = ctx.searcher.dense_recall("attention", fetch_n=2, where=None)
    dense_results = ctx.searcher.search("attention", k=2, candidates=2, rerank=False)

    assert [by_id[cid].paper_id for cid in dense_ids] == [r.paper_id for r in dense_results]
    top = by_id[dense_ids[0]]
    assert (top.score, top.text, top.body, top.breadcrumb) == (
        dense_results[0].score,
        dense_results[0].text,
        dense_results[0].body,
        dense_results[0].breadcrumb,
    )


def test_backfill_missing_adds_placeholder_results_outside_the_dense_pool(
    make_searcher, seed_chunks
):
    docs = [
        seed_chunks("paper-a", "Attention", "latent attention over the kv cache"),
        seed_chunks("paper-b", "Training", "large batch training run details"),
        seed_chunks("paper-c", "Inference", "fast inference serving pipeline"),
    ]
    ctx = make_searcher(docs)
    all_ids = [doc_id for doc_id, _, _ in docs]

    dense_ids, by_id = ctx.searcher.dense_recall("attention", fetch_n=1, where=None)
    real_score = by_id[dense_ids[0]].score

    ctx.searcher.backfill_missing(by_id, all_ids)

    assert set(by_id) == set(all_ids)
    backfilled_ids = [cid for cid in all_ids if cid not in dense_ids]
    assert len(backfilled_ids) == 2
    for cid in backfilled_ids:
        assert by_id[cid].score == 0.0
        assert by_id[cid].text  # metadata round-tripped, not left empty

    # Backfilling an id already present is a no-op — doesn't clobber the real dense-recall score
    # with the placeholder.
    ctx.searcher.backfill_missing(by_id, dense_ids)
    assert by_id[dense_ids[0]].score == real_score


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


class _HybridMultiQueryEmbedder:
    """Deterministic per-text vectors on distinct axes per query variant, so dense recall
    genuinely discriminates which variant favors which doc — unlike `_FlatEmbedder` below,
    which can't tell variants apart at all and is deliberately used elsewhere to isolate
    source-tagging from ranking."""

    _VECS = {
        "orig123": [10.0, 1.0, 0.0, 0.0],
        "para456": [0.0, 0.0, 10.0, 1.0],
        "filler text sharing no terms with either query": [10.0, 0.0, 0.0, 0.0],
        "para456 appears verbatim in this passage": [0.0, 0.0, 10.0, 0.0],
        "orig123 appears verbatim in this passage": [0.0, 1.0, 0.0, 0.0],
    }

    def name(self) -> str:
        return "hybrid-multi-query"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def test_multi_query_with_sparse_returns_sane_fused_set(make_config, fake_llm):
    # Proves hybrid + multi-query together combine real lexical and semantic evidence, not
    # just "both layers ran without crashing" (the assertion this replaces). paper-both-para
    # ranks LAST in a plain dense-only "orig123" search (cosine 0.0 to it) but FIRST once
    # hybrid + multi-query are both on; paper-dense-only ranks FIRST in that baseline and
    # LAST in the fused result — a full reversal that only happens if both fusion layers are
    # genuinely contributing, not a coincidence of one dominating.
    cfg = make_config()
    embedder = _HybridMultiQueryEmbedder()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [
        ("dense-only", "filler text sharing no terms with either query", "Alpha"),
        ("both-para", "para456 appears verbatim in this passage", "Beta"),
        ("both-orig", "orig123 appears verbatim in this passage", "Gamma"),
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

    baseline = Searcher(db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder)
    baseline_results = baseline.search("orig123", k=3, candidates=3, rerank=False)
    assert [r.paper_id for r in baseline_results] == [
        "paper-dense-only",
        "paper-both-orig",
        "paper-both-para",
    ]

    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        sparse_enabled=True,
        multi_query_enabled=True,
        multi_query_n=1,
        llm=fake_llm(answer='["para456"]'),
    )
    results = searcher.search("orig123", k=3, candidates=3, rerank=False)

    assert [r.paper_id for r in results] == [
        "paper-both-para",
        "paper-both-orig",
        "paper-dense-only",
    ]
    assert [r.source for r in results] == ["both", "both", "dense"]


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


# --- per_paper retrieval -----------------------------------------------------------------


def test_per_paper_raises_without_resolved_paper_filter(make_searcher, seed_chunks):
    ctx = make_searcher([seed_chunks("paper-a", "Attention", "latent attention over the kv cache")])
    with pytest.raises(ValueError, match="per_paper"):
        ctx.searcher.search("attention", per_paper=True)


def test_per_paper_candidate_budget_floor_binds_at_k(make_searcher, seed_chunks, monkeypatch):
    # candidates(6) // n_papers(3) = 2, below k(5) -> each paper's own fetch is floored to k.
    docs = [
        seed_chunks("paper-a", "S", "alpha content"),
        seed_chunks("paper-b", "S", "beta content"),
        seed_chunks("paper-c", "S", "gamma content"),
    ]
    ctx = make_searcher(docs)
    seen_n_results: list[int] = []
    original_query = ctx.searcher.collection.query

    def spy_query(*args, **kwargs):
        seen_n_results.append(kwargs["n_results"])
        return original_query(*args, **kwargs)

    monkeypatch.setattr(ctx.searcher.collection, "query", spy_query)

    ctx.searcher.search(
        "alpha",
        k=5,
        candidates=6,
        paper_ids=["paper-a", "paper-b", "paper-c"],
        per_paper=True,
        rerank=False,
    )

    assert seen_n_results == [5, 5, 5]


def test_per_paper_candidate_budget_mid_range_no_clamp_needed(
    make_searcher, seed_chunks, monkeypatch
):
    # candidates(20) // n_papers(2) = 10, between k(5) and candidates(20) -> used as-is.
    docs = [
        seed_chunks("paper-a", "S", "alpha content"),
        seed_chunks("paper-b", "S", "beta content"),
    ]
    ctx = make_searcher(docs)
    seen_n_results: list[int] = []
    original_query = ctx.searcher.collection.query

    def spy_query(*args, **kwargs):
        seen_n_results.append(kwargs["n_results"])
        return original_query(*args, **kwargs)

    monkeypatch.setattr(ctx.searcher.collection, "query", spy_query)

    ctx.searcher.search(
        "alpha",
        k=5,
        candidates=20,
        paper_ids=["paper-a", "paper-b"],
        per_paper=True,
        rerank=False,
    )

    assert seen_n_results == [10, 10]


def test_per_paper_single_resolved_paper_is_a_no_op(make_searcher, seed_chunks):
    docs = [
        seed_chunks("paper-a", "S1", "alpha one"),
        seed_chunks("paper-a", "S2", "alpha two"),
        seed_chunks("paper-b", "S1", "beta content"),
    ]
    ctx = make_searcher(docs)

    whole_scope = ctx.searcher.search(
        "alpha", k=2, candidates=5, paper="paper-a", per_paper=False, rerank=False
    )
    per_paper = ctx.searcher.search(
        "alpha", k=2, candidates=5, paper="paper-a", per_paper=True, rerank=False
    )

    assert [(r.paper_id, r.score) for r in whole_scope] == [
        (r.paper_id, r.score) for r in per_paper
    ]


class _CrowdingEmbedder:
    """paper-a's two chunks both outrank paper-b's one chunk by raw cosine score — proves
    per_paper's k-floor still surfaces paper-b's chunk when the nominal `candidates` budget
    is small enough for paper-a's chunk count alone to exhaust it. A whole-scope search of
    the same depth doesn't just rank paper-b's chunk lower — it never fetches it at all,
    and even returns fewer than k results."""

    _VECS = {
        "orig": [1.0, 0.0],
        "a1_text": [1.0, 0.0],  # cos(orig) = 1.0
        "a2_text": [4.0, 1.0],  # cos(orig) = 4/sqrt(17) ~= 0.970
        "b1_text": [1.0, 1.0],  # cos(orig) = 1/sqrt(2) ~= 0.707 (lowest of the three)
    }

    def name(self) -> str:
        return "crowding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def test_per_paper_prevents_one_paper_starving_another(make_config):
    cfg = make_config()
    embedder = _CrowdingEmbedder()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [
        ("a1", "a1_text", "paper-a", "S1"),
        ("a2", "a2_text", "paper-a", "S2"),
        ("b1", "b1_text", "paper-b", "S1"),
    ]
    metas = [
        {
            "paper_id": paper_id,
            "breadcrumb": f"Paper > {section}",
            "section_title": section,
            "section_number": "1",
            "body": text,
        }
        for _, text, paper_id, section in docs
    ]
    collection.upsert(
        ids=[doc_id for doc_id, _, _, _ in docs],
        embeddings=embedder([text for _, text, _, _ in docs]),
        documents=[text for _, text, _, _ in docs],
        metadatas=metas,
    )
    searcher = Searcher(db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder)

    whole_scope = searcher.search(
        "orig", k=3, candidates=2, paper_ids=["paper-a", "paper-b"], rerank=False
    )
    assert "paper-b" not in [r.paper_id for r in whole_scope]
    assert len(whole_scope) == 2  # under-filled: candidates=2 capped it below k=3

    per_paper = searcher.search(
        "orig",
        k=3,
        candidates=2,
        paper_ids=["paper-a", "paper-b"],
        per_paper=True,
        rerank=False,
    )
    assert {r.paper_id for r in per_paper} == {"paper-a", "paper-b"}
    assert len(per_paper) == 3


class _UnsortedPoolEmbedder:
    """paper-a's only chunk scores lower than paper-b's, but paper-a is fetched first (it's
    earlier in paper_ids) — pools in ascending-score order unless per_paper explicitly
    re-sorts before returning."""

    _VECS = {
        "orig": [1.0, 0.0],
        "a1_text": [0.5, 0.5],  # cos(orig) ~= 0.707
        "b1_text": [1.0, 0.0],  # cos(orig) = 1.0
    }

    def name(self) -> str:
        return "unsorted-pool"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def _seed_unsorted_pool(cfg, embedder):
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [("a1", "a1_text", "paper-a"), ("b1", "b1_text", "paper-b")]
    metas = [
        {
            "paper_id": paper_id,
            "breadcrumb": "Paper > S1",
            "section_title": "S1",
            "section_number": "1",
            "body": text,
        }
        for _, text, paper_id in docs
    ]
    collection.upsert(
        ids=[doc_id for doc_id, _, _ in docs],
        embeddings=embedder([text for _, text, _ in docs]),
        documents=[text for _, text, _ in docs],
        metadatas=metas,
    )


def test_per_paper_sorts_pooled_results_by_score_when_rerank_false(make_config):
    cfg = make_config()
    embedder = _UnsortedPoolEmbedder()
    _seed_unsorted_pool(cfg, embedder)
    searcher = Searcher(db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder)

    results = searcher.search(
        "orig",
        k=2,
        candidates=2,
        paper_ids=["paper-a", "paper-b"],
        per_paper=True,
        rerank=False,
    )

    assert [r.paper_id for r in results] == ["paper-b", "paper-a"]


def test_per_paper_sorts_pooled_results_by_score_when_reranker_fails(monkeypatch, make_config):
    monkeypatch.setattr("rag.search.CrossEncoderReranker", _RaisingCrossEncoderReranker)
    cfg = make_config()
    embedder = _UnsortedPoolEmbedder()
    _seed_unsorted_pool(cfg, embedder)
    searcher = Searcher(db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder)

    no_rerank = searcher.search(
        "orig", k=2, candidates=2, paper_ids=["paper-a", "paper-b"], per_paper=True, rerank=False
    )
    reranked = searcher.search(
        "orig", k=2, candidates=2, paper_ids=["paper-a", "paper-b"], per_paper=True, rerank=True
    )

    assert [r.paper_id for r in reranked] == [r.paper_id for r in no_rerank]


def test_per_paper_generates_paraphrases_once_not_per_paper(
    make_searcher, seed_chunks, fake_embedder, fake_llm
):
    docs = [
        seed_chunks("paper-a", "S", "alpha content"),
        seed_chunks("paper-b", "S", "beta content"),
        seed_chunks("paper-c", "S", "gamma content"),
    ]
    ctx = make_searcher(docs)
    llm = fake_llm(answer='["alternate phrasing"]')
    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=fake_embedder,
        multi_query_enabled=True,
        multi_query_n=1,
        llm=llm,
    )

    searcher.search(
        "alpha",
        k=1,
        candidates=3,
        paper_ids=["paper-a", "paper-b", "paper-c"],
        per_paper=True,
        rerank=False,
    )

    # One paraphrase call for the whole search() call, not one per paper.
    assert len(llm.complete_calls) == 1


def test_per_paper_composes_with_hybrid_and_multi_query(make_config, fake_llm):
    embedder = _FlatEmbedder()
    cfg = make_config()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name="flat")
    docs = [
        ("p1-chunk", "orig appears in this passage", "paper-p1"),
        ("p2-chunk", "zzflorble appears in this passage", "paper-p2"),
        # A third, out-of-scope doc so BM25's corpus isn't degenerate: with only 2 docs,
        # a term appearing in exactly one of them gets IDF log((2-1+.5)/(1+.5)) == log(1) ==
        # 0 exactly (BM25Okapi's unsmoothed IDF), so it would never rank above a zero score
        # no matter what. Not included in paper_ids below, so it never enters the search scope.
        ("filler-chunk", "neutral text sharing no terms with either query", "paper-filler"),
    ]
    metas = [
        {
            "paper_id": paper_id,
            "breadcrumb": "Paper > S1",
            "section_title": "S1",
            "section_number": "1",
            "body": text,
        }
        for _, text, paper_id in docs
    ]
    collection.upsert(
        ids=[doc_id for doc_id, _, _ in docs],
        embeddings=embedder([text for _, text, _ in docs]),
        documents=[text for _, text, _ in docs],
        metadatas=metas,
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

    results = searcher.search(
        "orig",
        k=2,
        candidates=2,
        paper_ids=["paper-p1", "paper-p2"],
        per_paper=True,
        rerank=False,
    )

    by_paper = {r.paper_id: r.source for r in results}
    assert set(by_paper) == {"paper-p1", "paper-p2"}
    assert by_paper["paper-p1"] == "both"  # dense (flat embedder) + BM25 on "orig"
    assert by_paper["paper-p2"] == "both"  # dense (flat embedder) + BM25 on the paraphrase
