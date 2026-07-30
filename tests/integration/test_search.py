"""Retrieval over a real temp Chroma DB with a fake embedder (rerank disabled)."""

from __future__ import annotations

from rag.index import open_collection
from rag.reranker import Reranker
from rag.search import Searcher
from rag.sparse import BM25Index


def _docs(seed_chunks):
    return [
        seed_chunks("paper-a", "Attention", "multi head latent attention shrinks kv cache"),
        seed_chunks("paper-b", "Training", "reinforcement learning from human feedback recipe"),
    ]


def test_retrieves_the_relevant_passage_first(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    results = ctx.searcher.search("latent attention kv cache", k=2, candidates=10, rerank=False)
    assert results
    assert results[0].paper_id == "paper-a"
    assert "latent attention" in results[0].body


def test_paper_filter_restricts_to_one_paper(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    results = ctx.searcher.search(
        "reinforcement learning", k=5, candidates=10, paper="paper-b", rerank=False
    )
    assert results
    assert {r.paper_id for r in results} == {"paper-b"}


def test_empty_paper_ids_matches_nothing(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    assert ctx.searcher.search("anything", paper_ids=[], rerank=False) == []


def test_paper_and_paper_ids_intersect(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    # `paper` not in the tag-derived `paper_ids` -> empty intersection -> no hits.
    out = ctx.searcher.search("attention", paper="paper-a", paper_ids=["paper-b"], rerank=False)
    assert out == []


def test_result_body_strips_breadcrumb_prefix(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    r = ctx.searcher.search("latent attention", candidates=10, rerank=False)[0]
    assert not r.body.startswith("Paper >")
    assert r.breadcrumb.startswith("Paper >")


def test_result_body_returns_full_multiparagraph_body(make_searcher, seed_chunks):
    # Body is read back from metadata, not re-split from the embedded doc, so a body
    # that itself contains a blank line comes back whole and breadcrumb-free — the
    # edge the old doc.split("\n\n", 1) contract fumbled.
    body = "para one about latent attention\n\npara two continues"
    ctx = make_searcher([seed_chunks("p", "Attention", body)])
    r = ctx.searcher.search("latent attention", candidates=10, rerank=False)[0]
    assert r.body == body
    assert "Paper >" not in r.body


class _KeywordReranker(Reranker):
    """Scores a passage 1.0 iff it contains ``needle``, else 0.0."""

    def __init__(self, needle: str):
        self.needle = needle

    def score(self, query, docs):
        return [1.0 if self.needle in d else 0.0 for d in docs]

    @classmethod
    def build(cls, model, *, device, llm):
        return cls("")


class _RaisingReranker(Reranker):
    """Fails every call, to exercise Searcher.search's fallback."""

    def score(self, query, docs):
        raise RuntimeError("boom")


class _ShortReranker(Reranker):
    """Returns one score short of what's needed — a malformed-output contract violation,
    distinct from an outright raise (proves the length check catches it too)."""

    def score(self, query, docs):
        return [1.0] * (len(docs) - 1)


def test_rerank_uses_injected_reranker(make_searcher, fake_embedder, seed_chunks):
    # An injected reranker reorders results (and proves the seam runs offline —
    # no cross-encoder model is loaded). The passage carrying the needle wins.
    ctx = make_searcher(
        [
            seed_chunks("paper-a", "Attention", "latent attention over the kv cache"),
            seed_chunks("paper-b", "Rewards", "reinforcement learning UNIQUENEEDLE recipe"),
        ]
    )
    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=fake_embedder,
        reranker=_KeywordReranker("UNIQUENEEDLE"),
    )
    results = searcher.search("learning", k=2, candidates=10, rerank=True)
    assert results[0].paper_id == "paper-b"
    assert results[0].score == 1.0
    assert results[0].score >= results[-1].score


def test_rerank_failure_falls_back_to_pre_rerank_order(make_searcher, fake_embedder, seed_chunks):
    # A reranker that raises must not crash the search — it should degrade to the same
    # order (and scores) a dense-only, rerank=False query would have returned.
    ctx = make_searcher(
        [
            seed_chunks("paper-a", "Attention", "latent attention over the kv cache"),
            seed_chunks("paper-b", "Rewards", "reinforcement learning UNIQUENEEDLE recipe"),
        ]
    )
    dense_searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db, collection=ctx.cfg.collection, embedder=fake_embedder
    )
    dense_order = dense_searcher.search("learning", k=2, candidates=10, rerank=False)

    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=fake_embedder,
        reranker=_RaisingReranker(),
    )
    results = searcher.search("learning", k=2, candidates=10, rerank=True)

    assert [r.paper_id for r in results] == [r.paper_id for r in dense_order]
    assert [r.score for r in results] == [r.score for r in dense_order]


def test_rerank_wrong_length_scores_falls_back_to_pre_rerank_order(
    make_searcher, fake_embedder, seed_chunks
):
    # A reranker that returns the wrong number of scores must not partially overwrite
    # Result.score before the fallback kicks in — the whole point of the explicit length
    # check in Searcher.search (a naive zip(strict=True) would raise only after mutating
    # every result but the last, leaving a mix of rerank-scale and pre-rerank-scale scores).
    ctx = make_searcher(
        [
            seed_chunks("paper-a", "Attention", "latent attention over the kv cache"),
            seed_chunks("paper-b", "Rewards", "reinforcement learning UNIQUENEEDLE recipe"),
        ]
    )
    dense_searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db, collection=ctx.cfg.collection, embedder=fake_embedder
    )
    dense_order = dense_searcher.search("learning", k=2, candidates=10, rerank=False)

    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=fake_embedder,
        reranker=_ShortReranker(),
    )
    results = searcher.search("learning", k=2, candidates=10, rerank=True)

    assert [r.paper_id for r in results] == [r.paper_id for r in dense_order]
    assert [r.score for r in results] == [r.score for r in dense_order]


class _DirectionalEmbedder:
    """Deterministic 2D embedder for the hybrid tests below: dense closeness to the query is
    controlled directly by an explicit marker, independent of BM25-relevant word content.

    ``FakeEmbedder`` (the shared hash-based fixture) *is* lexical — texts sharing words land
    closer together — which makes it awkward to construct a case where dense recall reliably
    misses a passage that shares exact words with the query (the whole point of these tests).
    This embedder decouples the two axes on purpose: anything without ``FAR_MARKER`` embeds
    identically to the query (cosine similarity 1.0); anything with it embeds orthogonally
    (cosine similarity 0.0), regardless of what words it contains.
    """

    FAR_MARKER = "FAR_MARKER"

    def _vec(self, text: str) -> list[float]:
        return [0.0, 1.0] if self.FAR_MARKER in text else [1.0, 0.0]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in input]


def _hybrid_pool(make_config, seed_chunks):
    """A pool where ``target`` is dense-worst (unique ``FAR_MARKER``) but lexically the only
    exact match for ``zzflorble`` outside two stronger BM25 competitors (``mid-0``/``mid-1``,
    which repeat the term and are dense-identical to the query) — so ``target`` is reachable
    only through a wide-enough fused pool, not through dense or a narrow BM25 window alone.
    """
    embedder = _DirectionalEmbedder()
    cfg = make_config()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name="directional")
    docs = [
        seed_chunks("target", "S", f"{embedder.FAR_MARKER} zzflorble appears once here"),
        seed_chunks("mid-0", "S", "zzflorble zzflorble zzflorble midtier document"),
        seed_chunks("mid-1", "S", "zzflorble zzflorble zzflorble another midtier document"),
        seed_chunks("dis-0", "S", "totally unrelated content number 0"),
        seed_chunks("dis-1", "S", "totally unrelated content number 1"),
        seed_chunks("dis-2", "S", "totally unrelated content number 2"),
        seed_chunks("dis-3", "S", "totally unrelated content number 3"),
    ]
    texts = [text for _, text, _ in docs]
    collection.upsert(
        ids=[doc_id for doc_id, _, _ in docs],
        embeddings=embedder(texts),
        documents=texts,
        metadatas=[meta for _, _, meta in docs],
    )
    return cfg, collection, embedder


def test_hybrid_surfaces_a_lexical_match_dense_only_misses(make_config, seed_chunks):
    cfg, _collection, embedder = _hybrid_pool(make_config, seed_chunks)
    dense_only = Searcher(
        db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder, sparse_enabled=False
    )
    hybrid = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        sparse_enabled=True,
        fetch_multiplier=3,
    )
    dense_hits = {
        r.paper_id for r in dense_only.search("zzflorble", k=3, candidates=3, rerank=False)
    }
    hybrid_hits = {r.paper_id for r in hybrid.search("zzflorble", k=3, candidates=3, rerank=False)}
    assert "target" not in dense_hits
    assert "target" in hybrid_hits


def test_hybrid_fetch_multiplier_is_load_bearing_not_decorative(make_config, seed_chunks):
    # Regression for the over-fetch margin (hybrid_retrieval_plan.md Revision note 2): at
    # fetch_multiplier=1 the pre-fusion pool is too narrow to reach "target" at all (it's
    # BM25-rank 3, worse than both midtier competitors, and dense-worst by construction);
    # widening to fetch_multiplier=3 gives fusion enough margin to surface it.
    cfg, _collection, embedder = _hybrid_pool(make_config, seed_chunks)

    def hits(fetch_multiplier: int) -> set[str]:
        searcher = Searcher(
            db_dir=cfg.paths.rag_db,
            collection=cfg.collection,
            embedder=embedder,
            sparse_enabled=True,
            fetch_multiplier=fetch_multiplier,
        )
        return {r.paper_id for r in searcher.search("zzflorble", k=3, candidates=3, rerank=False)}

    assert "target" not in hits(fetch_multiplier=1)
    assert "target" in hits(fetch_multiplier=3)


def test_sparse_index_rebuilds_after_rescan(make_searcher, seed_chunks):
    ctx = make_searcher(
        [seed_chunks("paper-a", "Attention", "multi head latent attention shrinks kv cache")]
    )
    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=ctx.searcher.embedder,
        sparse_enabled=True,
    )
    # First search builds the BM25 snapshot over the 1-paper pool. Dense recall has no
    # relevance threshold, so it still returns the one (unrelated) chunk it has — the point is
    # that nothing has ever seen "zzflorble" lexically yet.
    before = searcher.search("zzflorble", k=1, candidates=1, rerank=False)
    assert before[0].paper_id == "paper-a"

    # Simulate a live rescan (POST /api/admin/rescan) upserting a new paper's chunks directly
    # into the same collection the Searcher already has open.
    ctx.collection.upsert(
        ids=["new-paper-chunk"],
        embeddings=ctx.searcher.embedder(["zzflorble is a rare made-up term"]),
        documents=["zzflorble is a rare made-up term"],
        metadatas=[
            {
                "paper_id": "new-paper",
                "paper_title": "New Paper",
                "section_number": "1",
                "section_title": "Intro",
                "breadcrumb": "New Paper > Intro",
                "body": "zzflorble is a rare made-up term",
                "part": 0,
                "n_parts": 1,
            }
        ],
    )

    # Without a restart, the next search must promote the rescanned chunk above the old
    # dense-only fallback — proving the lazy `sparse` property re-checks staleness rather than
    # caching the pre-rescan snapshot forever.
    after = searcher.search("zzflorble", k=2, candidates=2, rerank=False)
    assert after[0].paper_id == "new-paper"


def test_dense_only_results_are_labeled_dense(make_searcher, seed_chunks):
    ctx = make_searcher(_docs(seed_chunks))
    results = ctx.searcher.search("latent attention kv cache", k=2, candidates=10, rerank=False)
    assert results
    assert all(r.source == "dense" for r in results)


class _FakeSparseIndex(BM25Index):
    """A ``BM25Index`` that returns a canned id list instead of running real BM25 scoring — the
    sparse-side counterpart to ``_KeywordReranker`` above. Gives exact control over which ids
    the sparse pool "finds", so `Result.source` tests don't have to tune real BM25/embedding
    geometry to force a particular dense/sparse split (see the fetch_multiplier regression test
    above for how fiddly that gets).
    """

    def __init__(self, ids: list[str], built_at_count: int):
        # built_at_count must match the real collection's count, or Searcher.sparse's
        # staleness check (built_at_count != collection.count()) silently discards this fake
        # and rebuilds a real BM25Index in its place.
        super().__init__(ids=[], paper_ids=[], built_at_count=built_at_count, bm25=None)
        self._canned = ids

    def search(self, query: str, n: int, allowed_ids: set[str] | None = None) -> list[str]:
        ids = self._canned
        if allowed_ids is not None:
            ids = [cid for cid in ids if cid in allowed_ids]
        return ids[:n]


def test_hybrid_source_is_both_when_a_hit_is_in_dense_and_sparse_pools(make_searcher, seed_chunks):
    # Both docs fit comfortably inside the dense fetch window (candidates=2, corpus=2), so no
    # exclusion is needed here — only pool membership (dense vs the fake sparse pool) matters.
    ctx = make_searcher(
        [
            seed_chunks("dense-hit", "S", "alpha content"),
            seed_chunks("both-hit", "S", "beta content"),
        ]
    )
    searcher = Searcher(
        db_dir=ctx.cfg.paths.rag_db,
        collection=ctx.cfg.collection,
        embedder=ctx.searcher.embedder,
        sparse_enabled=True,
        # chunk id, per seed_chunks: f"{paper_id}-{section}"; built_at_count=2 matches the pool.
        sparse=_FakeSparseIndex(["both-hit-S"], built_at_count=2),
        fetch_multiplier=1,
    )
    results = {
        r.paper_id: r for r in searcher.search("alpha beta", k=2, candidates=2, rerank=False)
    }
    assert results["dense-hit"].source == "dense"
    assert results["both-hit"].source == "both"


class _TieredEmbedder:
    """Three strictly distinct cosine similarities to the query (1.0 / 0.6 / 0.0, tagged
    "high"/"mid"/"low" in the doc text) — unlike `_DirectionalEmbedder`'s binary near/far split,
    nothing here ties, so dense recall's top-N is unambiguous regardless of `candidates`."""

    VECS = {"mid": [0.6, 0.8], "low": [0.0, 1.0]}

    def _vec(self, text: str) -> list[float]:
        for tag, v in self.VECS.items():
            if tag in text:
                return v
        return [1.0, 0.0]  # "high" (and the query itself, which carries no tag)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in input]


def test_hybrid_source_is_sparse_when_a_hit_is_excluded_from_the_dense_fetch_window(
    make_config, seed_chunks
):
    # A third ("mid") doc pads the corpus past the dense fetch window (candidates=2,
    # fetch_multiplier=1 -> fetch_n=2 of 3 docs) so "low" is deterministically excluded from
    # `dense_ids` by construction, not by a Chroma-internal tie-break — see the class docstring
    # above for why a 2-doc pool can't isolate this case (fetch_n and the final candidates
    # truncation share the same `candidates` value).
    embedder = _TieredEmbedder()
    cfg = make_config()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name="tiered")
    docs = [
        seed_chunks("dense-hit", "S", "high tier document"),
        seed_chunks("filler", "S", "mid tier document"),
        seed_chunks("sparse-only", "S", "low tier document"),
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
        # chunk id: f"{paper_id}-{section}"; built_at_count=3 matches the 3-doc pool.
        sparse=_FakeSparseIndex(["sparse-only-S"], built_at_count=3),
        fetch_multiplier=1,
    )
    results = {r.paper_id: r for r in searcher.search("query", k=2, candidates=2, rerank=False)}
    assert results["dense-hit"].source == "dense"
    assert results["sparse-only"].source == "sparse"
