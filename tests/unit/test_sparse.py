"""BM25 index, RRF fusion — pure Python except the staleness test (a real temp Chroma)."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from rag.sparse import BM25Index, build_sparse_index, reciprocal_rank_fusion, rrf_scores


def _bm25_index(docs: dict[str, str], paper_ids: dict[str, str] | None = None) -> BM25Index:
    from rag.sparse import _tokenize

    ids = list(docs)
    bm25 = BM25Okapi([_tokenize(docs[i]) for i in ids])
    pids = [paper_ids[i] for i in ids] if paper_ids else ["p"] * len(ids)
    return BM25Index(ids=ids, paper_ids=pids, built_at_count=len(ids), bm25=bm25)


def test_reciprocal_rank_fusion_orders_by_fused_score():
    # "a" ranks first in both lists -> highest fused score; "d" appears only in the second
    # ranking and should still surface, just lower than ids present in both.
    dense = ["a", "b", "c"]
    sparse = ["a", "d", "b"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused.index("b") < fused.index("c")  # "b" ranks in both lists, "c" only in dense


def test_reciprocal_rank_fusion_weights_bias_toward_the_weighted_ranking():
    dense = ["a", "b"]
    sparse = ["b", "a"]
    # Heavily weighting the sparse ranking flips who wins vs. the uniform-weight default.
    uniform = reciprocal_rank_fusion([dense, sparse], k=60)
    weighted = reciprocal_rank_fusion([dense, sparse], k=60, weights=[0.0, 1.0])
    assert uniform[0] == "a"  # tie broken by insertion order at equal weight
    assert weighted[0] == "b"  # sparse ranking alone decides when dense is zeroed out


def test_rrf_scores_matches_reciprocal_rank_fusion_order():
    rankings = [["a", "b", "c"], ["c", "a"]]
    scores = rrf_scores(rankings, k=10)
    fused = reciprocal_rank_fusion(rankings, k=10)
    assert fused == sorted(scores, key=lambda cid: scores[cid], reverse=True)


def test_bm25index_search_ranks_lexical_overlap_first():
    # 3+ docs: with only 2, a term in exactly one of them always gets idf=0 (log(1.5)-log(1.5)),
    # a degenerate BM25 property of tiny corpora, not something worth special-casing around.
    idx = _bm25_index(
        {
            "kv": "multi head latent attention shrinks the kv cache",
            "rl": "reinforcement learning from human feedback recipe",
            "fp8": "fp8 mixed precision training schedule",
        }
    )
    assert idx.search("latent attention kv cache", n=3) == ["kv"]


def test_bm25index_search_filters_zero_score_docs():
    # Regression: a query with fewer true lexical matches than `n` must not pad the result
    # with docs that share no vocabulary with it (score 0) just because n allows more slots.
    idx = _bm25_index(
        {
            "a": "the quick brown fox jumps over the lazy dog",
            "b": "machine learning models train on large datasets",
            "c": "completely unrelated text about gardening and plants",
        }
    )
    assert idx.search("fox dog", n=3) == ["a"]


def test_bm25index_search_respects_allowed_ids():
    # Regression: allowed_ids filters by paper_id, not by chunk id — chunk-a and chunk-b
    # both match the query and would both survive a (buggy) chunk-id-keyed filter that
    # happened to include "chunk-b", but only chunk-b's paper is actually allowed here.
    idx = _bm25_index(
        {
            "chunk-a": "latent attention over the kv cache",
            "chunk-b": "latent attention in a different paper",
            "chunk-c": "fp8 mixed precision training schedule",
        },
        paper_ids={"chunk-a": "paper-1", "chunk-b": "paper-2", "chunk-c": "paper-3"},
    )
    assert idx.search("latent attention", n=3, allowed_ids={"paper-2"}) == ["chunk-b"]


def test_build_sparse_index_detects_staleness(make_searcher, seed_chunks):
    ctx = make_searcher(
        [
            seed_chunks("p1", "Method", "latent attention shrinks the kv cache"),
            seed_chunks("p1", "Training", "reinforcement learning recipe"),
        ]
    )
    idx = build_sparse_index(ctx.collection)
    assert idx.built_at_count == ctx.collection.count() == 2

    ctx.collection.upsert(
        ids=["p1-extra"],
        embeddings=[[0.0] * 32],
        documents=["a brand new chunk of text"],
        metadatas=[
            {
                "paper_id": "p1",
                "paper_title": "Paper",
                "section_number": "2",
                "section_title": "Extra",
                "breadcrumb": "Paper > Extra",
                "body": "a brand new chunk of text",
                "part": 0,
                "n_parts": 1,
            }
        ],
    )
    assert idx.built_at_count != ctx.collection.count()
