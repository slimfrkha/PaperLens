"""Harness: a single-config run end-to-end over a real temp Chroma (fake embedder).

No reranker (make_config disables it), no network — the dense path is exercised against a
seeded collection, and the report's two metrics are asserted from a known layout.
"""

from __future__ import annotations

import json

import pytest

from eval.checkpoint import resume_units
from eval.harness import run, score_items
from eval.queryset import QAItem, load_queryset
from rag.search import Searcher


class _FakeReranker:
    """Deterministic no-model reranker for exercising the rerank=True path offline."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [1.0] * len(docs)


def _item(query: str, paper_id: str, section_title: str) -> QAItem:
    # section_number "1" matches seed_chunks' metadata; gold_span is unused by scoring
    # (relevance is section identity), so a placeholder is fine.
    return QAItem(
        query=query,
        paper_id=paper_id,
        gold_span=(0, 1),
        source_unit=f"1 {section_title}",
        section_number="1",
        section_title=section_title,
    )


def test_run_scores_a_seeded_pool(make_searcher, make_config, seed_chunks):
    docs = [
        seed_chunks("p1", "Method", "latent attention compresses the cache", doc_id="p1-method"),
        seed_chunks("p1", "Training", "fp8 mixed precision schedule warmup", doc_id="p1-train"),
        seed_chunks("p2", "Results", "benchmark scores accuracy evaluation", doc_id="p2-results"),
    ]
    ctx = make_searcher(docs)
    items = [
        _item("latent attention compresses the cache", "p1", "Method"),
        _item("fp8 mixed precision schedule warmup", "p1", "Training"),
        _item("benchmark scores accuracy evaluation", "p2", "Results"),
    ]
    report = run(make_config(), items, searcher=ctx.searcher)

    assert report.n_queries == 3
    assert report.rerank is False  # make_config disables the reranker
    assert report.n_ungoldable == 0
    # Every query's gold-section chunk is the closest bag-of-words match → in-pool + rank 1.
    assert report.success_at_candidates == 1.0
    assert report.n_conditioned == 3
    assert report.mrr_at_k == 1.0


def test_run_excludes_ungoldable_and_charges_a_stage1_miss_to_recall_not_mrr(
    make_searcher, make_config, seed_chunks
):
    docs = [seed_chunks("p1", "Method", "latent attention cache", doc_id="p1-method")]
    ctx = make_searcher(docs)
    items = [
        _item("latent attention cache", "p1", "Method"),  # goldable hit
        _item("unrelated words nobody indexed", "p1", "Ghost"),  # gold section absent → ungoldable
    ]
    report = run(make_config(), items, searcher=ctx.searcher)

    assert report.n_ungoldable == 1  # the Ghost section produced no chunks
    assert report.success_at_candidates == 1.0  # 1 hit over 1 goldable — ungoldable excluded
    assert report.n_conditioned == 1  # MRR averages over the reachable query only
    assert report.mrr_at_k == 1.0


def test_run_overrides_win_over_cfg_and_omitting_them_reproduces_cfg_behavior(
    make_searcher, make_config, seed_chunks
):
    docs = [
        seed_chunks("p1", "Method", "latent attention compresses the cache", doc_id="p1-method"),
        seed_chunks("p1", "Training", "fp8 mixed precision schedule warmup", doc_id="p1-train"),
    ]
    ctx = make_searcher(docs)
    items = [_item("latent attention compresses the cache", "p1", "Method")]
    cfg = make_config()  # retrieval.candidates=20, retrieval.k=5, reranker disabled by default

    default_report = run(cfg, items, searcher=ctx.searcher)
    assert default_report.candidates == cfg.retrieval.candidates
    assert default_report.k == cfg.retrieval.k
    assert default_report.rerank == cfg.reranker.enabled

    ctx.searcher._reranker = _FakeReranker()  # avoid loading a real cross-encoder for rerank=True
    override_report = run(cfg, items, searcher=ctx.searcher, candidates=1, k=1, rerank=True)
    assert override_report.candidates == 1
    assert override_report.k == 1
    assert override_report.rerank is True


def test_retrieve_respects_searcher_sparse_enabled(make_config, seed_chunks):
    # Regression: _retrieve used to bypass Searcher.sparse_enabled entirely (dense-only always),
    # so `run`/`confirm` silently ignored a config's `sparse.enabled: true`. A deterministic
    # embedder (FakeEmbedder is itself lexical — shared words land close — which makes "dense
    # misses it" hard to construct reliably; see tests/integration/test_search.py) decouples
    # dense closeness from BM25-relevant content, so this isolates the wiring being tested.
    from rag.index import open_collection

    class _DirectionalEmbedder:
        def _vec(self, text: str) -> list[float]:
            return [0.0, 1.0] if "FAR_MARKER" in text else [1.0, 0.0]

        def __call__(self, input: list[str]) -> list[list[float]]:
            return [self._vec(t) for t in input]

    embedder = _DirectionalEmbedder()
    cfg = make_config()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name="directional")
    docs = [
        seed_chunks("target", "Method", "FAR_MARKER zzflorble appears once here", doc_id="target"),
        seed_chunks("close-0", "Other", "totally unrelated content number 0", doc_id="close-0"),
        seed_chunks("close-1", "Other", "totally unrelated content number 1", doc_id="close-1"),
    ]
    texts = [text for _, text, _ in docs]
    collection.upsert(
        ids=[doc_id for doc_id, _, _ in docs],
        embeddings=embedder(texts),
        documents=texts,
        metadatas=[meta for _, _, meta in docs],
    )
    items = [_item("zzflorble", "target", "Method")]

    dense_only = Searcher(
        db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder, sparse_enabled=False
    )
    assert (
        score_items(dense_only, items, candidates=1, k=1, rerank=False)[0].ranked[0][0] != "target"
    )

    hybrid = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        sparse_enabled=True,
        fetch_multiplier=3,
    )
    assert score_items(hybrid, items, candidates=1, k=1, rerank=False)[0].ranked[0][0] == "target"


def test_score_items_pairs_each_query_with_its_relevant_set(
    make_searcher, make_config, seed_chunks
):
    docs = [
        seed_chunks("p1", "Method", "alpha beta gamma", doc_id="p1-method-0"),
        seed_chunks("p1", "Method", "alpha beta delta", doc_id="p1-method-1"),
    ]
    ctx = make_searcher(docs)
    scores = score_items(
        ctx.searcher, [_item("alpha beta gamma", "p1", "Method")], candidates=20, k=5, rerank=False
    )
    assert scores[0].relevant_ids == {"p1-method-0", "p1-method-1"}
    assert scores[0].gold_in_pool


class _CountingEmbedder:
    """Wraps a real embedder, counting calls and (optionally) raising past ``fail_after`` —
    simulates a crash partway through a per-query loop. ``fail_after=None`` lets everything
    through, so the same instance can be reused across an interrupted-then-resumed pair of
    calls to assert the total call count over *both* invocations, not just the second one.
    """

    def __init__(self, embedder, fail_after: int | None) -> None:
        self._embedder = embedder
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, input: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        return self._embedder(input)


def test_score_items_resumes_without_recomputing_already_scored_queries(
    make_searcher, fake_embedder, seed_chunks, tmp_path
):
    docs = [
        seed_chunks("p1", "Method", "alpha beta gamma", doc_id="p1-method"),
        seed_chunks("p2", "Training", "delta epsilon zeta", doc_id="p2-train"),
        seed_chunks("p3", "Results", "eta theta iota", doc_id="p3-results"),
    ]
    ctx = make_searcher(docs)
    items = [
        _item("alpha beta gamma", "p1", "Method"),
        _item("delta epsilon zeta", "p2", "Training"),
        _item("eta theta iota", "p3", "Results"),
    ]
    counting = _CountingEmbedder(fake_embedder, fail_after=2)
    ctx.searcher.embedder = counting
    ckpt = tmp_path / "run.ckpt.jsonl"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        score_items(ctx.searcher, items, candidates=20, k=5, rerank=False, checkpoint_path=ckpt)
    assert ckpt.exists()
    header = {
        "candidates": 20,
        "k": 5,
        "rerank": False,
        "index_count": ctx.searcher.collection.count(),
    }
    assert len(resume_units(ckpt, header)) == 2  # 2 queries completed before the 3rd raised
    calls_before_resume = counting.calls  # 2 succeeded + 1 failed attempt

    counting.fail_after = None  # let the interrupted run "come back up"
    scores = score_items(
        ctx.searcher, items, candidates=20, k=5, rerank=False, checkpoint_path=ckpt
    )

    assert len(scores) == 3
    # Only the one un-cached item is (re-)embedded on resume — the 2 already-checkpointed
    # queries are never touched again.
    assert counting.calls - calls_before_resume == 1
    assert not ckpt.exists()  # deleted once every unit is accounted for


def test_run_resume_reproduces_the_uninterrupted_report_exactly(
    make_searcher, make_config, fake_embedder, seed_chunks, tmp_path
):
    """Resume isn't just faster — cluster_bootstrap is seeded (stats.py), so the resumed
    report must equal, field for field, what an uninterrupted run would have produced."""
    docs = [
        seed_chunks("p1", "Method", "alpha beta gamma", doc_id="p1-method"),
        seed_chunks("p2", "Training", "delta epsilon zeta", doc_id="p2-train"),
        seed_chunks("p3", "Results", "eta theta iota", doc_id="p3-results"),
    ]
    items = [
        _item("alpha beta gamma", "p1", "Method"),
        _item("delta epsilon zeta", "p2", "Training"),
        _item("eta theta iota", "p3", "Results"),
    ]
    cfg = make_config()

    baseline_ctx = make_searcher(docs)
    baseline = run(cfg, items, searcher=baseline_ctx.searcher)

    resumed_ctx = make_searcher(docs)
    counting = _CountingEmbedder(fake_embedder, fail_after=2)
    resumed_ctx.searcher.embedder = counting
    ckpt = tmp_path / "run.ckpt.jsonl"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(cfg, items, searcher=resumed_ctx.searcher, checkpoint_path=ckpt)
    calls_before_resume = counting.calls
    counting.fail_after = None
    resumed = run(cfg, items, searcher=resumed_ctx.searcher, checkpoint_path=ckpt)

    assert resumed == baseline
    assert counting.calls - calls_before_resume == 1  # only the un-cached item re-embedded


def test_load_queryset_round_trips(tmp_path):
    it = QAItem(
        query="q",
        paper_id="p1",
        gold_span=(3, 9),
        source_unit="2 Method",
        section_number="2",
        section_title="Method",
        answer="a",
    )
    path = tmp_path / "fp.dev.jsonl"
    from eval.queryset import item_to_dict

    path.write_text(json.dumps(item_to_dict(it)) + "\n", encoding="utf-8")
    loaded = load_queryset(str(path))
    assert loaded == [it]
    assert loaded[0].gold_span == (3, 9)  # list -> tuple on load


def test_load_queryset_rejects_sets_missing_section_identity(tmp_path):
    # A row without section_title predates section-identity scoring; must fail loudly.
    path = tmp_path / "fp.dev.jsonl"
    path.write_text(json.dumps({"query": "q", "paper_id": "p", "gold_span": [0, 1]}) + "\n")
    with pytest.raises(SystemExit, match="regenerate"):
        load_queryset(str(path))
