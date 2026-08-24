"""Harness: a single-config run end-to-end over a real temp Chroma (fake embedder).

No reranker (make_config disables it), no network — the dense path is exercised against a
seeded collection, and the report's two metrics are asserted from a known layout.
"""

from __future__ import annotations

import json

import pytest

from eval.checkpoint import resume_units
from eval.comparative_queryset import ComparativeQAItem
from eval.harness import (
    COMPARATIVE_MIN_CLUSTERS,
    ComparativeArmResult,
    ComparativeConfirmResult,
    build_per_paper_scopes,
    comparative_confirm,
    comparative_sweep,
    format_comparative_confirm,
    format_comparative_sweep,
    format_elbow_screen_report,
    format_per_paper_confirm,
    format_per_paper_sweep,
    per_paper_confirm,
    per_paper_sweep,
    run,
    score_comparative_items_scoped,
    score_items,
    score_items_scoped,
    screen_elbow,
)
from eval.queryset import QAItem, Section, load_queryset
from eval.stats import BootResult, DeltaResult
from rag.index import open_collection
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


def test_run_elbow_metrics_are_none_when_not_reranked(make_searcher, make_config, seed_chunks):
    docs = [seed_chunks("p1", "Method", "latent attention cache", doc_id="p1-method")]
    ctx = make_searcher(docs)
    items = [_item("latent attention cache", "p1", "Method")]
    report = run(make_config(), items, searcher=ctx.searcher)  # rerank=False (make_config default)

    assert report.mean_returned_at_elbow is None
    assert report.precision_at_elbow is None
    assert report.recall_at_elbow is None


def test_run_elbow_metrics_populated_when_reranked(make_searcher, make_config, seed_chunks):
    # additive: never disturbs success_at_candidates/mrr_at_k, computed alongside them.
    docs = [
        seed_chunks("p1", "Method", "latent attention compresses the cache", doc_id="p1-method"),
        seed_chunks("p1", "Training", "fp8 mixed precision schedule warmup", doc_id="p1-train"),
        seed_chunks("p2", "Results", "benchmark scores accuracy evaluation", doc_id="p2-results"),
    ]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = _FakeReranker()  # avoid loading a real cross-encoder
    items = [_item("latent attention compresses the cache", "p1", "Method")]
    report = run(make_config(), items, searcher=ctx.searcher, rerank=True)

    assert report.success_at_candidates == 1.0  # unaffected by the elbow metrics below
    assert report.mrr_at_k == 1.0
    # _FakeReranker ties every score at 1.0 -> flat window, no real cliff -> "no_elbow",
    # cutoff = whole 3-chunk pool.
    assert report.mean_returned_at_elbow == 3.0
    assert report.recall_at_elbow == 1.0
    assert report.precision_at_elbow == pytest.approx(1 / 3)


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
    cfg = make_config()  # retrieval.candidates=20, retrieval.max_k=10, reranker disabled by default

    default_report = run(cfg, items, searcher=ctx.searcher)
    assert default_report.candidates == cfg.retrieval.candidates
    assert default_report.k == cfg.retrieval.max_k
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


def test_screen_elbow_reports_default_plus_grid_arms_with_paired_deltas(
    make_searcher, make_config, seed_chunks
):
    docs = [
        seed_chunks("p1", "Method", "latent attention compresses the cache", doc_id="p1-method"),
        seed_chunks("p1", "Training", "fp8 mixed precision schedule warmup", doc_id="p1-train"),
        seed_chunks("p2", "Results", "benchmark scores accuracy evaluation", doc_id="p2-results"),
    ]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = _FakeReranker()
    items = [
        _item("latent attention compresses the cache", "p1", "Method"),
        _item("fp8 mixed precision schedule warmup", "p1", "Training"),
    ]
    cfg = make_config()

    report = screen_elbow(cfg, items, searcher=ctx.searcher, mad_grid=[1.5, 3.0])

    labels = [r.label for r in report.results]
    assert labels[0] == "default"
    assert "mad_multiplier=1.5" in labels  # 3.0 (the cfg default) is skipped, not duplicated
    default_result = report.results[0]
    assert default_result.recall_delta is None  # the arm being compared against itself
    non_default = report.results[1]
    assert non_default.recall_delta is not None
    # score_items only ran once (one dense query + one rerank pass per item, shared by
    # every arm) — this is the whole point of screen_elbow being cache-free-but-cheap.
    assert report.n_queries == 2


def test_format_elbow_screen_report_is_readable(make_searcher, make_config, seed_chunks):
    docs = [seed_chunks("p1", "Method", "latent attention compresses the cache")]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = _FakeReranker()
    items = [_item("latent attention compresses the cache", "p1", "Method")]
    report = screen_elbow(make_config(), items, searcher=ctx.searcher, mad_grid=[1.5])

    text = format_elbow_screen_report(report)
    assert "default" in text
    assert "mad_multiplier=1.5" in text


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


# --- per-paper: build_per_paper_scopes / per_paper_sweep / per_paper_confirm ---------------


def test_build_per_paper_scopes_includes_gold_and_is_seed_deterministic():
    items = [_item("q1", "p1", "S"), _item("q2", "p2", "S")]
    pool = {"p1": "md", "p2": "md", "p3": "md", "p4": "md", "p5": "md"}

    scopes_a = build_per_paper_scopes(items, pool, n_papers=3, seed=0)
    scopes_b = build_per_paper_scopes(items, pool, n_papers=3, seed=0)

    assert scopes_a == scopes_b  # deterministic given the same seed
    assert scopes_a["0"][0] == "p1"  # each item's own gold paper is always present
    assert scopes_a["1"][0] == "p2"
    assert all(len(scope) == 3 for scope in scopes_a.values())
    assert all(len(set(scope)) == 3 for scope in scopes_a.values())  # no duplicates


def test_build_per_paper_scopes_different_seeds_draw_different_others():
    items = [_item("q", "p1", "S")]
    pool = {f"p{i}": "md" for i in range(1, 8)}  # p1 (gold) + 7 candidate "others"

    scopes_0 = build_per_paper_scopes(items, pool, n_papers=3, seed=0)
    scopes_1 = build_per_paper_scopes(items, pool, n_papers=3, seed=1)

    assert scopes_0["0"][0] == scopes_1["0"][0] == "p1"
    assert scopes_0 != scopes_1


def test_build_per_paper_scopes_degrades_when_pool_smaller_than_n_papers():
    items = [_item("q", "p1", "S")]
    pool = {"p1": "md", "p2": "md"}
    scopes = build_per_paper_scopes(items, pool, n_papers=10, seed=0)
    assert set(scopes["0"]) == {"p1", "p2"}


class _CrowdingEmbedder:
    """Mirrors test_search.py's _CrowdingEmbedder: paper-a's two chunks both outrank
    paper-b's one chunk, so a shared candidates=2 budget crowds paper-b out entirely
    unless per_paper is on."""

    _VECS = {
        "orig": [1.0, 0.0],
        "a1_text": [1.0, 0.0],
        "a2_text": [4.0, 1.0],
        "b1_text": [1.0, 1.0],
    }

    def name(self) -> str:
        return "pp-crowding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def _seed_crowding_pool(cfg):
    embedder = _CrowdingEmbedder()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [
        ("a1", "a1_text", "paper-a", "S1"),
        ("a2", "a2_text", "paper-a", "S2"),
        ("b1", "b1_text", "paper-b", "S1"),
    ]
    metas = [
        {
            "paper_id": pid,
            "breadcrumb": f"Paper > {s}",
            "section_title": s,
            "section_number": "1",
            "body": t,
        }
        for _, t, pid, s in docs
    ]
    collection.upsert(
        ids=[d for d, _, _, _ in docs],
        embeddings=embedder([t for _, t, _, _ in docs]),
        documents=[t for _, t, _, _ in docs],
        metadatas=metas,
    )
    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        reranker=_FakeReranker(),
    )
    return searcher


def test_per_paper_sweep_recovers_a_crowded_paper(make_config):
    cfg = make_config()
    searcher = _seed_crowding_pool(cfg)
    items = [_item("orig", "paper-b", "S1")]
    pool = {"paper-a": "md", "paper-b": "md"}

    report = per_paper_sweep(
        cfg, items, pool, n_papers=2, candidates_grid=[2], seed=0, searcher=searcher
    )

    by_label = {r.label: r for r in report.results}
    assert by_label["off"].success.point == 0.0  # paper-b's chunk is crowded out
    assert by_label["on (production)"].success.point == 1.0
    # The allocation itself helps, not just extra fetch volume: budget-matched holds the
    # SAME total pool size as the off-arm and still recovers paper-b's chunk.
    assert by_label["on (budget-matched)"].success.point == 1.0
    assert by_label["on (budget-matched)"].mean_pool_size == by_label["off"].mean_pool_size

    text = format_per_paper_sweep(report)
    assert "candidates=2" in text
    assert "Candidate points to confirm" in text


class _EvenlyRelevantEmbedder:
    """Mirrors test_search.py's _EvenlyRelevantEmbedder: two papers, comparable
    relevance, generous budget -- nothing for per_paper to fix."""

    _VECS = {"orig": [1.0, 0.0], "x1_text": [1.0, 0.0], "y1_text": [0.9, 0.1]}

    def name(self) -> str:
        return "pp-evenly-relevant"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def test_per_paper_sweep_no_false_positive_when_not_crowded(make_config):
    cfg = make_config()
    embedder = _EvenlyRelevantEmbedder()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [("x1", "x1_text", "paper-x", "S1"), ("y1", "y1_text", "paper-y", "S1")]
    metas = [
        {
            "paper_id": pid,
            "breadcrumb": f"Paper > {s}",
            "section_title": s,
            "section_number": "1",
            "body": t,
        }
        for _, t, pid, s in docs
    ]
    collection.upsert(
        ids=[d for d, _, _, _ in docs],
        embeddings=embedder([t for _, t, _, _ in docs]),
        documents=[t for _, t, _, _ in docs],
        metadatas=metas,
    )
    searcher = Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        reranker=_FakeReranker(),
    )
    items = [_item("orig", "paper-y", "S1")]
    pool = {"paper-x": "md", "paper-y": "md"}

    # Generous candidates relative to a 2-chunk corpus -- nothing to crowd.
    report = per_paper_sweep(
        cfg, items, pool, n_papers=2, candidates_grid=[10], seed=0, searcher=searcher
    )

    by_label = {r.label: r for r in report.results}
    assert by_label["off"].success.point == 1.0
    assert by_label["on (production)"].success.point == 1.0
    assert by_label["on (production)"].success_delta.delta == 0.0


class _GoldNotHighestScoreEmbedder:
    """Two papers, one chunk each; the NON-gold paper's chunk scores higher for the query
    than the gold paper's own chunk. build_per_paper_scopes always puts the gold paper
    first in the scope list -- regression guard for _retrieve_scoped's rerank=False path,
    where a naive per-scope concatenation would rank paper-a's chunk first purely because
    it's scope[0], not because it's more relevant."""

    _VECS = {"orig": [1.0, 0.0], "a1_text": [0.6, 0.4], "z1_text": [0.95, 0.05]}

    def name(self) -> str:
        return "pp-gold-not-highest"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._VECS[t] for t in input]


def test_score_items_scoped_per_paper_no_rerank_sorts_across_scopes_by_score(make_config):
    cfg = make_config()  # reranker disabled by default -> exercises the rerank=False path
    embedder = _GoldNotHighestScoreEmbedder()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [("a1", "a1_text", "paper-a", "S1"), ("z1", "z1_text", "paper-z", "S1")]
    metas = [
        {
            "paper_id": pid,
            "breadcrumb": f"Paper > {s}",
            "section_title": s,
            "section_number": "1",
            "body": t,
        }
        for _, t, pid, s in docs
    ]
    collection.upsert(
        ids=[d for d, _, _, _ in docs],
        embeddings=embedder([t for _, t, _, _ in docs]),
        documents=[t for _, t, _, _ in docs],
        metadatas=metas,
    )
    searcher = Searcher(db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder)
    items = [_item("orig", "paper-a", "S1")]  # paper-a is gold -> always scopes["0"][0]
    scopes = build_per_paper_scopes(items, {"paper-a": "md", "paper-z": "md"}, n_papers=2, seed=0)
    assert scopes["0"][0] == "paper-a"  # sanity: gold really is scope[0]

    scores = score_items_scoped(
        searcher, items, scopes, per_paper=True, candidates=1, k=1, rerank=False
    )

    # paper-z's chunk scores higher for this query -- it must rank first despite being
    # scope[1], not paper-a's lower-scoring chunk winning by concatenation order.
    assert scores[0].ranked[0][0] == "z1"


def test_per_paper_sweep_and_confirm_respect_cfg_reranker_enabled(monkeypatch, make_config):
    # Regression guard: rerank must come from cfg.reranker.enabled, matching run/
    # screen_retrieval and what ChatAgent._build_search_executor actually calls in
    # production -- not hardcoded True regardless of the loaded config.
    from eval import harness as harness_mod
    from rag.config import HFRerankerCfg

    items = [_item("orig", "paper-b", "S1")]
    pool = {"paper-a": "md", "paper-b": "md"}
    seen_rerank: list[bool] = []
    real_score_items_scoped = harness_mod.score_items_scoped

    def spy(*args, **kwargs):
        seen_rerank.append(kwargs["rerank"])
        return real_score_items_scoped(*args, **kwargs)

    monkeypatch.setattr(harness_mod, "score_items_scoped", spy)

    cfg_off = make_config()  # reranker disabled by default
    searcher_off = _seed_crowding_pool(cfg_off)
    per_paper_sweep(
        cfg_off, items, pool, n_papers=2, candidates_grid=[2], seed=0, searcher=searcher_off
    )
    assert seen_rerank and all(r is False for r in seen_rerank)

    seen_rerank.clear()
    cfg_on = make_config(reranker=HFRerankerCfg(enabled=True))
    searcher_on = _seed_crowding_pool(cfg_on)
    per_paper_confirm(
        cfg_on,
        items,
        pool,
        n_papers=2,
        candidates=2,
        variant="production",
        seed=1,
        searcher=searcher_on,
    )
    assert seen_rerank and all(r is True for r in seen_rerank)


def test_per_paper_confirm_threads_its_own_seed_not_a_hardcoded_one(monkeypatch, make_config):
    # Regression guard: per_paper_confirm must build its scopes from the seed it was
    # actually called with, never a sweep's leftover seed or an internal default.
    from eval import harness as harness_mod

    calls: list[int] = []
    real = harness_mod.build_per_paper_scopes

    def spy(items, pool, n_papers, seed):
        calls.append(seed)
        return real(items, pool, n_papers, seed)

    monkeypatch.setattr(harness_mod, "build_per_paper_scopes", spy)

    cfg = make_config()
    searcher = _seed_crowding_pool(cfg)
    items = [_item("orig", "paper-b", "S1")]
    pool = {"paper-a": "md", "paper-b": "md"}

    per_paper_confirm(
        cfg, items, pool, n_papers=2, candidates=2, variant="production", seed=7, searcher=searcher
    )

    assert calls == [7]


def test_per_paper_confirm_rejects_unknown_variant(make_config):
    cfg = make_config()
    searcher = _seed_crowding_pool(cfg)
    items = [_item("orig", "paper-b", "S1")]
    pool = {"paper-a": "md", "paper-b": "md"}

    with pytest.raises(ValueError, match="variant"):
        per_paper_confirm(
            cfg, items, pool, n_papers=2, candidates=2, variant="bogus", seed=1, searcher=searcher
        )


def test_per_paper_confirm_report_reflects_a_confirmed_result(make_config):
    cfg = make_config()
    searcher = _seed_crowding_pool(cfg)
    items = [_item("orig", "paper-b", "S1")]
    pool = {"paper-a": "md", "paper-b": "md"}

    result = per_paper_confirm(
        cfg, items, pool, n_papers=2, candidates=2, variant="production", seed=1, searcher=searcher
    )
    text = format_per_paper_confirm(result, n_papers=2, seed=1)
    assert "confirmed" in text
    assert "seed=1" in text


def test_per_paper_sweep_checkpoint_header_trip_wires_on_seed_and_grid(
    monkeypatch, make_searcher, seed_chunks, tmp_path
):
    docs = [seed_chunks("paper-a", "S1", "alpha content")]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = _FakeReranker()  # avoid loading a real cross-encoder
    items = [_item("alpha", "paper-a", "S1")]
    pool = {"paper-a": "md"}

    from eval import harness as harness_mod

    captured_headers: list[dict] = []
    real_init = harness_mod.CheckpointWriter.__init__

    def spy_init(self, path, header):
        captured_headers.append(header)
        real_init(self, path, header)

    monkeypatch.setattr(harness_mod.CheckpointWriter, "__init__", spy_init)

    ckpt = tmp_path / "pp.ckpt.jsonl"
    per_paper_sweep(
        ctx.cfg,
        items,
        pool,
        n_papers=1,
        candidates_grid=[5],
        seed=3,
        searcher=ctx.searcher,
        checkpoint_path=ckpt,
    )

    assert captured_headers
    header = captured_headers[0]
    assert header["seed"] == 3
    assert header["n_papers"] == 1
    assert header["candidates_grid"] == [5]


def test_per_paper_sweep_resumes_without_recomputing_a_finished_arm(
    monkeypatch, make_config, tmp_path
):
    # Regression guard: each per-paper arm's checkpoint record must round-trip through
    # CheckpointWriter/resume_units correctly (a dict, per the module's contract — not a
    # bare list), and a finished arm must never be recomputed on resume.
    from eval import harness as harness_mod

    cfg = make_config()
    searcher = _seed_crowding_pool(cfg)
    items = [_item("orig", "paper-b", "S1")]
    pool = {"paper-a": "md", "paper-b": "md"}
    ckpt = tmp_path / "pp.ckpt.jsonl"

    real_score_items_scoped = harness_mod.score_items_scoped
    state = {"n": 0, "fail": True}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["fail"] and state["n"] == 2:  # fail on the 2nd arm ("on (production)")
            raise RuntimeError("simulated interruption")
        return real_score_items_scoped(*args, **kwargs)

    monkeypatch.setattr(harness_mod, "score_items_scoped", flaky)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        per_paper_sweep(
            cfg,
            items,
            pool,
            n_papers=2,
            candidates_grid=[2],
            seed=0,
            searcher=searcher,
            checkpoint_path=ckpt,
        )
    assert ckpt.exists()
    calls_before_resume = state["n"]  # off succeeded (1), production failed (2)

    state["fail"] = False  # let the interrupted run "come back up"
    report = per_paper_sweep(
        cfg,
        items,
        pool,
        n_papers=2,
        candidates_grid=[2],
        seed=0,
        searcher=searcher,
        checkpoint_path=ckpt,
    )

    # Only the 2 un-cached arms (production, budget-matched) are recomputed — the
    # already-checkpointed "off" arm is loaded from disk, never re-scored.
    assert state["n"] - calls_before_resume == 2
    by_label = {r.label: r for r in report.results}
    assert by_label["off"].success.point == 0.0
    assert by_label["on (production)"].success.point == 1.0
    assert by_label["on (budget-matched)"].success.point == 1.0
    assert not ckpt.exists()  # deleted once every arm is accounted for


# --- comparative: cross-paper synthesis questions (gold spans 2+ papers) ------------------


class _ComparativeCrowdingEmbedder:
    """Mirrors _CrowdingEmbedder: paper-b's two chunks both outrank paper-c's one chunk,
    so a shared candidates=2 budget crowds paper-c out of a comparative item spanning
    both papers -- unless per_paper is on. Unlike the per-paper crowding fixture, there's
    no third "extra" paper in the scope at all: for a comparative item, crowding happens
    directly between the item's own gold papers."""

    _VECS = {
        "orig": [1.0, 0.0],
        "b1_text": [1.0, 0.0],
        "b2_text": [4.0, 1.0],
        "c1_text": [1.0, 1.0],
    }

    def name(self) -> str:
        return "cmp-crowding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Any query text not one of the fixture's exact docs (e.g. a checkpoint trip-wire
        # test's differently-worded query) resolves to "orig"'s direction -- only the doc
        # texts' own vectors matter for the crowding ranking these tests assert on.
        return [self._VECS.get(t, self._VECS["orig"]) for t in input]


def _seed_comparative_crowding_pool(cfg):
    embedder = _ComparativeCrowdingEmbedder()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())
    docs = [
        ("b1", "b1_text", "paper-b", "S1"),
        ("b2", "b2_text", "paper-b", "S2"),
        ("c1", "c1_text", "paper-c", "S1"),
    ]
    metas = [
        {
            "paper_id": pid,
            "breadcrumb": f"Paper > {s}",
            "section_title": s,
            "section_number": "1",
            "body": t,
        }
        for _, t, pid, s in docs
    ]
    collection.upsert(
        ids=[d for d, _, _, _ in docs],
        embeddings=embedder([t for _, t, _, _ in docs]),
        documents=[t for _, t, _, _ in docs],
        metadatas=metas,
    )
    return Searcher(
        db_dir=cfg.paths.rag_db,
        collection=cfg.collection,
        embedder=embedder,
        reranker=_FakeReranker(),
    )


def _comparative_item(query: str, papers: list[str]) -> ComparativeQAItem:
    # section_number "1" matches seed_chunks'/_seed_comparative_crowding_pool's metadata.
    return ComparativeQAItem(
        query=query,
        sections=[
            Section(paper_id=pid, number="1", title="S1", body="", start=0, end=1) for pid in papers
        ],
    )


def test_comparative_sweep_recovers_a_crowded_paper(make_config):
    cfg = make_config()
    searcher = _seed_comparative_crowding_pool(cfg)
    items = [_comparative_item("orig", ["paper-b", "paper-c"])]

    report = comparative_sweep(cfg, items, candidates_grid=[2], searcher=searcher)

    by_label = {r.label: r for r in report.results}
    assert by_label["off"].success.point == 0.0  # paper-c's chunk is crowded out
    assert by_label["on (production)"].success.point == 1.0
    # The allocation itself helps, not just extra fetch volume: budget-matched holds the
    # SAME total pool size as the off-arm and still recovers paper-c's chunk.
    assert by_label["on (budget-matched)"].success.point == 1.0
    assert by_label["on (budget-matched)"].mean_pool_size == by_label["off"].mean_pool_size
    assert report.n_papers_min == report.n_papers_max == 2

    text = format_comparative_sweep(report)
    assert "candidates=2" in text
    assert "n_papers_per_item=2..2" in text
    assert "Candidate points to confirm" in text


def test_comparative_confirm_recovers_a_crowded_paper(make_config):
    cfg = make_config()
    searcher = _seed_comparative_crowding_pool(cfg)
    items = [_comparative_item("orig", ["paper-b", "paper-c"])]

    result = comparative_confirm(cfg, items, candidates=2, variant="production", searcher=searcher)

    assert result.off.success.point == 0.0
    assert result.on.success.point == 1.0
    assert result.on.success_delta is not None
    assert result.on.success_delta.delta == 1.0


def test_score_comparative_items_scoped_sets_primary_paper_id_from_sections_zero(make_config):
    # Regression guard: primary_paper_id (the bootstrap cluster key) must come from the
    # item's own sections[0] -- the trial's earliest-drawn paper at generation time -- not
    # get recomputed some other way (alphabetically, by scope order, etc.) at scoring
    # time. Two items with the SAME gold papers but OPPOSITE sections[0] ordering must
    # score to opposite primary_paper_id values.
    cfg = make_config()
    searcher = _seed_comparative_crowding_pool(cfg)
    b_first = _comparative_item("orig", ["paper-b", "paper-c"])
    c_first = _comparative_item("orig", ["paper-c", "paper-b"])

    scores_b_first = score_comparative_items_scoped(
        searcher, [b_first], per_paper=False, candidates=2, k=10, rerank=True
    )
    scores_c_first = score_comparative_items_scoped(
        searcher, [c_first], per_paper=False, candidates=2, k=10, rerank=True
    )

    assert scores_b_first[0].primary_paper_id == "paper-b"
    assert scores_c_first[0].primary_paper_id == "paper-c"


def test_comparative_sweep_checkpoint_trip_wires_on_items_fingerprint(
    monkeypatch, make_config, tmp_path
):
    from eval import harness as harness_mod

    cfg = make_config()
    searcher = _seed_comparative_crowding_pool(cfg)
    items = [_comparative_item("orig", ["paper-b", "paper-c"])]

    captured_headers: list[dict] = []
    real_init = harness_mod.CheckpointWriter.__init__

    def spy_init(self, path, header):
        captured_headers.append(header)
        real_init(self, path, header)

    monkeypatch.setattr(harness_mod.CheckpointWriter, "__init__", spy_init)

    ckpt = tmp_path / "cmp.ckpt.jsonl"
    comparative_sweep(cfg, items, candidates_grid=[2], searcher=searcher, checkpoint_path=ckpt)

    assert captured_headers
    header = captured_headers[0]
    assert "items_fingerprint" in header

    # A different items list (different query) must trip-wire to a different fingerprint —
    # otherwise a stale checkpoint from an old comparative gen run could silently merge
    # with a fresh dev split's scores.
    other_items = [_comparative_item("a completely different query", ["paper-b", "paper-c"])]
    captured_headers.clear()
    ckpt2 = tmp_path / "cmp2.ckpt.jsonl"
    comparative_sweep(
        cfg, other_items, candidates_grid=[2], searcher=searcher, checkpoint_path=ckpt2
    )
    assert captured_headers[0]["items_fingerprint"] != header["items_fingerprint"]


def _comparative_confirm_result(
    n_clusters: int, *, delta_n_clusters: int | None = None
) -> ComparativeConfirmResult:
    """``delta_n_clusters`` defaults to ``n_clusters`` (the common case, and what every
    existing caller wants) but can be set independently to prove the floor gates on the
    delta's own resolution, not either arm's standalone one."""
    delta_n_clusters = n_clusters if delta_n_clusters is None else delta_n_clusters
    off = ComparativeArmResult(
        label="off",
        candidates=10,
        mean_pool_size=10.0,
        success=BootResult(
            point=0.3, ci_lo=0.1, ci_hi=0.5, se=0.1, n_clusters=n_clusters, n_eligible=n_clusters
        ),
        mrr=BootResult(
            point=0.2, ci_lo=0.1, ci_hi=0.3, se=0.05, n_clusters=n_clusters, n_eligible=n_clusters
        ),
        success_delta=None,
        mrr_delta=None,
    )
    on = ComparativeArmResult(
        label="on (production)",
        candidates=10,
        mean_pool_size=40.0,
        success=BootResult(
            point=0.6, ci_lo=0.4, ci_hi=0.8, se=0.1, n_clusters=n_clusters, n_eligible=n_clusters
        ),
        mrr=BootResult(
            point=0.4, ci_lo=0.3, ci_hi=0.5, se=0.05, n_clusters=n_clusters, n_eligible=n_clusters
        ),
        success_delta=DeltaResult(
            delta=0.3,
            ci_lo=0.1,
            ci_hi=0.5,
            se=0.1,
            mdd=0.28,
            n_clusters=delta_n_clusters,
            n_paired=delta_n_clusters,
        ),
        mrr_delta=None,
    )
    return ComparativeConfirmResult(
        off=off, on=on, n_papers_min=2, n_papers_max=2, n_papers_mean=2.0
    )


def test_format_comparative_confirm_hard_floor_blocks_confirmed_below_threshold():
    # A delta whose CI clears zero (0.1 to 0.5, doesn't straddle) must still refuse to say
    # "confirmed" when n_clusters is below the floor -- this is the whole point of the
    # floor existing (confirm's recommendation language must never speak with more
    # confidence than the resolution can support).
    assert COMPARATIVE_MIN_CLUSTERS > 4  # sanity: the fixture below is genuinely under it
    result = _comparative_confirm_result(n_clusters=4)

    text = format_comparative_confirm(result, candidates=10)

    assert "Too few clusters to confirm anything" in text
    assert f"n_clusters=4 < COMPARATIVE_MIN_CLUSTERS={COMPARATIVE_MIN_CLUSTERS}" in text
    assert "*** confirmed" not in text
    assert not text.rstrip().endswith("confirmed.")  # no bare "Confirmed: ..." verdict line
    assert "Confirmed:" not in text


def test_format_comparative_confirm_prints_confirmed_at_or_above_floor():
    result = _comparative_confirm_result(n_clusters=COMPARATIVE_MIN_CLUSTERS)

    text = format_comparative_confirm(result, candidates=10)

    assert "Confirmed:" in text
    assert "*** confirmed" in text
    assert "Too few clusters" not in text


def test_format_comparative_confirm_floor_gates_on_the_deltas_own_n_clusters():
    # Regression guard: the floor must read the DELTA's own n_clusters (paired_delta's
    # intersection-conditioned resolution), not either arm's standalone one -- even when
    # the on-arm's own success.n_clusters clears the floor, a delta backed by fewer
    # clusters than that must still be blocked from saying "confirmed."
    result = _comparative_confirm_result(
        n_clusters=COMPARATIVE_MIN_CLUSTERS + 10, delta_n_clusters=COMPARATIVE_MIN_CLUSTERS - 1
    )

    text = format_comparative_confirm(result, candidates=10)

    assert "Too few clusters to confirm anything" in text
    assert f"n_clusters={COMPARATIVE_MIN_CLUSTERS - 1}" in text
    assert "Confirmed:" not in text
