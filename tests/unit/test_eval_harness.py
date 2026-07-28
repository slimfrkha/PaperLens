"""Phase 2 harness: a single-config run end-to-end over a real temp Chroma (fake embedder).

No reranker (make_config disables it), no network — the dense path is exercised against a
seeded collection, and the report's two metrics are asserted from a known layout.
"""

from __future__ import annotations

import json

import pytest

from eval.harness import run, score_items
from eval.queryset import QAItem, load_queryset


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


def test_load_queryset_rejects_pre_phase2_sets(tmp_path):
    # A row without section_title predates section-identity scoring; must fail loudly.
    path = tmp_path / "fp.dev.jsonl"
    path.write_text(json.dumps({"query": "q", "paper_id": "p", "gold_span": [0, 1]}) + "\n")
    with pytest.raises(SystemExit, match="regenerate"):
        load_queryset(str(path))
