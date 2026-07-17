"""Phase 4 Tier-A optimizer: cache equivalence, paired deltas, and the OFAT arm set.

Offline against a real temp Chroma (fake embedder). A fake reranker lets the tests move the
stage-2 ordering deterministically without loading the cross-encoder.
"""

from __future__ import annotations

import pytest

from eval.harness import score_items
from eval.optimizer import _arms, _delta_cell, build_cache, score_from_cache, screen_tier_a
from eval.queryset import QAItem
from eval.stats import DeltaResult


def _item(query: str, paper_id: str, section_title: str) -> QAItem:
    return QAItem(
        query=query,
        paper_id=paper_id,
        gold_span=(0, 1),
        source_unit=f"1 {section_title}",
        section_number="1",
        section_title=section_title,
    )


class FakeReranker:
    """Scores a doc by how many of ``boost`` words it contains — deterministic, no model."""

    def __init__(self, boost: set[str]) -> None:
        self.boost = boost

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(sum(w in d.lower() for w in self.boost)) for d in docs]


def test_score_from_cache_equals_direct_retrieval_no_rerank(make_searcher, seed_chunks):
    # The equivalence that licenses caching: cache-derived default arm == score_items().
    docs = [
        seed_chunks("p1", "Method", "latent attention compresses the cache", doc_id="p1-method"),
        seed_chunks("p1", "Training", "fp8 mixed precision schedule", doc_id="p1-train"),
        seed_chunks("p2", "Results", "benchmark accuracy evaluation", doc_id="p2-results"),
    ]
    ctx = make_searcher(docs)
    items = [
        _item("latent attention compresses the cache", "p1", "Method"),
        _item("fp8 mixed precision schedule", "p1", "Training"),
    ]
    cache = build_cache(ctx.searcher, items, max_candidates=20, reranker=FakeReranker(set()))
    from_cache = [score_from_cache(c, candidates=20, k=5, rerank=False) for c in cache]
    direct = score_items(ctx.searcher, items, candidates=20, k=5, rerank=False)

    for a, b in zip(from_cache, direct, strict=True):
        assert a.candidate_ids == b.candidate_ids
        assert [cid for cid, _ in a.ranked] == [cid for cid, _ in b.ranked]
        assert a.relevant_ids == b.relevant_ids


def test_score_from_cache_matches_direct_rerank_ordering(make_searcher, seed_chunks):
    docs = [
        seed_chunks("p1", "Method", "alpha beta gamma", doc_id="p1-m0"),
        seed_chunks("p1", "Method", "alpha delta", doc_id="p1-m1"),
        seed_chunks("p1", "Other", "alpha epsilon", doc_id="p1-o0"),
    ]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = FakeReranker({"delta"})  # ranks the "delta" doc first
    items = [_item("alpha", "p1", "Method")]

    cache = build_cache(ctx.searcher, items, max_candidates=20, reranker=ctx.searcher._reranker)
    from_cache = score_from_cache(cache[0], candidates=20, k=5, rerank=True)
    direct = score_items(ctx.searcher, items, candidates=20, k=5, rerank=True)[0]

    assert [cid for cid, _ in from_cache.ranked] == [cid for cid, _ in direct.ranked]
    assert from_cache.ranked[0][0] == "p1-m1"  # the "delta" doc reranked to the top


def test_arms_are_ofat_around_default(make_config):
    cfg = make_config()  # candidates=20, rerank disabled by make_config
    arms = _arms(cfg, [10, 20, 30])
    labels = [a.label for a in arms]
    assert labels[0] == "default"
    assert labels[1] == "rerank=on"  # default has rerank off → toggle is "on"
    # candidates==default (20) is skipped; each candidates arm keeps the default rerank state.
    assert [(a.candidates, a.rerank) for a in arms if a.label.startswith("candidates")] == [
        (10, False),
        (30, False),
    ]


def test_screen_detects_a_candidates_drop(make_searcher, make_config, seed_chunks):
    # Two papers, gold sections that only reach the pool at a deep candidates setting: shrinking
    # candidates to 1 pushes gold out for the noise-heavy queries → negative success delta.
    docs = [
        seed_chunks("p1", "Method", "quiver plankton", doc_id="p1-gold"),
        seed_chunks("p1", "Filler", "quiver quiver quiver plankton extra", doc_id="p1-noise"),
        seed_chunks("p2", "Method", "zebra cactus", doc_id="p2-gold"),
        seed_chunks("p2", "Filler", "zebra zebra zebra cactus extra", doc_id="p2-noise"),
    ]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = FakeReranker(set())  # the rerank=on toggle arm needs one; no model
    items = [_item("quiver plankton", "p1", "Method"), _item("zebra cactus", "p2", "Method")]
    cfg = make_config()  # candidates=20, rerank off

    report = screen_tier_a(cfg, items, candidate_grid=[1, 20], searcher=ctx.searcher)
    labels = [r.arm.label for r in report.results]
    assert "candidates=1" in labels
    assert report.default.candidates == 20
    c1 = next(r for r in report.results if r.arm.label == "candidates=1")
    # At candidates=1 only the noisier filler chunk survives for at least one query → success drops.
    assert c1.success.point <= report.results[0].success.point
    assert c1.success_delta is not None and c1.success_delta.delta <= 0.0


def test_format_screen_report_leads_with_resolution(make_searcher, make_config, seed_chunks):
    docs = [seed_chunks("p1", "Method", "alpha beta", doc_id="p1-m")]
    ctx = make_searcher(docs)
    ctx.searcher._reranker = FakeReranker(set())  # rerank=on toggle arm; no model download
    items = [_item("alpha beta", "p1", "Method")]
    report = screen_tier_a(make_config(), items, candidate_grid=[10, 20], searcher=ctx.searcher)

    from eval.optimizer import format_screen_report

    text = format_screen_report(report)
    assert text.splitlines()[0].startswith("Tier-A screen")
    assert "n_clusters=" in text
    assert "default arm:" in text


def _delta(delta, lo, hi, mdd):
    return DeltaResult(delta=delta, ci_lo=lo, ci_hi=hi, se=0.0, mdd=mdd, n_clusters=8, n_paired=100)


def test_delta_cell_never_stars_a_null_effect():
    # Regression: a rerank toggle leaves success@candidates exactly unchanged → Δ=0, MDD=0.
    # `abs(0) >= 0` used to falsely star it as "reliably detectable".
    assert "*" not in _delta_cell(_delta(0.0, 0.0, 0.0, 0.0))
    # A large effect whose CI excludes zero and clears the MDD IS starred.
    assert _delta_cell(_delta(-0.106, -0.145, -0.070, 0.053)).endswith("*")
    # A gain significant at 95% (CI excludes zero) but below the MDD is NOT starred.
    assert not _delta_cell(_delta(0.017, 0.006, 0.032, 0.019)).endswith("*")


def test_parse_grid_tolerates_junk_and_rejects_bad_values():
    from eval.cli import _parse_grid

    assert _parse_grid(None) is None
    assert _parse_grid("") is None
    assert _parse_grid(",") is None
    assert _parse_grid("30,10,20") == [10, 20, 30]  # sorted + de-duplicated
    assert _parse_grid("20, 10, 20,") == [10, 20]  # whitespace, dupes, trailing comma
    for bad in ("10,x", "0,10", "-5,10"):
        with pytest.raises(SystemExit):
            _parse_grid(bad)
