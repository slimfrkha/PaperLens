"""Retrieval optimizer: cache equivalence, paired deltas, and the OFAT arm set.

Offline against a real temp Chroma (fake embedder). A fake reranker lets the tests move the
stage-2 ordering deterministically without loading the cross-encoder.
"""

from __future__ import annotations

import pytest

from eval.harness import score_items
from eval.optimizer import _arms, _delta_cell, build_cache, score_from_cache, screen_retrieval
from eval.queryset import QAItem
from eval.stats import DeltaResult
from rag.sparse import build_sparse_index


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

    report = screen_retrieval(cfg, items, candidate_grid=[1, 20], searcher=ctx.searcher)
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
    report = screen_retrieval(make_config(), items, candidate_grid=[10, 20], searcher=ctx.searcher)

    from eval.optimizer import format_screen_report

    text = format_screen_report(report)
    assert text.splitlines()[0].startswith("Retrieval screen")
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


# --- Hybrid: build_cache(sparse=...), score_from_cache(sparse=True), the "hybrid=on" arm ---


def test_build_cache_with_sparse_preserves_default_arm_exactness(make_searcher, seed_chunks):
    # Widening the cache for a hybrid arm must not change what the *default* (non-hybrid) arm
    # reads back — score_from_cache(sparse=False) has to be bit-identical whether or not the
    # cache underneath it was built wide for a hybrid arm elsewhere.
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
    sparse_index = build_sparse_index(ctx.collection)
    plain = build_cache(ctx.searcher, items, max_candidates=2, reranker=FakeReranker(set()))
    widened = build_cache(
        ctx.searcher,
        items,
        max_candidates=2,
        reranker=FakeReranker(set()),
        sparse=sparse_index,
        fetch_multiplier=2,
    )
    assert all(c.sparse_ids for c in widened)  # the whole point of passing `sparse`
    for w, p in zip(widened, plain, strict=True):
        from_widened = score_from_cache(w, candidates=2, k=5, rerank=False)
        from_plain = score_from_cache(p, candidates=2, k=5, rerank=False)
        assert from_widened.candidate_ids == from_plain.candidate_ids
        assert [cid for cid, _ in from_widened.ranked] == [cid for cid, _ in from_plain.ranked]


def test_score_from_cache_sparse_fuses_bm25_into_the_pool(make_searcher, seed_chunks):
    # score_from_cache(sparse=True) RRF-fuses candidate_ids with sparse_ids rather than just
    # reading the dense pool — the plumbing test_hybrid_surfaces_a_lexical_match_dense_only_misses
    # (tests/integration/test_search.py) covers at the Searcher level.
    docs = [
        seed_chunks("p1", "Method", "zzflorble is a rare made-up term", doc_id="p1-rare"),
        seed_chunks("p2", "Other", "completely unrelated filler content", doc_id="p2-noise"),
        # A 3rd doc: with only 2, a term in exactly one of them always gets idf=0
        # (log(1.5)-log(1.5)) — a degenerate BM25 property of tiny corpora (see test_sparse.py).
        seed_chunks("p3", "Other", "yet more unrelated filler content here", doc_id="p3-noise"),
    ]
    ctx = make_searcher(docs)
    items = [_item("zzflorble", "p1", "Method")]
    sparse_index = build_sparse_index(ctx.collection)
    cache = build_cache(
        ctx.searcher,
        items,
        max_candidates=1,
        reranker=FakeReranker(set()),
        sparse=sparse_index,
        fetch_multiplier=2,
    )[0]
    assert cache.sparse_ids == ["p1-rare"]  # the only doc BM25 finds any relevance in
    hybrid = score_from_cache(cache, candidates=2, k=2, rerank=False, sparse=True)
    assert "p1-rare" in hybrid.candidate_ids  # RRF-fused in via sparse_ids, not just dense


def test_screen_retrieval_hybrid_arm_plumbs_through_paired_delta(make_config, seed_chunks):
    from types import SimpleNamespace

    from rag.index import open_collection
    from rag.search import Searcher

    class _DirectionalEmbedder:
        """See tests/integration/test_search.py — decouples dense closeness from lexical
        content so a hybrid-only hit can be constructed deterministically."""

        def _vec(self, text: str) -> list[float]:
            return [0.0, 1.0] if "FAR_MARKER" in text else [1.0, 0.0]

        def __call__(self, input: list[str]) -> list[list[float]]:
            return [self._vec(t) for t in input]

    embedder = _DirectionalEmbedder()
    cfg = make_config()
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name="directional")
    docs = [
        seed_chunks("target", "S", "FAR_MARKER zzflorble appears once here"),
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
    searcher = Searcher(db_dir=cfg.paths.rag_db, collection=cfg.collection, embedder=embedder)
    searcher._reranker = FakeReranker(set())  # the rerank=on toggle arm needs one; no model
    ctx = SimpleNamespace(searcher=searcher, cfg=cfg, collection=collection)

    cfg.retrieval.candidates = 3
    cfg.sparse.fetch_multiplier = 3
    items = [_item("zzflorble", "target", "S")]

    report = screen_retrieval(cfg, items, searcher=ctx.searcher, hybrid=True)
    labels = [r.arm.label for r in report.results]
    assert "hybrid=on" in labels
    hybrid_result = next(r for r in report.results if r.arm.label == "hybrid=on")
    assert hybrid_result.success_delta is not None
    assert any("fetch_multiplier" in c for c in report.caveats)


# --- Chunking: isolated re-index screen + sweep, offline over a temp Chroma ---------------

# A two-paper pool with numbered sections. Numbered sections are immune to the noise_ratio
# filter, so a noise_ratio arm re-chunks to the *same* chunks — an inert knob by construction.
_CHUNKING_POOL = {
    "p1": "## Paper One\n\nAuthors\n\n"
    "## 1. Method\n\nlatent attention compresses the key value cache substantially\n\n"
    "## 2. Training\n\nfp8 mixed precision schedule with a long warmup for stability\n",
    "p2": "## Paper Two\n\nAuthors\n\n"
    "## 1. Method\n\nrotary embeddings extend the context window efficiently\n\n"
    "## 2. Training\n\nadamw optimizer with cosine decay and gradient clipping\n",
}
_CHUNKING_ITEMS = [
    _item("latent attention compresses the key value cache", "p1", "Method"),
    _item("fp8 mixed precision schedule long warmup", "p1", "Training"),
    _item("rotary embeddings extend the context window", "p2", "Method"),
    _item("adamw optimizer cosine decay gradient clipping", "p2", "Training"),
]


def test_chunking_arms_are_ofat_around_default(make_config):
    from eval.optimizer import chunking_arms

    cfg = make_config()  # default chunking: max_tokens=512, overlap=64, min=24, noise=0.4
    arms = chunking_arms(cfg, grids={"max_tokens": [256, 512, 1024], "noise_ratio": [0.9]})
    assert arms[0].label == "default"
    # 512 == default is skipped; each off-default value becomes exactly one arm.
    assert [a.label for a in arms[1:]] == ["max_tokens=256", "max_tokens=1024", "noise_ratio=0.9"]
    assert arms[1].chunking.max_tokens == 256  # exactly one knob changed per arm
    assert arms[1].chunking.overlap_tokens == cfg.chunking.overlap_tokens


def test_screen_chunking_does_not_star_an_inert_knob(make_config, tmp_path, fake_embedder):
    from eval.optimizer import _delta_cell, screen_chunking

    cfg = make_config()  # reranker disabled → no model, rerank=off path
    report = screen_chunking(
        cfg,
        _CHUNKING_POOL,
        _CHUNKING_ITEMS,
        db_dir=str(tmp_path),
        grids={"noise_ratio": [0.9]},  # inert on numbered sections
        embedder=fake_embedder,
    )
    assert report.results[0].label == "default"
    arm = next(r for r in report.results if r.label == "noise_ratio=0.9")
    # An inert knob re-chunks identically → per-query delta is exactly 0 → never starred.
    assert arm.success_delta is not None and arm.success_delta.delta == 0.0
    assert arm.mrr_delta is not None and arm.mrr_delta.delta == 0.0
    assert not _delta_cell(arm.success_delta).rstrip().endswith("*")
    assert not _delta_cell(arm.mrr_delta).rstrip().endswith("*")


def test_screen_chunking_report_leads_with_resolution(make_config, tmp_path, fake_embedder):
    from eval.optimizer import format_chunking_report, screen_chunking

    report = screen_chunking(
        make_config(),
        _CHUNKING_POOL,
        _CHUNKING_ITEMS,
        db_dir=str(tmp_path),
        grids={"max_tokens": [256]},
        embedder=fake_embedder,
    )
    text = format_chunking_report(report)
    assert text.splitlines()[0].startswith("Chunking screen")
    assert "n_clusters=" in text
    assert "default arm:" in text and "max_tokens=512" in text
    # A max_tokens arm is present → the confound caveat is surfaced; the eligibility one always.
    assert "entangles with pool depth" in text
    assert "trust the starred paired Δ" in text


def test_screen_chunking_omits_confound_caveat_without_a_max_tokens_arm(
    make_config, tmp_path, fake_embedder
):
    from eval.optimizer import format_chunking_report, screen_chunking

    report = screen_chunking(
        make_config(),
        _CHUNKING_POOL,
        _CHUNKING_ITEMS,
        db_dir=str(tmp_path),
        grids={"noise_ratio": [0.9]},  # no max_tokens arm → confound caveat does not apply
        embedder=fake_embedder,
    )
    text = format_chunking_report(report)
    assert "entangles with pool depth" not in text
    assert "trust the starred paired Δ" in text  # eligibility caveat still applies


def test_sweep_runs_offline_and_derives_the_grid(make_config, tmp_path, fake_embedder):
    from eval.optimizer import sweep

    cfg = make_config()  # chunking max_tokens=512, candidates=20, rerank off
    report = sweep(
        cfg,
        _CHUNKING_POOL,
        _CHUNKING_ITEMS,
        db_dir=str(tmp_path),
        max_tokens_grid=[256, 512],
        candidate_grid=[10, 20],
        embedder=fake_embedder,
        reranker=FakeReranker(set()),  # rerank=on arms need one; no cross-encoder download
    )
    labels = [r.label for r in report.results]
    assert labels[0] == "default"
    # The re-index axis (max_tokens) × cached retrieval axis (candidates × rerank) is enumerated.
    assert any("mt=256" in label for label in labels)
    assert any("rr=on" in label for label in labels)
    assert report.n_queries == len(_CHUNKING_ITEMS)
