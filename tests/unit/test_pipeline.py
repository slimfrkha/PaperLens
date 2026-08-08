"""Pure pipeline helpers (no I/O)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag import pipeline
from rag.config import Paper
from rag.manifest import Manifest
from rag.pipeline import _title, normalize_manifest_tags, pending_papers


def test_pending_papers_excludes_already_ingested(tmp_path, make_config):
    cfg = make_config(
        papers=[Paper(name="a", arxiv_id="1"), Paper(name="b", arxiv_id="2")],
    )
    manifest = Manifest(cfg.paths.rag_db)
    manifest.upsert({"paper_id": "a", "tags": [], "n_chunks": 1})

    pending = pending_papers(cfg.for_ingest(), manifest)
    assert [p.name for p in pending] == ["b"]


def test_title_reads_first_h2_and_strips_comments():
    md = "<!-- image -->\n\n## Real Title\n\nbody\n\n## Later"
    assert _title(md) == "Real Title"


def test_title_empty_when_no_heading():
    assert _title("just text, no headings") == ""


def test_normalize_manifest_tags_rewrites_and_dedupes(make_config, monkeypatch):
    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    manifest.upsert({"paper_id": "a", "tags": ["moe", "mixture-of-experts"], "n_chunks": 1})
    manifest.upsert({"paper_id": "b", "tags": ["rl"], "n_chunks": 1})

    monkeypatch.setattr(
        pipeline, "normalize_tags", lambda vocab, spec: {"moe": "mixture-of-experts"}
    )

    mapping = normalize_manifest_tags(cfg.for_ingest(), manifest)
    assert mapping == {"moe": "mixture-of-experts"}
    # "moe" folds into the existing "mixture-of-experts"; dedupe collapses the pair.
    assert manifest.get("a")["tags"] == ["mixture-of-experts"]
    assert manifest.get("b")["tags"] == ["rl"]


def test_normalize_manifest_tags_noop_on_empty_map(make_config, monkeypatch):
    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    manifest.upsert({"paper_id": "a", "tags": ["moe"], "n_chunks": 1})
    monkeypatch.setattr(pipeline, "normalize_tags", lambda vocab, spec: {})
    assert normalize_manifest_tags(cfg.for_ingest(), manifest) == {}
    assert manifest.get("a")["tags"] == ["moe"]


def test_normalize_manifest_tags_survives_failure(make_config, monkeypatch):
    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    manifest.upsert({"paper_id": "a", "tags": ["moe"], "n_chunks": 1})

    def _boom(vocab, spec):
        raise RuntimeError("no tagging key")

    monkeypatch.setattr(pipeline, "normalize_tags", _boom)
    # A failed normalization must leave every paper's tags untouched.
    assert normalize_manifest_tags(cfg.for_ingest(), manifest) == {}
    assert manifest.get("a")["tags"] == ["moe"]


def _fake_ingest(rec_for=lambda paper: {"paper_id": paper.name, "tags": [], "n_chunks": 1}):
    def _ingest(paper, cfg, embedder, collection, manifest, on_stage=None, retag=True):
        if on_stage:
            on_stage("download", 0.0)
            on_stage("done", 1.0)
        return rec_for(paper)

    return _ingest


def test_run_batch_builds_embedder_and_collection_when_not_supplied(make_config, monkeypatch):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    fake_embedder = SimpleNamespace(name=lambda: "fake")
    fake_collection = object()
    built = {"embedder": False, "collection": False}

    def fake_build(cfg):
        built["embedder"] = True
        return fake_embedder

    def fake_open(db_dir, collection, embedder_name=None, reset=False):
        built["collection"] = True
        return fake_collection

    monkeypatch.setattr(pipeline, "build_embedder_from_config", fake_build)
    monkeypatch.setattr(pipeline, "open_collection", fake_open)
    monkeypatch.setattr(pipeline, "ingest_paper", _fake_ingest())

    result = pipeline.run_batch(cfg, manifest, [Paper(name="a", arxiv_id="1")])
    assert built == {"embedder": True, "collection": True}
    assert result.embedder is fake_embedder
    assert result.collection is fake_collection
    assert result.records == [{"paper_id": "a", "tags": [], "n_chunks": 1}]


def test_run_batch_reuses_supplied_embedder_and_collection(make_config, monkeypatch):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)

    def fail(*a, **k):
        raise AssertionError("should not build when embedder/collection are supplied")

    monkeypatch.setattr(pipeline, "build_embedder_from_config", fail)
    monkeypatch.setattr(pipeline, "open_collection", fail)
    monkeypatch.setattr(pipeline, "ingest_paper", _fake_ingest())

    sentinel_embedder, sentinel_collection = object(), object()
    result = pipeline.run_batch(
        cfg,
        manifest,
        [Paper(name="a", arxiv_id="1")],
        embedder=sentinel_embedder,
        collection=sentinel_collection,
    )
    assert result.embedder is sentinel_embedder
    assert result.collection is sentinel_collection


def test_run_batch_fires_hooks_in_order_with_correct_args(make_config, monkeypatch):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    papers = [Paper(name="a", arxiv_id="1"), Paper(name="b", arxiv_id="2")]
    calls = []

    monkeypatch.setattr(
        pipeline, "build_embedder_from_config", lambda cfg: SimpleNamespace(name=lambda: "f")
    )
    monkeypatch.setattr(pipeline, "open_collection", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "ingest_paper", _fake_ingest())

    pipeline.run_batch(
        cfg,
        manifest,
        papers,
        on_paper_start=lambda p: calls.append(("start", p.name)),
        on_stage=lambda p, s, pct: calls.append(("stage", p.name, s, pct)),
        on_paper_done=lambda p, rec, exc: calls.append(("done", p.name, rec is not None, exc)),
    )

    assert calls == [
        ("start", "a"),
        ("stage", "a", "download", 0.0),
        ("stage", "a", "done", 1.0),
        ("done", "a", True, None),
        ("start", "b"),
        ("stage", "b", "download", 0.0),
        ("stage", "b", "done", 1.0),
        ("done", "b", True, None),
    ]


def test_run_batch_stop_on_error_true_raises_and_stops_immediately(make_config, monkeypatch):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    papers = [Paper(name="a", arxiv_id="1"), Paper(name="b", arxiv_id="2")]
    attempted = []

    def fake_ingest(paper, *a, **k):
        attempted.append(paper.name)
        if paper.name == "a":
            raise RuntimeError("boom")
        return {"paper_id": paper.name, "tags": [], "n_chunks": 1}

    monkeypatch.setattr(
        pipeline, "build_embedder_from_config", lambda cfg: SimpleNamespace(name=lambda: "f")
    )
    monkeypatch.setattr(pipeline, "open_collection", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "ingest_paper", fake_ingest)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_batch(cfg, manifest, papers, stop_on_error=True)
    assert attempted == ["a"]  # never reached "b" — batch stopped at the first failure


def test_run_batch_stop_on_error_false_continues_past_failure(make_config, monkeypatch):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    papers = [Paper(name="a", arxiv_id="1"), Paper(name="b", arxiv_id="2")]
    errors = []

    def fake_ingest(paper, *a, **k):
        if paper.name == "a":
            raise RuntimeError("boom")
        return {"paper_id": paper.name, "tags": [], "n_chunks": 1}

    monkeypatch.setattr(
        pipeline, "build_embedder_from_config", lambda cfg: SimpleNamespace(name=lambda: "f")
    )
    monkeypatch.setattr(pipeline, "open_collection", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "ingest_paper", fake_ingest)

    result = pipeline.run_batch(
        cfg,
        manifest,
        papers,
        on_paper_done=lambda p, rec, exc: errors.append((p.name, exc)) if exc else None,
        stop_on_error=False,
    )
    assert [r["paper_id"] for r in result.records] == ["b"]  # "a" failed, "b" still ran
    assert len(errors) == 1 and errors[0][0] == "a" and isinstance(errors[0][1], RuntimeError)


def test_run_batch_does_not_call_normalize(make_config, monkeypatch):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    called = {"normalize": False}

    monkeypatch.setattr(
        pipeline, "build_embedder_from_config", lambda cfg: SimpleNamespace(name=lambda: "f")
    )
    monkeypatch.setattr(pipeline, "open_collection", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "ingest_paper", _fake_ingest())

    def fake_normalize(cfg, manifest):
        called["normalize"] = True
        return {}

    monkeypatch.setattr(pipeline, "normalize_manifest_tags", fake_normalize)
    pipeline.run_batch(cfg, manifest, [Paper(name="a", arxiv_id="1")])
    assert called["normalize"] is False
