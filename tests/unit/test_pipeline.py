"""Pure pipeline helpers (no I/O)."""

from __future__ import annotations

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
