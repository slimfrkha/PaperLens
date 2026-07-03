"""Pure pipeline helpers (no I/O)."""

from __future__ import annotations

from rag.config import Paper
from rag.manifest import Manifest
from rag.pipeline import _title, pending_papers


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
