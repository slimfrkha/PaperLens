"""End-to-end ingestion of one paper: download -> markdown -> index -> tag -> manifest.

Network (arXiv), Docling extraction, and the tagging LLM are stubbed; chunking,
embedding into a real temp Chroma, and the manifest write run for real.
"""

from __future__ import annotations

from pathlib import Path

from rag import pipeline
from rag.config import ChunkingCfg, Paper
from rag.index import open_collection
from rag.manifest import Manifest

_MARKDOWN = """## Paper A

## Abstract
A strong MoE model.

## 1 Architecture
""" + " ".join(["multi latent attention shrinks the kv cache"] * 12)


def _fake_download(arxiv_id: str, dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(b"%PDF")


def test_ingest_paper_populates_db_and_manifest(make_config, fake_embedder, monkeypatch):
    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    collection = open_collection(
        cfg.paths.rag_db, cfg.collection, embedder_name=fake_embedder.name()
    )

    # Stub the three external stages.
    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path, **kw: _MARKDOWN)
    monkeypatch.setattr(
        pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: ["moe", "attention"]
    )

    stages: list[str] = []
    record = pipeline.ingest_paper(
        Paper(name="paper-a", arxiv_id="0000.00001"),
        cfg.for_ingest(),
        fake_embedder,
        collection,
        manifest,
        on_stage=lambda name, pct: stages.append(name),
    )

    assert record["title"] == "Paper A"
    assert record["tags"] == ["moe", "attention"]
    assert record["n_chunks"] > 0
    assert collection.count() == record["n_chunks"]
    # Manifest persisted and markdown cached to disk.
    assert manifest.is_ingested("paper-a")
    assert (Path(cfg.paths.markdown_dir) / "paper-a.md").exists()
    assert stages[0] == "download" and stages[-1] == "done"


def test_ingest_survives_tagging_failure(make_config, fake_embedder, monkeypatch):
    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    collection = open_collection(cfg.paths.rag_db, cfg.collection)

    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path, **kw: _MARKDOWN)

    def _boom(*a, **k):
        raise RuntimeError("no tagging key")

    monkeypatch.setattr(pipeline, "generate_tags", _boom)

    record = pipeline.ingest_paper(
        Paper(name="p", arxiv_id="1"), cfg.for_ingest(), fake_embedder, collection, manifest
    )
    # Tagging degrades to empty tags; the rest of the record still lands.
    assert record["tags"] == []
    assert record["n_chunks"] > 0


def test_ingest_propagates_index_failure(make_config, fake_embedder, monkeypatch):
    """An indexing failure must surface (not be swallowed by the tag thread) and
    leave the paper pending — no manifest write."""
    import pytest

    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    collection = open_collection(cfg.paths.rag_db, cfg.collection)

    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path, **kw: _MARKDOWN)
    monkeypatch.setattr(pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: ["moe"])

    def _boom(*a, **k):
        raise RuntimeError("index blew up")

    monkeypatch.setattr(pipeline, "index_markdown", _boom)

    with pytest.raises(RuntimeError, match="index blew up"):
        pipeline.ingest_paper(
            Paper(name="p", arxiv_id="1"), cfg.for_ingest(), fake_embedder, collection, manifest
        )
    assert not manifest.is_ingested("p")


def test_chunking_config_reaches_chunk_markdown(make_config, fake_embedder, monkeypatch):
    """The cfg.chunking -> index_markdown -> chunk_markdown forwarding chain.

    Unit tests call chunk_markdown directly, so a kwarg dropped anywhere along
    this chain would otherwise go unnoticed.
    """
    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path, **kw: _MARKDOWN)
    monkeypatch.setattr(pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: [])

    def _ingest(chunking: ChunkingCfg | None) -> int:
        cfg = make_config() if chunking is None else make_config(chunking=chunking)
        manifest = Manifest(cfg.paths.rag_db)
        collection = open_collection(
            cfg.paths.rag_db, cfg.collection, embedder_name=fake_embedder.name(), reset=True
        )
        record = pipeline.ingest_paper(
            Paper(name="paper-a", arxiv_id="0000.00001"),
            cfg.for_ingest(),
            fake_embedder,
            collection,
            manifest,
        )
        return record["n_chunks"]

    # Defaults: the short Abstract is dropped, "1 Architecture" survives.
    assert _ingest(None) == 1
    # min_tokens=1 keeps the Abstract too.
    assert _ingest(ChunkingCfg(min_tokens=1)) == 2
    # ...and an extra skip pattern drops the one section that was surviving.
    assert _ingest(ChunkingCfg(extra_skip_titles=["architecture"])) == 0
