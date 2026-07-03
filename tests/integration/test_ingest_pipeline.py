"""End-to-end ingestion of one paper: download -> markdown -> index -> tag -> manifest.

Network (arXiv), Docling extraction, and the tagging LLM are stubbed; chunking,
embedding into a real temp Chroma, and the manifest write run for real.
"""

from __future__ import annotations

from pathlib import Path

from rag import pipeline
from rag.config import Paper
from rag.index import open_collection
from rag.manifest import Manifest

_MARKDOWN = """## DeepSeek-V3

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
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path: _MARKDOWN)
    monkeypatch.setattr(
        pipeline, "generate_tags", lambda md, spec, existing_tags: ["moe", "attention"]
    )

    stages: list[str] = []
    record = pipeline.ingest_paper(
        Paper(name="deepseek-v3", arxiv_id="2412.19437"),
        cfg.for_ingest(),
        fake_embedder,
        collection,
        manifest,
        on_stage=lambda name, pct: stages.append(name),
    )

    assert record["title"] == "DeepSeek-V3"
    assert record["tags"] == ["moe", "attention"]
    assert record["n_chunks"] > 0
    assert collection.count() == record["n_chunks"]
    # Manifest persisted and markdown cached to disk.
    assert manifest.is_ingested("deepseek-v3")
    assert (Path(cfg.paths.markdown_dir) / "deepseek-v3.md").exists()
    assert stages[0] == "download" and stages[-1] == "done"


def test_ingest_survives_tagging_failure(make_config, fake_embedder, monkeypatch):
    cfg = make_config()
    manifest = Manifest(cfg.paths.rag_db)
    collection = open_collection(cfg.paths.rag_db, cfg.collection)

    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path: _MARKDOWN)

    def _boom(*a, **k):
        raise RuntimeError("no tagging key")

    monkeypatch.setattr(pipeline, "generate_tags", _boom)

    record = pipeline.ingest_paper(
        Paper(name="p", arxiv_id="1"), cfg.for_ingest(), fake_embedder, collection, manifest
    )
    # Tagging degrades to empty tags; the rest of the record still lands.
    assert record["tags"] == []
    assert record["n_chunks"] > 0
