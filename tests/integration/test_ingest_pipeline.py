"""End-to-end ingestion of one paper: download -> markdown -> index -> tag -> manifest.

Network (arXiv), Docling extraction, and the tagging LLM are stubbed; chunking,
embedding into a real temp Chroma, and the manifest write run for real.
"""

from __future__ import annotations

from pathlib import Path

from rag import pipeline
from rag.config import ChunkingCfg, ExtractionCfg, Paper
from rag.index import open_collection
from rag.manifest import Manifest

_MARKDOWN = """## Paper A

## Abstract
A strong MoE model.

## 1 Architecture
""" + " ".join(["multi latent attention shrinks the kv cache"] * 12)

# A body long enough that a small max_tokens packs it into several parts and a large one
# packs it into fewer — same shape as test_index.py / test_eval_index_isolated.py's pool,
# needed here to make chunk COUNT observably config-dependent.
_PARA = "the model applies latent attention over compressed key value cache states here now"
_BODY = "\n\n".join(_PARA for _ in range(12))
_LONG_MARKDOWN = f"## Paper A\n\n## 1 Architecture\n\n{_BODY}\n"


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


def test_ingest_paper_reindex_after_chunking_change_does_not_leak_stale_chunks(
    make_config, fake_embedder, monkeypatch
):
    """Re-ingesting a paper under a different chunking.max_tokens (the 'I changed my
    config' scenario --reindex exists for) must leave the collection reflecting only the
    new chunking, not the union of both — the same bug test_index.py proves directly
    against index_markdown, exercised here through the full ingest_paper pipeline."""
    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path, **kw: _LONG_MARKDOWN)
    monkeypatch.setattr(pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: ["moe"])

    paper = Paper(name="paper-a", arxiv_id="0000.00001")

    cfg_small = make_config(chunking=ChunkingCfg(max_tokens=128))
    manifest = Manifest(cfg_small.paths.rag_db)
    collection = open_collection(
        cfg_small.paths.rag_db, cfg_small.collection, embedder_name=fake_embedder.name()
    )
    rec_small = pipeline.ingest_paper(
        paper, cfg_small.for_ingest(), fake_embedder, collection, manifest
    )
    assert collection.count() == rec_small["n_chunks"]

    cfg_large = make_config(chunking=ChunkingCfg(max_tokens=512))
    rec_large = pipeline.ingest_paper(
        paper, cfg_large.for_ingest(), fake_embedder, collection, manifest
    )

    assert rec_large["n_chunks"] < rec_small["n_chunks"]
    assert collection.count() == rec_large["n_chunks"]


def test_ingest_paper_retag_false_preserves_existing_tags(make_config, fake_embedder, monkeypatch):
    """--reindex passes retag=False: chunks are re-embedded under the new config, but
    tags are carried forward from the manifest instead of regenerated (and regenerated
    tags aren't even guaranteed deterministic, so re-running the tagger on every reindex
    would risk silent tag churn unrelated to the chunking change that triggered it)."""
    monkeypatch.setattr(pipeline, "_download", _fake_download)
    monkeypatch.setattr(pipeline, "pdf_to_markdown", lambda path, **kw: _LONG_MARKDOWN)

    paper = Paper(name="paper-a", arxiv_id="0000.00001")
    cfg_small = make_config(chunking=ChunkingCfg(max_tokens=128))
    manifest = Manifest(cfg_small.paths.rag_db)
    collection = open_collection(
        cfg_small.paths.rag_db, cfg_small.collection, embedder_name=fake_embedder.name()
    )

    monkeypatch.setattr(
        pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: ["moe", "attention"]
    )
    rec1 = pipeline.ingest_paper(paper, cfg_small.for_ingest(), fake_embedder, collection, manifest)
    assert rec1["tags"] == ["moe", "attention"]

    # A differently-stubbed generate_tags must NOT be consulted when retag=False.
    monkeypatch.setattr(
        pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: ["should-not-appear"]
    )
    cfg_large = make_config(chunking=ChunkingCfg(max_tokens=512))
    rec2 = pipeline.ingest_paper(
        paper, cfg_large.for_ingest(), fake_embedder, collection, manifest, retag=False
    )

    assert rec2["tags"] == ["moe", "attention"]  # carried forward, not regenerated
    assert rec2["n_chunks"] < rec1["n_chunks"]  # indexing still ran under the new config


def test_ingest_paper_rerenders_when_text_cached_but_display_missing(
    make_config, fake_embedder, monkeypatch
):
    """render_images was turned on for a pool that already has cached text markdown for
    this paper — the display file is still missing, so extraction must run again (to get
    it) even though the RAG text itself doesn't need re-extracting."""
    cfg = make_config(extraction=ExtractionCfg(render_images=True))
    manifest = Manifest(cfg.paths.rag_db)
    collection = open_collection(
        cfg.paths.rag_db, cfg.collection, embedder_name=fake_embedder.name()
    )
    Path(cfg.paths.markdown_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.paths.markdown_dir) / "paper-a.md").write_text(_MARKDOWN)  # cached text only

    monkeypatch.setattr(pipeline, "_download", _fake_download)
    calls = []
    monkeypatch.setattr(
        pipeline, "pdf_to_markdown", lambda path, **kw: calls.append(kw) or _MARKDOWN
    )
    monkeypatch.setattr(pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: [])

    pipeline.ingest_paper(
        Paper(name="paper-a", arxiv_id="0000.00001"),
        cfg.for_ingest(),
        fake_embedder,
        collection,
        manifest,
    )

    assert len(calls) == 1
    assert calls[0]["render_images"] is True
    assert calls[0]["paper_id"] == "paper-a"
    assert calls[0]["display_md_path"].endswith("paper-a_display.md")


def test_ingest_paper_skips_extract_when_text_and_display_both_cached(
    make_config, fake_embedder, monkeypatch
):
    cfg = make_config(extraction=ExtractionCfg(render_images=True))
    manifest = Manifest(cfg.paths.rag_db)
    collection = open_collection(
        cfg.paths.rag_db, cfg.collection, embedder_name=fake_embedder.name()
    )
    Path(cfg.paths.markdown_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.paths.markdown_dir) / "paper-a.md").write_text(_MARKDOWN)
    (Path(cfg.paths.markdown_dir) / "paper-a_display.md").write_text(_MARKDOWN)

    monkeypatch.setattr(pipeline, "_download", _fake_download)

    def _fail(*a, **k):
        raise AssertionError("pdf_to_markdown should not run — both files are cached")

    monkeypatch.setattr(pipeline, "pdf_to_markdown", _fail)
    monkeypatch.setattr(pipeline, "generate_tags", lambda md, spec, existing_tags, **kw: [])

    record = pipeline.ingest_paper(
        Paper(name="paper-a", arxiv_id="0000.00001"),
        cfg.for_ingest(),
        fake_embedder,
        collection,
        manifest,
    )
    assert record["title"] == "Paper A"
