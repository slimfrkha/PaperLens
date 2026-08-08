"""End-to-end ingestion pipeline: download -> markdown -> (index || tag) -> manifest.

Indexing and tagging are independent given the markdown (tags live in the
manifest, not in chunk metadata), so they run concurrently and meet at the
manifest write. Idempotent per stage (existing PDF/markdown are reused).
`ingest_paper` runs one paper through the stages; `run_batch` runs a batch of
papers through it, owning the embedder/collection lifecycle. Shared by the
headless CLI (`rag.ingest`) and the in-process worker (`server.worker`).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import IngestConfig, Paper
from .extract import pdf_to_markdown
from .index import index_markdown, open_collection
from .manifest import Manifest
from .tagger import generate_tags, normalize_tags

ARXIV_PDF = "https://arxiv.org/pdf/{id}"

# on_stage(stage_name, fraction_complete)
OnStage = Callable[[str, float], None]

# on_paper_start(paper)
OnPaperStart = Callable[[Paper], None]
# on_paper_stage(paper, stage_name, fraction_complete)
OnPaperStage = Callable[[Paper, str, float], None]
# on_paper_done(paper, record_or_None_on_failure, exception_or_None_on_success)
OnPaperDone = Callable[[Paper, dict | None, Exception | None], None]


def build_embedder_from_config(cfg: IngestConfig):
    """Construct the embedder described by config.embedding (reused across papers)."""
    from .embedders import build_embedder

    return build_embedder(cfg.embedding)


def pending_papers(cfg: IngestConfig, manifest: Manifest) -> list[Paper]:
    """Config papers not yet in the manifest."""
    return [p for p in cfg.papers if not manifest.is_ingested(p.name)]


def _download(arxiv_id: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET", ARXIV_PDF.format(id=arxiv_id), follow_redirects=True, timeout=120
    ) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def _title(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    for ln in md.splitlines():
        if ln.startswith("## "):
            return ln[3:].strip()
    return ""


def _existing_tags(manifest: Manifest, paper_id: str) -> list[str]:
    rec = next((r for r in manifest.papers() if r["paper_id"] == paper_id), None)
    return rec["tags"] if rec else []


def ingest_paper(
    paper: Paper,
    cfg: IngestConfig,
    embedder,
    collection,
    manifest: Manifest,
    on_stage: OnStage | None = None,
    retag: bool = True,
) -> dict:
    """Run one paper through the full pipeline and record it in the manifest.

    When ``retag`` is False (used by ``--reindex``), tags are carried forward from the
    existing manifest record instead of regenerated — re-chunking/re-embedding shouldn't
    also churn tags or spend an LLM call unrelated to the change that triggered it.
    """

    def stage(name: str, pct: float = 0.0):
        if on_stage:
            on_stage(name, pct)

    pdf_path = os.path.join(cfg.paths.pdf_dir, f"{paper.name}.pdf")
    md_path = os.path.join(cfg.paths.markdown_dir, f"{paper.name}.md")

    stage("download", 0.0)
    _download(paper.arxiv_id, pdf_path)

    stage("extract", 0.25)
    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        md = Path(md_path).read_text()
    else:
        md = pdf_to_markdown(pdf_path, ocr_enabled=cfg.extraction.ocr_enabled)
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text(md)

    # Index (embedder, compute-bound) and tag (LLM, I/O-bound) are independent
    # given the markdown, so overlap them. The executor context joins the tag
    # thread even if indexing raises, so a failed index still propagates and
    # leaves the paper pending (no manifest write); a failed tag degrades to [].
    def _tags() -> list[str]:
        try:
            return generate_tags(
                md,
                cfg.tagging,
                existing_tags=[t["tag"] for t in manifest.all_tags()],
                max_tags=cfg.tagger.max_tags,
                min_tags=cfg.tagger.min_tags,
                max_excerpt_chars=cfg.tagger.max_excerpt_chars,
            )
        except Exception as e:  # tagging needs an API key; degrade gracefully
            print(f"  [warn] tag generation failed for {paper.name}: {e}")
            return []

    stage("index", 0.5)
    if retag:
        with ThreadPoolExecutor(max_workers=1) as ex:
            tags_future = ex.submit(_tags)
            n_chunks = index_markdown(
                collection,
                embedder,
                md_path,
                paper.name,
                batch_size=cfg.embedding.batch_size,
                chunking=cfg.chunking,
            )
            tags = tags_future.result()
    else:
        n_chunks = index_markdown(
            collection,
            embedder,
            md_path,
            paper.name,
            batch_size=cfg.embedding.batch_size,
            chunking=cfg.chunking,
        )
        tags = _existing_tags(manifest, paper.name)
    stage("tag", 0.85)

    record = {
        "paper_id": paper.name,
        "name": paper.name,
        "title": _title(md) or paper.name,
        "arxiv_id": paper.arxiv_id,
        "tags": tags,
        "n_chunks": n_chunks,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest.upsert(record)
    stage("done", 1.0)
    return record


@dataclass
class BatchResult:
    records: list[dict]
    embedder: Any
    collection: Any


def run_batch(
    cfg: IngestConfig,
    manifest: Manifest,
    papers: list[Paper],
    embedder=None,
    collection=None,
    on_paper_start: OnPaperStart | None = None,
    on_stage: OnPaperStage | None = None,
    on_paper_done: OnPaperDone | None = None,
    retag: bool = True,
    stop_on_error: bool = True,
) -> BatchResult:
    """Run ``papers`` through ``ingest_paper``, building embedder/collection if not supplied.

    The CLI (plain-ingest, ``--reindex``) and the background worker all build-if-needed,
    loop, and report per paper — they differ only in *how* they report
    (``on_paper_start``/``on_stage``/``on_paper_done``) and in what a per-paper failure
    should do to the batch (``stop_on_error``).

    ``stop_on_error=True`` (default, the CLI's fail-fast behavior) re-raises the first
    per-paper exception after notifying ``on_paper_done`` — no manifest write for that
    paper, the run crashes with a traceback. ``stop_on_error=False`` (the worker) instead
    swallows it there and continues the batch.

    Embedder/collection are built once if not passed in (each independently — passing
    one without the other reuses just that one). A caller that re-polls for new papers
    across multiple ``run_batch`` calls (the worker) passes the previous call's
    ``result.embedder``/``result.collection`` back in to avoid rebuilding.

    Does NOT call ``normalize_manifest_tags`` — callers want different reporting around
    it (and ``--reindex`` skips it entirely), so it stays a separate call after the batch.
    """
    if embedder is None:
        embedder = build_embedder_from_config(cfg)
    if collection is None:
        collection = open_collection(
            cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name()
        )

    records: list[dict] = []
    for paper in papers:
        if on_paper_start:
            on_paper_start(paper)
        try:
            rec = ingest_paper(
                paper,
                cfg,
                embedder,
                collection,
                manifest,
                on_stage=(lambda s, pct, _p=paper: on_stage(_p, s, pct)) if on_stage else None,
                retag=retag,
            )
        except Exception as exc:
            if on_paper_done:
                on_paper_done(paper, None, exc)
            if stop_on_error:
                raise
            continue
        records.append(rec)
        if on_paper_done:
            on_paper_done(paper, rec, None)

    return BatchResult(records=records, embedder=embedder, collection=collection)


def normalize_manifest_tags(cfg: IngestConfig, manifest: Manifest) -> dict[str, str]:
    """Consolidate near-duplicate tags across the whole library, in place.

    Builds one LLM-generated ``{tag -> canonical}`` map over the full vocabulary and
    rewrites every paper's tags through it (de-duplicating, order preserved). A no-op
    when nothing merges or tagging is unavailable — never wipes existing tags.
    """
    vocab = [t["tag"] for t in manifest.all_tags()]
    try:
        mapping = normalize_tags(vocab, cfg.tagging)
    except Exception as e:  # tagging needs an API key; degrade gracefully
        print(f"  [warn] tag normalization skipped: {e}")
        return {}
    if not mapping:
        return {}
    for rec in manifest.papers():
        tags = rec.get("tags", [])
        merged = list(dict.fromkeys(mapping.get(t, t) for t in tags))
        if merged != tags:
            rec["tags"] = merged
            manifest.upsert(rec)
    return mapping
