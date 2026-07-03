"""End-to-end ingestion pipeline: download -> markdown -> (index || tag) -> manifest.

Indexing and tagging are independent given the markdown (tags live in the
manifest, not in chunk metadata), so they run concurrently and meet at the
manifest write. Idempotent per stage (existing PDF/markdown are reused). Shared
by the headless CLI (`rag.ingest`) and the in-process worker (`server.worker`).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .config import IngestConfig, Paper
from .extract import pdf_to_markdown
from .index import index_markdown
from .manifest import Manifest
from .tagger import generate_tags

ARXIV_PDF = "https://arxiv.org/pdf/{id}"

# on_stage(stage_name, fraction_complete)
OnStage = Callable[[str, float], None]


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


def ingest_paper(
    paper: Paper,
    cfg: IngestConfig,
    embedder,
    collection,
    manifest: Manifest,
    on_stage: OnStage | None = None,
) -> dict:
    """Run one paper through the full pipeline and record it in the manifest."""

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
        md = pdf_to_markdown(pdf_path)
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text(md)

    # Index (embedder, compute-bound) and tag (LLM, I/O-bound) are independent
    # given the markdown, so overlap them. The executor context joins the tag
    # thread even if indexing raises, so a failed index still propagates and
    # leaves the paper pending (no manifest write); a failed tag degrades to [].
    def _tags() -> list[str]:
        try:
            return generate_tags(
                md, cfg.tagging, existing_tags=[t["tag"] for t in manifest.all_tags()]
            )
        except Exception as e:  # tagging needs an API key; degrade gracefully
            print(f"  [warn] tag generation failed for {paper.name}: {e}")
            return []

    stage("index", 0.5)
    with ThreadPoolExecutor(max_workers=1) as ex:
        tags_future = ex.submit(_tags)
        n_chunks = index_markdown(
            collection, embedder, md_path, paper.name, batch_size=cfg.embedding.batch_size
        )
        tags = tags_future.result()
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
