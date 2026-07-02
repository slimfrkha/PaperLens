"""End-to-end ingestion pipeline: download -> markdown -> index -> tag -> manifest.

Idempotent per stage (existing PDF/markdown are reused). Shared by the headless
CLI (`rag.ingest`) and the in-process worker (`server.worker`).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable

import httpx

from .config import Config, Paper
from .extract import pdf_to_markdown
from .index import index_markdown, open_collection
from .manifest import Manifest
from .tagger import generate_tags

ARXIV_PDF = "https://arxiv.org/pdf/{id}"

# on_stage(stage_name, fraction_complete)
OnStage = Callable[[str, float], None]


def build_embedder_from_config(cfg: Config):
    """Construct the embedder described by config.embedding (reused across papers)."""
    from .embedders import build_embedder

    e = cfg.embedding
    return build_embedder(
        e.model, e.type, batch_size=e.batch_size,
        api_base=e.api_base, api_key_env=e.api_key_env, max_seq_length=e.max_seq_length,
    )


def pending_papers(cfg: Config, manifest: Manifest) -> list[Paper]:
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
    cfg: Config,
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

    stage("index", 0.5)
    n_chunks = index_markdown(
        collection, embedder, md_path, paper.name, batch_size=cfg.embedding.batch_size
    )

    stage("tag", 0.85)
    try:
        tags = generate_tags(
            md, cfg.llm.tagging, existing_tags=[t["tag"] for t in manifest.all_tags()]
        )
    except Exception as e:  # tagging needs an API key; degrade gracefully
        print(f"  [warn] tag generation failed for {paper.name}: {e}")
        tags = []

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
