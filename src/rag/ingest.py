"""Headless ingestion CLI: run every pending paper through the pipeline.

    python -m rag.ingest              # ingest papers in config.yaml not yet in the DB
    python -m rag.ingest --config other.yaml
    python -m rag.ingest --retag      # regenerate tags for already-ingested papers

Same code path the app's background worker uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .index import open_collection
from .manifest import Manifest
from .pipeline import build_embedder_from_config, ingest_paper, pending_papers
from .tagger import generate_tags


def retag(cfg, manifest: Manifest) -> None:
    """Regenerate tags for already-ingested papers (no re-indexing)."""
    for rec in manifest.papers():
        md_path = Path(cfg.paths.markdown_dir) / f"{rec['paper_id']}.md"
        if not md_path.exists():
            continue
        tags = generate_tags(
            md_path.read_text(),
            cfg.llm.tagging,
            existing_tags=[t["tag"] for t in manifest.all_tags()],
        )
        rec["tags"] = tags
        manifest.upsert(rec)
        print(f"  {rec['paper_id']}: {', '.join(tags) or '(none)'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--retag",
        action="store_true",
        help="regenerate tags for already-ingested papers, then exit",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    manifest = Manifest(cfg.paths.rag_db)

    if args.retag:
        print("== Regenerating tags ==")
        retag(cfg, manifest)
        return

    pending = pending_papers(cfg, manifest)
    if not pending:
        print("Nothing to ingest — all configured papers are already in the DB.")
        return

    print(f"== Ingesting {len(pending)} paper(s): {', '.join(p.name for p in pending)} ==")
    embedder = build_embedder_from_config(cfg)
    collection = open_collection(cfg.paths.rag_db, cfg.collection, embedder_name=embedder.name())

    for paper in pending:
        print(f"\n-- {paper.name} ({paper.arxiv_id}) --")
        rec = ingest_paper(
            paper,
            cfg,
            embedder,
            collection,
            manifest,
            on_stage=lambda s, pct: print(f"   {s:9s} {int(pct * 100):3d}%"),
        )
        print(f"   -> {rec['n_chunks']} chunks, tags: {', '.join(rec['tags']) or '(none)'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
