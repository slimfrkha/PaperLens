"""Headless ingestion CLI: run every pending paper through the pipeline.

    python -m rag.ingest                          # ingest papers not yet in the DB
    python -m rag.ingest --config_path other.yaml
    python -m rag.ingest --retag                  # regenerate tags for ingested papers

Accepts draccus per-field overrides too (e.g. ``--llm.tagging.model=...``). Same
code path the app's background worker uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import parse_config
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
    # --retag is a CLI action, not a config field; pull it out before draccus parses.
    argv = [a for a in sys.argv[1:] if a != "--retag"]
    do_retag = "--retag" in sys.argv[1:]

    cfg = parse_config(argv)
    manifest = Manifest(cfg.paths.rag_db)

    if do_retag:
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
