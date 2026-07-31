"""In-process ingestion worker.

Runs pending-paper ingestion on a background thread so Docling/embedding never
block the API event loop. Exposes a thread-safe status snapshot that the admin
panel polls for live progress.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

from rag.config import IngestConfig
from rag.index import open_collection
from rag.manifest import Manifest
from rag.pipeline import (
    build_embedder_from_config,
    ingest_paper,
    normalize_manifest_tags,
    pending_papers,
)


class IngestionWorker:
    def __init__(self, cfg: IngestConfig, manifest: Manifest):
        self.cfg = cfg
        self.manifest = manifest
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "state": "idle",  # idle | running | error
            "total": 0,
            "done": 0,
            "current": None,  # {"name", "stage", "pct"}
            "errors": [],  # [{"name", "error"}]
        }

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._status)
            s["current"] = dict(s["current"]) if s["current"] else None
            s["errors"] = list(s["errors"])
            return s

    def _set(self, **kw) -> None:
        with self._lock:
            self._status.update(kw)

    def _error(self, name: str, err: str) -> None:
        with self._lock:
            self._status["errors"].append({"name": name, "error": err})

    def trigger(self) -> bool:
        """Start a run over currently-pending papers. No-op if already running."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def _run(self) -> None:
        try:
            embedder = None
            collection = None
            # A name this run has already attempted (success or failure) — excluded
            # from every later re-check so a paper that keeps failing can't spin the
            # loop below forever; pending_papers() would keep returning it, since a
            # failed ingest never reaches the manifest write.
            attempted: set[str] = set()
            first_batch = True
            # Loop over batches, not just one pass: a paper can be added (admin
            # add-paper route mutates cfg.papers in place) while this run is already
            # in flight, and trigger() is single-flight — without re-checking here,
            # that paper would sit in `pending` until someone happens to click
            # rescan again. Re-scanning right before going idle picks it up for free.
            while True:
                pending = [
                    p for p in pending_papers(self.cfg, self.manifest) if p.name not in attempted
                ]
                if not pending:
                    # Only reset total/done on the very first (empty) pass — once a
                    # batch has run, its final counts are what the snapshot should
                    # keep reporting, not get zeroed out by this re-check.
                    if first_batch:
                        self._set(total=0, done=0, current=None, state="idle")
                    else:
                        self._set(current=None, state="idle")
                    return
                first_batch = False
                self._set(total=len(pending), done=0, current=None, state="running")

                if embedder is None:
                    embedder = build_embedder_from_config(self.cfg)
                    collection = open_collection(
                        self.cfg.paths.rag_db, self.cfg.collection, embedder_name=embedder.name()
                    )

                for i, paper in enumerate(pending):
                    attempted.add(paper.name)
                    self._set(current={"name": paper.name, "stage": "queued", "pct": 0.0})
                    try:
                        ingest_paper(
                            paper,
                            self.cfg,
                            embedder,
                            collection,
                            self.manifest,
                            on_stage=lambda s, pct, name=paper.name: self._set(
                                current={"name": name, "stage": s, "pct": pct}
                            ),
                        )
                    except Exception as e:
                        self._error(paper.name, str(e))
                    self._set(done=i + 1)

                # Consolidate near-duplicate tags across the library once, after all
                # pending papers in this batch are tagged.
                self._set(current={"name": "library", "stage": "normalize", "pct": 0.0})
                normalize_manifest_tags(self.cfg, self.manifest)
        except Exception:
            self._set(state="error", current=None)
            self._error("worker", traceback.format_exc())
