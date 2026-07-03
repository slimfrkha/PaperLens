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
from rag.pipeline import build_embedder_from_config, ingest_paper, pending_papers


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
            pending = pending_papers(self.cfg, self.manifest)
            self._set(
                total=len(pending), done=0, current=None, state="running" if pending else "idle"
            )
            if not pending:
                return

            embedder = build_embedder_from_config(self.cfg)
            collection = open_collection(
                self.cfg.paths.rag_db, self.cfg.collection, embedder_name=embedder.name()
            )

            for i, paper in enumerate(pending):
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

            self._set(state="idle", current=None)
        except Exception:
            self._set(state="error", current=None)
            self._error("worker", traceback.format_exc())
