"""Paper-level metadata store: `rag_db/papers.json`.

Source of truth for the papers list, tags, and ingestion bookkeeping. Read on
every access (no in-memory cache) so the API always sees the worker's latest
writes; writes are serialized with a lock.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class Manifest:
    def __init__(self, rag_db: str):
        self.path = Path(rag_db) / "papers.json"
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def papers(self) -> list[dict]:
        return list(self._load().values())

    def get(self, paper_id: str) -> dict | None:
        return self._load().get(paper_id)

    def is_ingested(self, paper_id: str) -> bool:
        return paper_id in self._load()

    def upsert(self, record: dict) -> None:
        with self._lock:
            data = self._load()
            data[record["paper_id"]] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2))

    def all_tags(self) -> list[dict]:
        """Tags with paper counts, most common first."""
        counts: dict[str, int] = {}
        for rec in self._load().values():
            for t in rec.get("tags", []):
                counts[t] = counts.get(t, 0) + 1
        return sorted(
            ({"tag": t, "count": c} for t, c in counts.items()),
            key=lambda x: (-x["count"], x["tag"]),
        )

    def paper_ids_for_tags(self, tags: list[str]) -> list[str]:
        """Paper ids tagged with ANY of the given tags (OR semantics)."""
        wanted = set(tags)
        return [
            rec["paper_id"]
            for rec in self._load().values()
            if wanted & set(rec.get("tags", []))
        ]
