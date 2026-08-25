"""Paper-level metadata store: `rag_db/papers.json`.

Source of truth for the papers list, tags, and ingestion bookkeeping. Read on
every access (no in-memory cache) so the API always sees the worker's latest
writes. Writes are serialized across both threads and processes by a
`filelock.FileLock` on a sibling `papers.json.lock` file — this also guards the
server's `IngestionWorker` against a concurrently-running `paperlens-ingest` CLI
invocation, since both get independent `Manifest` instances over the same file.
Each write is atomic (temp file + rename, like `extract.py`'s display-markdown
write), so reads never need the lock: they only ever see a fully-old or
fully-new file, never a partial one.
"""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock


class Manifest:
    def __init__(self, rag_db: str):
        self.path = Path(rag_db) / "papers.json"
        self._file_lock = FileLock(str(self.path.parent / (self.path.name + ".lock")))

    def _load(self) -> dict[str, dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def _atomic_write(self, data: dict[str, dict]) -> None:
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.path)  # atomic on the same filesystem

    def papers(self) -> list[dict]:
        return list(self._load().values())

    def get(self, paper_id: str) -> dict | None:
        return self._load().get(paper_id)

    def is_ingested(self, paper_id: str) -> bool:
        return paper_id in self._load()

    def upsert(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            data = self._load()
            data[record["paper_id"]] = record
            self._atomic_write(data)

    def remove(self, paper_id: str) -> bool:
        with self._file_lock:
            data = self._load()
            if paper_id not in data:
                return False
            del data[paper_id]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(data)
            return True

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

    def discriminating_tags(self) -> list[dict]:
        """Tags useful as a filter: those *not* present on every paper.

        A tag shared by all papers can't narrow a search, so it's hidden from the
        user-facing list. With a single paper — where every tag is trivially
        universal — nothing is dropped.
        """
        n = len(self._load())
        tags = self.all_tags()
        if n <= 1:
            return tags
        return [t for t in tags if t["count"] < n]

    def paper_ids_for_tags(self, tags: list[str]) -> list[str]:
        """Paper ids tagged with ANY of the given tags (OR semantics)."""
        wanted = set(tags)
        return [
            rec["paper_id"] for rec in self._load().values() if wanted & set(rec.get("tags", []))
        ]
