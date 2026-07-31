"""File-backed paper annotation store.

Each paper's annotations are one JSON file in the configured `annotations` directory:

    [
      {
        "id": "...", "snippet": "...", "section_title": "...", "section_slug": "...",
        "note": "...", "created_at": "...", "updated_at": "..."
      },
      ...
    ]

`snippet` anchors the annotation to a passage of text (not a chunk id or character offset)
so it survives markdown reflow across re-ingests — the same mechanism the citation
highlighter uses. `section_slug` is the heading id of the section the passage was selected
in; the backend treats it as an opaque string, but the frontend uses it to scope
re-anchoring to that section so a phrase recurring elsewhere in the paper can't cause the
annotation to silently attach to the wrong occurrence.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class AnnotationStore:
    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, paper_id: str) -> Path:
        return self.dir / f"{paper_id}.json"

    def list_all(self, paper_id: str) -> list[dict]:
        p = self._path(paper_id)
        return json.loads(p.read_text()) if p.exists() else []

    def create(
        self, paper_id: str, snippet: str, section_title: str, section_slug: str, note: str
    ) -> dict:
        with self._lock:
            annotations = self.list_all(paper_id)
            annotation = {
                "id": uuid.uuid4().hex[:12],
                "snippet": snippet,
                "section_title": section_title,
                "section_slug": section_slug,
                "note": note,
                "created_at": _now(),
                "updated_at": _now(),
            }
            annotations.append(annotation)
            self._write(paper_id, annotations)
            return annotation

    def update(self, paper_id: str, annotation_id: str, note: str) -> dict | None:
        with self._lock:
            annotations = self.list_all(paper_id)
            for a in annotations:
                if a["id"] == annotation_id:
                    a["note"] = note
                    a["updated_at"] = _now()
                    self._write(paper_id, annotations)
                    return a
            return None

    def delete(self, paper_id: str, annotation_id: str) -> bool:
        with self._lock:
            annotations = self.list_all(paper_id)
            remaining = [a for a in annotations if a["id"] != annotation_id]
            if len(remaining) == len(annotations):
                return False
            self._write(paper_id, remaining)
            return True

    def remove_paper(self, paper_id: str) -> None:
        """Drop every annotation for one paper — used when the paper itself is removed
        (admin remove-paper route), so annotations don't outlive the paper they anchor
        to and keep showing up against a now-404ing paper_id."""
        with self._lock:
            self._path(paper_id).unlink(missing_ok=True)

    def _write(self, paper_id: str, annotations: list[dict]) -> None:
        # Write-temp-then-rename so a crash or full disk mid-write can't leave a
        # truncated JSON file that list() would then fail to parse forever.
        p = self._path(paper_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(annotations, indent=2))
        tmp.replace(p)
