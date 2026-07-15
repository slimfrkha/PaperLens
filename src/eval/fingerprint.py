"""Corpus fingerprint + pool loading.

The eval set is keyed to the *loaded pool*: swap the papers (or re-extract one) and
the fingerprint changes, so a stale cached set is detected and regenerated. The
fingerprint is over the ingested markdown — the pool the harness actually optimizes
against — not the config's declared ``papers:`` list.
"""

from __future__ import annotations

import glob
import hashlib
import os


def load_pool(markdown_dir: str) -> dict[str, str]:
    """Map ``paper_id -> markdown text`` for every ``*.md`` in the dir.

    ``paper_id`` is the filename stem, matching how ``rag.index`` and the manifest
    key papers elsewhere.
    """
    pool: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(markdown_dir, "*.md"))):
        paper_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            pool[paper_id] = f.read()
    return pool


def corpus_fingerprint(pool: dict[str, str]) -> str:
    """Stable 16-hex-char id of the pool.

    Changes iff the set of paper ids or any markdown content changes, so the eval
    set can detect when it has gone stale against the pool it was built from.
    """
    h = hashlib.sha256()
    for paper_id in sorted(pool):
        h.update(paper_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(hashlib.sha256(pool[paper_id].encode("utf-8")).digest())
    return h.hexdigest()[:16]
