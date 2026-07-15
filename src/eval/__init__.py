"""PaperLens evaluation harness: a per-pool config optimizer.

The harness recalibrates ``config.yaml`` for whatever pool of papers is loaded — it
is not a study of any fixed corpus. It composes ``rag`` (Searcher, llm, chunking,
config) behind a ``paperlens-eval`` CLI; ``rag`` never imports ``eval``.

Phase 1 (this slice): ``gen`` builds a span-anchored QA eval set from the loaded pool.
Gold is a character span in the source markdown (config-independent), so the set
survives a re-chunk and keeps chunking configs comparable.
"""

from __future__ import annotations

from .fingerprint import corpus_fingerprint, load_pool
from .queryset import (
    GenConfig,
    QAItem,
    build_queryset,
    held_out_paper_ids,
    iter_queryset,
    iter_sections,
    split_by_paper,
)

__all__ = [
    "GenConfig",
    "QAItem",
    "build_queryset",
    "corpus_fingerprint",
    "held_out_paper_ids",
    "iter_queryset",
    "iter_sections",
    "load_pool",
    "split_by_paper",
]
