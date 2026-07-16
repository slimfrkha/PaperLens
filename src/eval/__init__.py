"""PaperLens evaluation harness: a per-pool config optimizer.

The harness recalibrates ``config.yaml`` for whatever pool of papers is loaded — it
is not a study of any fixed corpus. It composes ``rag`` (Searcher, llm, chunking,
config) behind a ``paperlens-eval`` CLI; ``rag`` never imports ``eval``.

Phase 1: ``gen`` builds a span-anchored QA eval set from the loaded pool. Gold is a
character span in the source markdown (config-independent), so the set survives a
re-chunk and keeps chunking configs comparable.

Phase 2: ``run`` scores one config on the dev set — ``success@candidates`` (stage-1
ceiling) and gold-conditioned ``MRR@k`` (stage-2). Relevance is section identity, and the
section is one relevant unit, so both metrics are chunking-independent.
"""

from __future__ import annotations

from .fingerprint import corpus_fingerprint, load_pool
from .harness import RunReport, run
from .metrics import QueryScore, mrr_at_k, reciprocal_rank, relevant_ids, success_at_candidates
from .queryset import (
    GenConfig,
    QAItem,
    build_queryset,
    held_out_paper_ids,
    iter_queryset,
    iter_sections,
    load_queryset,
    split_by_paper,
)
from .stats import (
    BootResult,
    DeltaResult,
    Sample,
    cluster_bootstrap,
    mdd,
    mrr_samples,
    paired_delta,
    resolution_warning,
    success_samples,
)

__all__ = [
    "BootResult",
    "DeltaResult",
    "GenConfig",
    "QAItem",
    "QueryScore",
    "RunReport",
    "Sample",
    "build_queryset",
    "cluster_bootstrap",
    "corpus_fingerprint",
    "held_out_paper_ids",
    "iter_queryset",
    "iter_sections",
    "load_pool",
    "load_queryset",
    "mdd",
    "mrr_samples",
    "mrr_at_k",
    "paired_delta",
    "reciprocal_rank",
    "relevant_ids",
    "resolution_warning",
    "run",
    "split_by_paper",
    "success_at_candidates",
    "success_samples",
]
