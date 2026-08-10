"""paperlens-eval — per-pool config-optimization harness.

``gen`` builds the span-anchored QA eval set from the loaded pool and writes it, keyed on
the corpus fingerprint, under ``evals/`` at the project root::

    paperlens-eval gen                 # discover config.yaml, build the set
    paperlens-eval gen --config path/to/config.yaml

``run`` scores one config; ``screen`` / ``sweep`` optimize retrieval knobs (reranker,
candidates) and chunking knobs over the pool; ``confirm`` validates the winner on the
held-out test split.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import draccus

from rag.config import ChunkingCfg, Config, EmbeddingCfg, HFEmbeddingCfg, RerankerCfg, load_config
from rag.llm import build_llm
from rag.reranker import Reranker
from rag.search import Searcher

from .comparative_queryset import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_N_PAPERS_MAX,
    DEFAULT_N_PAPERS_MIN,
    DEFAULT_TARGET_P,
    ComparativeGenConfig,
    ComparativeQAItem,
    build_comparative_queryset,
    comparative_item_to_dict,
    load_comparative_queryset,
)
from .fingerprint import corpus_fingerprint, load_pool
from .genfilter import GenFilterConfig, check_leak
from .harness import (
    DEFAULT_PER_PAPER_N,
    comparative_confirm,
    comparative_sweep,
    format_comparative_confirm,
    format_comparative_sweep,
    format_elbow_screen_report,
    format_per_paper_confirm,
    format_per_paper_sweep,
    format_report,
    per_paper_confirm,
    per_paper_sweep,
    run,
    screen_elbow,
)
from .index_isolated import build_isolated_searcher, chunks_for
from .optimizer import (
    DEFAULT_CANDIDATE_GRID,
    DEFAULT_CHUNK_GRIDS,
    DEFAULT_MAX_TOKENS_GRID,
    build_reranker_for_cfg,
    chunking_arms,
    format_chunking_report,
    format_screen_report,
    screen_chunking,
    screen_retrieval,
    sweep,
)
from .queryset import GenConfig, held_out_paper_ids, item_to_dict, iter_queryset, load_queryset


def cmd_gen(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pool = load_pool(cfg.paths.markdown_dir)
    if not pool:
        raise SystemExit(
            f"No markdown papers in {cfg.paths.markdown_dir} — ingest the pool first "
            f"(uv run paperlens-ingest)."
        )
    if args.limit:
        pool = dict(list(pool.items())[: args.limit])

    fingerprint = corpus_fingerprint(pool)
    gen = GenConfig()
    gfcfg = GenFilterConfig(enabled=args.genfilter)
    if args.genfilter_threshold is not None:
        gfcfg = replace(gfcfg, match_threshold=args.genfilter_threshold)
    test_ids = held_out_paper_ids(pool, gen)
    out_dir = Path(cfg.root) / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / f"{fingerprint}.dev.jsonl"
    test_path = out_dir / f"{fingerprint}.test.jsonl"
    genfilter_path = out_dir / f"{fingerprint}.genfilter.jsonl"

    smoke = "  [smoke: --limit]" if args.limit else ""
    flt = f"  [genfilter on, threshold={gfcfg.match_threshold}]" if gfcfg.enabled else ""
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}{smoke}{flt}")
    print(f"Generating with {cfg.llm.tagging.model} (one question per section)...")

    llm = build_llm(cfg.llm.tagging)
    n_dev = n_test = n_filtered = 0
    # Stream to disk as questions are produced: a crash keeps partial progress, and a
    # per-section failure in iter_queryset is skipped rather than losing the batch.
    # genfilter_path is opened unconditionally (simpler than an Optional-file dance) but
    # only written to when gfcfg.enabled — an empty file when --genfilter is off is an
    # honest "nothing was checked" signal, not a file that sometimes doesn't exist.
    with (
        open(dev_path, "w", encoding="utf-8") as fdev,
        open(test_path, "w", encoding="utf-8") as ftest,
        open(genfilter_path, "w", encoding="utf-8") as faudit,
    ):
        for it in iter_queryset(pool, llm, gen, show_progress=True):
            if gfcfg.enabled:
                check = check_leak(it, llm, gfcfg.match_threshold)
                faudit.write(
                    json.dumps(
                        {
                            "query": it.query,
                            "paper_id": it.paper_id,
                            "gold_answer": it.answer,
                            "closed_book_answer": check.predicted,
                            "score": check.score,
                            "leaked": check.leaked,
                            "error": check.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if check.leaked:
                    n_filtered += 1
                    continue
            line = json.dumps(item_to_dict(it), ensure_ascii=False) + "\n"
            if it.paper_id in test_ids:
                ftest.write(line)
                n_test += 1
            else:
                fdev.write(line)
                n_dev += 1

    (out_dir / f"{fingerprint}.meta.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_papers": len(pool),
                "n_questions": n_dev + n_test,
                "n_dev": n_dev,
                "n_test": n_test,
                "gen_config": asdict(gen),
                "gen_model": cfg.llm.tagging.model,
                "limit": args.limit,
                "genfilter": {
                    "enabled": gfcfg.enabled,
                    "match_threshold": gfcfg.match_threshold,
                    "n_filtered": n_filtered,
                },
            },
            indent=2,
        )
    )
    filt_msg = (
        f"  ({n_filtered} discarded as closed-book-answerable — see {genfilter_path.name})"
        if gfcfg.enabled
        else ""
    )
    print(f"Wrote {n_dev} dev / {n_test} test questions to {out_dir}{filt_msg}")


def _load_dev_set(args: argparse.Namespace):
    """Discover config, load the pool, and load the dev split keyed on its fingerprint.

    Shared by ``run`` and ``screen``: both score the loaded pool's dev set. Returns
    ``(cfg, pool, fingerprint, items)``; exits with a clear message if the pool is
    un-ingested or the eval set has not been generated for it yet.
    """
    cfg = load_config(args.config)
    pool = load_pool(cfg.paths.markdown_dir)
    if not pool:
        raise SystemExit(
            f"No markdown papers in {cfg.paths.markdown_dir} — ingest the pool first "
            f"(uv run paperlens-ingest)."
        )
    if args.limit:
        pool = dict(list(pool.items())[: args.limit])

    fingerprint = corpus_fingerprint(pool)
    dev_path = Path(cfg.root) / "evals" / f"{fingerprint}.dev.jsonl"
    if not dev_path.exists():
        raise SystemExit(
            f"No eval set for this pool (fingerprint={fingerprint}) at {dev_path} — "
            f"generate it first with `paperlens-eval gen`."
        )
    return cfg, pool, fingerprint, load_queryset(str(dev_path))


def _checkpoint_path(cfg: Config, fingerprint: str, name: str) -> Path:
    """Where a resumable command's ``.ckpt.jsonl`` lives — see ``eval.checkpoint``.

    Distinct from every other fingerprint-keyed file under ``evals/`` (dev/test/meta/
    genfilter/confirm marker): these are transient, deleted by the command itself on a
    fully successful run, and safe to delete by hand at any time (``--fresh`` does
    exactly that before starting).
    """
    return Path(cfg.root) / "evals" / f"{fingerprint}.{name}.ckpt.jsonl"


def cmd_run(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    print("Running default config on the dev split...")
    checkpoint_path = _checkpoint_path(cfg, fingerprint, "run")
    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)
    print(format_report(run(cfg, items, desc="scoring dev set", checkpoint_path=checkpoint_path)))


def _parse_grid(raw: str | None) -> list[int] | None:
    """Parse ``--candidates`` into a sorted, de-duplicated list of positive ints (else None).

    Tolerates whitespace and stray/trailing commas; rejects non-integer or non-positive
    values with a clean message rather than a raw traceback.
    """
    if not raw:
        return None
    try:
        vals = [int(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError as e:
        raise SystemExit(f"--candidates must be comma-separated integers, got: {raw!r}") from e
    if any(v <= 0 for v in vals):
        raise SystemExit(f"--candidates values must be positive, got: {raw!r}")
    return sorted(set(vals)) or None


def _parse_float_grid(raw: str | None, *, flag: str) -> list[float] | None:
    """Parse a ``--mad-multiplier``/``--prominence``-style flag into positive floats."""
    if not raw:
        return None
    try:
        vals = [float(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError as e:
        raise SystemExit(f"{flag} must be comma-separated numbers, got: {raw!r}") from e
    if any(v <= 0 for v in vals):
        raise SystemExit(f"{flag} values must be positive, got: {raw!r}")
    return sorted(set(vals)) or None


def _print_index_sizes(pool: dict[str, str], cells: list[tuple[str, ChunkingCfg]]) -> None:
    """Model-free preamble: each cell's chunk count and Δ vs the default, so a null in the
    report is read correctly — an inert knob (index barely moved) is distinguished from a
    live knob that moved the index yet showed no retrieval effect (a real finding). Cheap
    (``chunks_for`` needs no models), so it prints before any weights load.
    """
    default_n = len(chunks_for(pool, cells[0][1]))
    parts = [f"{cells[0][0]}={default_n}"]
    for label, chunking in cells[1:]:
        n = len(chunks_for(pool, chunking))
        parts.append(f"{label}={n}({n - default_n:+d})")
    print("  index sizes (chunks): " + "  ".join(parts))


def _chunking_grids(args: argparse.Namespace) -> dict[str, list[float]] | None:
    """Screen grids with an optional ``--max-tokens`` override of the headline knob."""
    mt = _parse_grid(args.max_tokens)
    if mt is None:
        return None  # DEFAULT_CHUNK_GRIDS
    grids = {k: list(v) for k, v in DEFAULT_CHUNK_GRIDS.items()}
    grids["max_tokens"] = list(mt)
    return grids


def cmd_screen(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    if args.tier == "retrieval":
        grid = _parse_grid(args.candidates)
        arm_msg = "".join(
            [" + hybrid=on" if args.hybrid else "", " + multi_query=on" if args.multi_query else ""]
        )
        print(f"Screening retrieval knobs (reranker/candidates{arm_msg}) on the dev split...")
        checkpoint_path = _checkpoint_path(cfg, fingerprint, "screen-retrieval")
        if args.fresh:
            checkpoint_path.unlink(missing_ok=True)
        print(
            format_screen_report(
                screen_retrieval(
                    cfg,
                    items,
                    candidate_grid=grid,
                    show_progress=True,
                    hybrid=args.hybrid,
                    multi_query=args.multi_query,
                    checkpoint_path=checkpoint_path,
                )
            )
        )
        return
    if args.tier == "elbow":
        mad_grid = _parse_float_grid(args.mad_multiplier, flag="--mad-multiplier")
        prom_grid = _parse_float_grid(args.prominence, flag="--prominence")
        print("Screening elbow knobs (mad_multiplier/prominence) on the dev split...")
        print(
            format_elbow_screen_report(
                screen_elbow(
                    cfg,
                    items,
                    mad_grid=mad_grid,
                    prominence_grid=prom_grid,
                    show_progress=True,
                )
            )
        )
        return
    # Chunking: each arm is an isolated re-index — print the cell count up front (not a
    # fabricated ETA), then run in a throwaway temp dir (the prod collection is never touched).
    grids = _chunking_grids(args)
    arms = chunking_arms(cfg, grids=grids)
    print(
        f"Screening chunking knobs on the dev split: {len(arms)} cells, "
        f"re-indexing {len(pool)} papers each..."
    )
    _print_index_sizes(pool, [(a.label, a.chunking) for a in arms])
    checkpoint_path = _checkpoint_path(cfg, fingerprint, "screen-chunking")
    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="paperlens-eval-") as tmp:
        t0 = time.time()
        report = screen_chunking(
            cfg,
            pool,
            items,
            db_dir=tmp,
            grids=grids,
            show_progress=True,
            checkpoint_path=checkpoint_path,
        )
        print(format_chunking_report(report))
        print(f"  ({time.time() - t0:.1f}s wall)")


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    mt_grid = _parse_grid(args.max_tokens) or DEFAULT_MAX_TOKENS_GRID
    cand_grid = _parse_grid(args.candidates) or None
    cells = sorted({cfg.chunking.max_tokens, *mt_grid})
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    print(
        f"Sweep: re-indexing {len(cells)} cells (max_tokens={cells}) over {len(pool)} papers; "
        f"candidates × rerank slices derived from cache (no extra re-index)..."
    )
    _print_index_sizes(
        pool,
        [("default", cfg.chunking)]
        + [
            (f"max_tokens={mt}", replace(cfg.chunking, max_tokens=mt))
            for mt in cells
            if mt != cfg.chunking.max_tokens
        ],
    )
    checkpoint_dir = Path(cfg.root) / "evals"
    if args.fresh:
        for mt in cells:
            (checkpoint_dir / f"{fingerprint}.sweep.mt{mt}.ckpt.jsonl").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="paperlens-eval-") as tmp:
        t0 = time.time()
        report = sweep(
            cfg,
            pool,
            items,
            db_dir=tmp,
            max_tokens_grid=mt_grid,
            candidate_grid=cand_grid,
            show_progress=True,
            checkpoint_dir=checkpoint_dir,
            fingerprint=fingerprint,
        )
        print(format_chunking_report(report))
        print(f"  ({time.time() - t0:.1f}s wall)")


def cmd_per_paper_sweep(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    n_papers = args.per_paper_n
    grid = _parse_grid(args.candidates) or DEFAULT_CANDIDATE_GRID
    if grid[0] < cfg.retrieval.max_k:
        raise SystemExit(
            f"--candidates includes {grid[0]}, below retrieval.max_k={cfg.retrieval.max_k} — "
            f"the per_paper floor (max(max_k, ...)) would make each paper's fetch exceed its "
            f"nominal candidates share, confounding the pool-size comparison."
        )
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    print(
        f"per-paper sweep: n_papers={n_papers}  candidates={grid}  seed={args.seed} — "
        f"exploratory, standalone (not part of gen→run→screen→sweep→confirm; see "
        f"per-paper-eval-spec.md). This is the most expensive command in the harness: up "
        f"to {len(items)} questions × (1 off-arm + up to 2 on-arm variants) × {len(grid)} "
        f"candidates points, each on-arm point running {n_papers} scoped retrievals."
    )
    checkpoint_path = _checkpoint_path(cfg, fingerprint, "per-paper-sweep")
    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)
    report = per_paper_sweep(
        cfg,
        items,
        pool,
        n_papers=n_papers,
        candidates_grid=grid,
        seed=args.seed,
        show_progress=True,
        checkpoint_path=checkpoint_path,
    )
    print(format_per_paper_sweep(report))


def cmd_per_paper_confirm(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    if args.candidates < cfg.retrieval.max_k:
        raise SystemExit(
            f"--candidates={args.candidates} is below retrieval.max_k={cfg.retrieval.max_k} — "
            f"the per_paper floor (max(max_k, ...)) would make each paper's fetch exceed its "
            f"nominal candidates share, confounding the pool-size comparison."
        )
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    print(
        f"per-paper confirm: candidates={args.candidates}  variant={args.variant}  "
        f"n_papers={args.per_paper_n}  seed={args.seed} (must be a fresh seed, independent "
        f"of the sweep being confirmed — dev split, not held-out test; see "
        f"per-paper-eval-spec.md)"
    )
    result = per_paper_confirm(
        cfg,
        items,
        pool,
        n_papers=args.per_paper_n,
        candidates=args.candidates,
        variant=args.variant,
        seed=args.seed,
    )
    print(format_per_paper_confirm(result, n_papers=args.per_paper_n, seed=args.seed))


def cmd_comparative_gen(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pool = load_pool(cfg.paths.markdown_dir)
    if not pool:
        raise SystemExit(
            f"No markdown papers in {cfg.paths.markdown_dir} — ingest the pool first "
            f"(uv run paperlens-ingest)."
        )
    if args.limit:
        pool = dict(list(pool.items())[: args.limit])
    if args.n_papers_min > args.n_papers_max:
        raise SystemExit(
            f"--n-papers-min ({args.n_papers_min}) must be <= --n-papers-max ({args.n_papers_max})."
        )

    fingerprint = corpus_fingerprint(pool)
    gen_cfg = ComparativeGenConfig(
        target_p=args.target_p,
        max_trials=args.max_trials,
        n_papers_min=args.n_papers_min,
        n_papers_max=args.n_papers_max,
        seed=args.seed,
    )
    out_dir = Path(cfg.root) / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / f"{fingerprint}.comparative.dev.jsonl"
    test_path = out_dir / f"{fingerprint}.comparative.test.jsonl"

    smoke = "  [smoke: --limit]" if args.limit else ""
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}{smoke}")
    print(
        f"comparative gen: target_p={gen_cfg.target_p}  max_trials={gen_cfg.max_trials}  "
        f"n_papers={gen_cfg.n_papers_min}..{gen_cfg.n_papers_max}  seed={gen_cfg.seed} — "
        f"up to {gen_cfg.max_trials} spotting calls + up to {gen_cfg.target_p} writing "
        f"calls, LLM={cfg.llm.tagging.model}. Recommended first run: a cheap pilot "
        f"(--target-p 5 --max-trials 20) before trusting a full run's worth of LLM calls "
        f'to an unvalidated prompt — see comparative-eval-spec.md\'s "CLI". Check TWO '
        f"things in the pilot's output, not just the hit rate: read a sample of the "
        f"written questions in dev.jsonl (body is kept in each section precisely for "
        f"this) and confirm by eye that they genuinely need every section shown, not "
        f"just the best one — an unaudited write call can produce a 'comparative' "
        f"question actually answerable from a single paper, which the strict-AND metric "
        f"then over-penalizes for no real reason."
    )

    llm = build_llm(cfg.llm.tagging)
    split_rng = random.Random(gen_cfg.seed)
    n_dev = n_test = 0

    with (
        open(dev_path, "w", encoding="utf-8") as fdev,
        open(test_path, "w", encoding="utf-8") as ftest,
    ):

        def _on_item(it: ComparativeQAItem) -> None:
            nonlocal n_dev, n_test
            line = json.dumps(comparative_item_to_dict(it), ensure_ascii=False) + "\n"
            if split_rng.random() < gen_cfg.test_frac:
                ftest.write(line)
                n_test += 1
            else:
                fdev.write(line)
                n_dev += 1

        result = build_comparative_queryset(
            pool, llm, gen_cfg, show_progress=True, on_item=_on_item
        )

    (out_dir / f"{fingerprint}.comparative.meta.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_papers": len(pool),
                "n_items": len(result.items),
                "n_trials": result.trials,
                "n_dev": n_dev,
                "n_test": n_test,
                "gen_config": asdict(gen_cfg),
                "gen_model": cfg.llm.tagging.model,
                "limit": args.limit,
                "paper_pair_frequency": {
                    f"{a},{b}": n for (a, b), n in result.paper_pair_frequency.most_common()
                },
            },
            indent=2,
        )
    )

    hit_rate = len(result.items) / result.trials if result.trials else 0.0
    print(
        f"Wrote {n_dev} dev / {n_test} test comparative items from {len(result.items)} "
        f"successful match groups across {result.trials} trials (hit rate: {hit_rate:.0%})."
    )
    top_pairs = result.paper_pair_frequency.most_common(5)
    if top_pairs:
        pairs_str = "  ".join(f"({a}, {b})={n}" for (a, b), n in top_pairs)
        print(f"  paper-pair frequency (top 5): {pairs_str}")
        print(
            "  (counts every pairwise combination inside a match group, so a single "
            "N-paper item contributes C(N,2) pairs, not 1 -- a few large-group items can "
            "outweigh many small ones here without the pool actually being that skewed)"
        )


def _load_comparative_dev_set(args: argparse.Namespace):
    """Mirrors ``_load_dev_set`` exactly except for the split file/loader — comparative
    items have their own dev/test split, separate from the main pipeline's (different
    item shape; see comparative-eval-spec.md's "Disk layout")."""
    cfg = load_config(args.config)
    pool = load_pool(cfg.paths.markdown_dir)
    if not pool:
        raise SystemExit(
            f"No markdown papers in {cfg.paths.markdown_dir} — ingest the pool first "
            f"(uv run paperlens-ingest)."
        )
    if args.limit:
        pool = dict(list(pool.items())[: args.limit])

    fingerprint = corpus_fingerprint(pool)
    dev_path = Path(cfg.root) / "evals" / f"{fingerprint}.comparative.dev.jsonl"
    if not dev_path.exists():
        raise SystemExit(
            f"No comparative eval set for this pool (fingerprint={fingerprint}) at "
            f"{dev_path} — generate it first with `paperlens-eval comparative gen`."
        )
    return cfg, pool, fingerprint, load_comparative_queryset(str(dev_path))


def _load_comparative_test_set(args: argparse.Namespace):
    """Mirrors ``_load_comparative_dev_set`` exactly except for the split file — used
    only by ``confirm``, the one command allowed to touch it."""
    cfg = load_config(args.config)
    pool = load_pool(cfg.paths.markdown_dir)
    if not pool:
        raise SystemExit(
            f"No markdown papers in {cfg.paths.markdown_dir} — ingest the pool first "
            f"(uv run paperlens-ingest)."
        )
    if args.limit:
        pool = dict(list(pool.items())[: args.limit])

    fingerprint = corpus_fingerprint(pool)
    test_path = Path(cfg.root) / "evals" / f"{fingerprint}.comparative.test.jsonl"
    if not test_path.exists():
        raise SystemExit(
            f"No comparative eval set for this pool (fingerprint={fingerprint}) at "
            f"{test_path} — generate it first with `paperlens-eval comparative gen`."
        )
    return cfg, pool, fingerprint, load_comparative_queryset(str(test_path))


def cmd_comparative_sweep(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_comparative_dev_set(args)
    grid = _parse_grid(args.candidates) or DEFAULT_CANDIDATE_GRID
    if grid[0] < cfg.retrieval.max_k:
        raise SystemExit(
            f"--candidates includes {grid[0]}, below retrieval.max_k={cfg.retrieval.max_k} — "
            f"the per_paper floor (max(max_k, ...)) would make each paper's fetch exceed its "
            f"nominal candidates share, confounding the pool-size comparison."
        )
    print(
        f"Pool: {len(pool)} papers  fingerprint={fingerprint}  "
        f"comparative dev={len(items)} questions"
    )
    # From the actually-loaded items, not the default --n-papers-min/--n-papers-max
    # range -- gen may have used a different range, and a cost estimate that guesses
    # wrong defeats the point of printing one before the sweep runs.
    group_sizes = [len(it.sections) for it in items]
    n_papers_desc = f"{min(group_sizes)}-{max(group_sizes)}" if group_sizes else "0-0"
    print(
        f"comparative sweep: candidates={grid} — exploratory, standalone (not part of "
        f"gen→run→screen→sweep→confirm; see comparative-eval-spec.md). Up to "
        f"{len(items)} questions × (1 off-arm + up to 2 on-arm variants) × {len(grid)} "
        f"candidates points, each on-arm point running up to n_papers ({n_papers_desc}, "
        f"per this dev split) scoped retrievals per item."
    )
    checkpoint_path = _checkpoint_path(cfg, fingerprint, "comparative-sweep")
    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)
    report = comparative_sweep(
        cfg, items, candidates_grid=grid, show_progress=True, checkpoint_path=checkpoint_path
    )
    print(format_comparative_sweep(report))


def cmd_comparative_confirm(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_comparative_test_set(args)
    if args.candidates < cfg.retrieval.max_k:
        raise SystemExit(
            f"--candidates={args.candidates} is below retrieval.max_k={cfg.retrieval.max_k} — "
            f"the per_paper floor (max(max_k, ...)) would make each paper's fetch exceed its "
            f"nominal candidates share, confounding the pool-size comparison."
        )
    print(
        f"Pool: {len(pool)} papers  fingerprint={fingerprint}  "
        f"comparative test={len(items)} questions"
    )
    print(
        f"comparative confirm: candidates={args.candidates}  variant={args.variant}  "
        f"(test split, touched once; see comparative-eval-spec.md)"
    )
    result = comparative_confirm(cfg, items, candidates=args.candidates, variant=args.variant)
    print(format_comparative_confirm(result, candidates=args.candidates))


def _load_test_set(args: argparse.Namespace):
    """Discover config, load the pool, and load the held-out test split.

    Mirrors ``_load_dev_set`` exactly except for the split file — used only by ``confirm``,
    the one command allowed to touch it.
    """
    cfg = load_config(args.config)
    pool = load_pool(cfg.paths.markdown_dir)
    if not pool:
        raise SystemExit(
            f"No markdown papers in {cfg.paths.markdown_dir} — ingest the pool first "
            f"(uv run paperlens-ingest)."
        )
    if args.limit:
        pool = dict(list(pool.items())[: args.limit])

    fingerprint = corpus_fingerprint(pool)
    test_path = Path(cfg.root) / "evals" / f"{fingerprint}.test.jsonl"
    if not test_path.exists():
        raise SystemExit(
            f"No eval set for this pool (fingerprint={fingerprint}) at {test_path} — "
            f"generate it first with `paperlens-eval gen`."
        )
    return cfg, pool, fingerprint, load_queryset(str(test_path))


# Not resolved here: the harness scores whole questions, but production decomposes them
# into short, keyword-shaped sub-queries before retrieving (src/server/agent.py:126-138) —
# a confirmed ~4x length/shape gap. Printed so a recommendation is never read as fully
# validated against the deployed retriever.
_ESTIMAND_CAVEAT = (
    "estimand caveat: scored on whole questions; production retrieves on decomposed "
    "sub-queries (agent.py:126-138) — this ranking is not yet validated against that gap."
)


@dataclass
class _ConfigBlock:
    chunking: ChunkingCfg
    embedding: EmbeddingCfg
    reranker: RerankerCfg


def format_config_block(
    cfg: Config, *, chunking: ChunkingCfg, candidates: int, rerank: bool
) -> str:
    """A paste-ready ``config.yaml`` block for the confirmed config.

    ``chunking``/``embedding``/``reranker`` are dumped via ``draccus.dump`` (correctly emits
    each ``ChoiceRegistry`` variant's ``type:`` discriminator when the dataclass field is typed
    as the base class, as here). ``retrieval.candidates`` is appended by hand — the block scopes
    to just that one field, not the whole ``RetrievalCfg`` (which also holds ``k`` and
    ``max_rounds``; ``k`` stays a product decision, out of the optimizer's scope, and printing
    it alongside a tuned value would be misleading).
    """
    block = _ConfigBlock(
        chunking=chunking,
        embedding=cfg.embedding,
        reranker=replace(cfg.reranker, enabled=rerank),
    )
    yaml_text = draccus.dump(block)
    return (
        f"{yaml_text}retrieval:\n"
        f"  candidates: {candidates}  # only candidates is tuned here — k stays product-chosen\n"
    )


def cmd_confirm(
    args: argparse.Namespace,
    *,
    searcher: Searcher | None = None,
    embedder: Any = None,
    reranker: Reranker | None = None,
) -> None:
    """Score the winning config once on the held-out test split, then emit a config block.

    "The winner" is read from ``--max-tokens``/``--candidates``/``--rerank`` (each falling back
    to the loaded config's own value) — the user reads ``screen``/``sweep`` output and passes
    the values themselves; there is no auto-selection formula (the retrieval screen's own live
    report needed a human trade-off call between success and MRR).

    ``searcher``/``embedder``/``reranker`` are injectable for offline tests; production leaves
    them ``None`` and builds real ones.
    """
    cfg, pool, fingerprint, items = _load_test_set(args)

    max_tokens = cfg.chunking.max_tokens if args.max_tokens is None else args.max_tokens
    candidates = cfg.retrieval.candidates if args.candidates is None else args.candidates
    rerank = cfg.reranker.enabled if args.rerank is None else args.rerank
    if candidates < cfg.retrieval.max_k:
        raise SystemExit(
            f"--candidates={candidates} is below retrieval.max_k={cfg.retrieval.max_k} — "
            f"scoring would silently return fewer than max_k ranked results per query."
        )

    evals_dir = Path(cfg.root) / "evals"
    marker_path = evals_dir / f"{fingerprint}.confirm.json"
    if marker_path.exists():
        prior = json.loads(marker_path.read_text())
        print(
            f"⚠ test split for this pool was already confirmed at {prior['timestamp']} with "
            f"max_tokens={prior['max_tokens']} candidates={prior['candidates']} "
            f"rerank={prior['rerank']} — re-running weakens the held-out guarantee."
        )

    # Distinct from marker_path above: this is a transient resume checkpoint, deleted on a
    # fully successful run, never the "touched once" record — --fresh only clears this one.
    checkpoint_path = _checkpoint_path(cfg, fingerprint, "confirm")
    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)

    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  test={len(items)} questions")
    print("Confirming on the HELD-OUT TEST SPLIT (touched once per pool).")
    print(f"Config: max_tokens={max_tokens} candidates={candidates} rerank={rerank}")

    # Built once regardless of `rerank`'s value ("the on-arm always needs it", mirroring
    # optimizer.py's screen arms) and regardless of `cfg.reranker.enabled` — that flag is the
    # loaded config's own default, which `--rerank` may override, so gating construction on it
    # would silently fall back to Searcher.reranker's lazy hf cross-encoder for a `type: llm`
    # config whenever the override turns reranking on. build_reranker_for_cfg handles the
    # LLMRerankerCfg case correctly; the lazy default does not.
    active_reranker = reranker if reranker is not None else build_reranker_for_cfg(cfg)

    if max_tokens == cfg.chunking.max_tokens:
        # No chunking change -> the existing on-disk collection already holds exactly this
        # config's chunks (same markdown, same ChunkingCfg -> same content-hashed chunk ids),
        # so read it directly rather than paying a full re-embed to reproduce it from scratch.
        active_searcher = (
            searcher
            if searcher is not None
            else Searcher(
                db_dir=cfg.paths.rag_db,
                collection=cfg.collection,
                embedder=embedder,
                embedder_model=cfg.embedding.model,
                query_prefix=cfg.embedding.query_prefix
                if isinstance(cfg.embedding, HFEmbeddingCfg)
                else "",
                reranker=active_reranker,
            )
        )
        report = run(
            cfg,
            items,
            searcher=active_searcher,
            candidates=candidates,
            k=cfg.retrieval.max_k,
            rerank=rerank,
            desc="scoring test split",
            checkpoint_path=checkpoint_path,
        )
    else:
        # Chunking changed -> a fresh index is required; isolate it (Guard 1) so the prod
        # collection at paths.rag_db is never touched.
        chunking = replace(cfg.chunking, max_tokens=max_tokens)
        with tempfile.TemporaryDirectory(prefix="paperlens-eval-") as tmp:
            active_searcher = (
                searcher
                if searcher is not None
                else build_isolated_searcher(
                    pool,
                    chunking,
                    cfg.embedding,
                    db_dir=tmp,
                    embedder=embedder,
                    reranker=active_reranker,
                    desc=f"embed max_tokens={max_tokens}",
                )
            )
            report = run(
                cfg,
                items,
                searcher=active_searcher,
                candidates=candidates,
                k=cfg.retrieval.max_k,
                rerank=rerank,
                desc="scoring test split",
                checkpoint_path=checkpoint_path,
            )

    print(format_report(report))
    print(f"  · {_ESTIMAND_CAVEAT}")

    winning_chunking = replace(cfg.chunking, max_tokens=max_tokens)
    print()
    print("--- config.yaml block ---")
    print(format_config_block(cfg, chunking=winning_chunking, candidates=candidates, rerank=rerank))

    evals_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "max_tokens": max_tokens,
                "candidates": candidates,
                "rerank": rerank,
                "success_at_candidates": report.success_at_candidates,
                "mrr_at_k": report.mrr_at_k,
                "n_clusters": report.n_clusters,
            },
            indent=2,
        )
    )


_FRESH_HELP = (
    "discard any resumable checkpoint from an interrupted prior run and start over "
    "(a no-op if there isn't one)"
)


def main() -> None:
    p = argparse.ArgumentParser(prog="paperlens-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="build the span-anchored eval set from the loaded pool")
    g.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    g.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: cap the pool to the first N papers (own fingerprint, no clobber)",
    )
    g.add_argument(
        "--genfilter",
        action="store_true",
        help=(
            "closed-book leakage filter — discard questions the model can already "
            "answer without the section (rag.llm, no judge call). Roughly doubles gen's LLM "
            "call volume. Off by default; add only if a screen/sweep report's MDD is too "
            "high to be useful, and hand-check <fingerprint>.genfilter.jsonl before trusting "
            "the default threshold on a new pool."
        ),
    )
    g.add_argument(
        "--genfilter-threshold",
        type=float,
        default=None,
        help="override GenFilterConfig.match_threshold (token-F1 cutoff; default 0.5, unvalidated)",
    )
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("run", help="score one config on the dev set (recall@candidates + nDCG@k)")
    r.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    r.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    r.add_argument("--fresh", action="store_true", help=_FRESH_HELP)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser(
        "screen",
        help="OFAT screen: retrieval knobs (reranker/candidates, no re-index) or chunking knobs",
    )
    s.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    s.add_argument(
        "--tier",
        default="retrieval",
        choices=["retrieval", "chunking", "elbow"],
        help="retrieval: reranker/candidates, no re-index; "
        "chunking: chunking knobs, isolated re-index per cell; "
        "elbow: elbow_mad_multiplier/elbow_prominence, no re-retrieve (min_k/max_k stay "
        "out — a product decision, same as retrieval.k always was)",
    )
    s.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    s.add_argument(
        "--candidates",
        default=None,
        help="--tier retrieval: comma-separated candidates grid (default 10,20,30,50)",
    )
    s.add_argument(
        "--hybrid",
        action="store_true",
        help="--tier retrieval: add a 'hybrid=on' arm (BM25 fused via RRF) — independent of "
        "config.yaml's sparse.enabled, so this is how you measure hybrid before turning it on",
    )
    s.add_argument(
        "--multi-query",
        action="store_true",
        help="--tier retrieval: add a 'multi_query=on' arm (llm.chat paraphrases the query, "
        "every variant's pool RRF-fused) — independent of config.yaml's multi_query.enabled. "
        "Not free like --hybrid: widens the cache-build pass with one LLM call plus "
        "n_paraphrases extra dense queries per item",
    )
    s.add_argument(
        "--max-tokens",
        default=None,
        help="--tier chunking: comma-separated max_tokens grid overriding the default screen",
    )
    s.add_argument(
        "--mad-multiplier",
        default=None,
        help="--tier elbow: comma-separated mad_multiplier grid (default 1.5,2,3,4.5)",
    )
    s.add_argument(
        "--prominence",
        default=None,
        help="--tier elbow: comma-separated prominence grid (default 0.05,0.1,0.15,0.25)",
    )
    s.add_argument("--fresh", action="store_true", help=_FRESH_HELP)
    s.set_defaults(func=cmd_screen)

    w = sub.add_parser(
        "sweep",
        help="chunking × retrieval grid: re-index per max_tokens × cached candidates/rerank slices",
    )
    w.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    w.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    w.add_argument(
        "--max-tokens",
        default=None,
        help="comma-separated max_tokens grid, the re-index axis (default 256,512,1024)",
    )
    w.add_argument(
        "--candidates",
        default=None,
        help="comma-separated candidates grid, derived from cache (default 10,20,30,50)",
    )
    w.add_argument("--fresh", action="store_true", help=_FRESH_HELP)
    w.set_defaults(func=cmd_sweep)

    c = sub.add_parser(
        "confirm",
        help="score the winning config once on the held-out test split, then emit a config block",
    )
    c.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    c.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    c.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="winning chunking.max_tokens (default: the config's own value); confirm only "
        "covers the sweep grid's axes (max_tokens, candidates, rerank) — a screen --tier "
        "chunking winner on overlap_tokens/min_tokens/noise_ratio isn't confirmable here",
    )
    c.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="winning retrieval.candidates (default: the config's own value)",
    )
    c.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="winning reranker.enabled (default: the config's own value)",
    )
    c.add_argument("--fresh", action="store_true", help=_FRESH_HELP)
    c.set_defaults(func=cmd_confirm)

    pp = sub.add_parser(
        "per-paper",
        help="standalone: measure Searcher.search(per_paper=True) vs False on a "
        "randomly-sampled multi-paper scope (see per-paper-eval-spec.md; not part of "
        "gen→run→screen→sweep→confirm, no config.yaml output)",
    )
    pp_sub = pp.add_subparsers(dest="per_paper_cmd", required=True)

    pps = pp_sub.add_parser(
        "sweep",
        help="exploratory: per_paper on (production + budget-matched) vs off, paired "
        "across a candidates grid",
    )
    pps.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    pps.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    pps.add_argument(
        "--per-paper-n",
        type=int,
        default=DEFAULT_PER_PAPER_N,
        help=f"papers per scope, gold paper plus n-1 random others (default {DEFAULT_PER_PAPER_N})",
    )
    pps.add_argument(
        "--candidates",
        default=None,
        help="comma-separated candidates grid (default 10,20,30,50)",
    )
    pps.add_argument(
        "--seed",
        type=int,
        default=0,
        help="scope-assignment seed (which other papers join each scope)",
    )
    pps.add_argument("--fresh", action="store_true", help=_FRESH_HELP)
    pps.set_defaults(func=cmd_per_paper_sweep)

    ppc = pp_sub.add_parser(
        "confirm",
        help="one candidates value, one variant, a fresh seed — the only per-paper mode "
        "allowed to recommend",
    )
    ppc.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    ppc.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    ppc.add_argument(
        "--per-paper-n",
        type=int,
        default=DEFAULT_PER_PAPER_N,
        help=f"papers per scope — must match the sweep run being confirmed "
        f"(default {DEFAULT_PER_PAPER_N})",
    )
    ppc.add_argument(
        "--candidates", type=int, required=True, help="the candidates value to confirm"
    )
    ppc.add_argument(
        "--variant",
        required=True,
        choices=["production", "budget-matched"],
        help="which claim to confirm: production's max(k, ...) floor, or the budget-matched "
        "no-floor allocation",
    )
    ppc.add_argument(
        "--seed",
        type=int,
        required=True,
        help="a FRESH scope-assignment seed, never the sweep's own",
    )
    ppc.set_defaults(func=cmd_per_paper_confirm)

    cp = sub.add_parser(
        "comparative",
        help="standalone: genuinely cross-paper synthesis questions (gold spans 2-6 "
        "papers) — measures Searcher.search(per_paper=True) the way per-paper can't "
        "(see comparative-eval-spec.md; not part of gen→run→screen→sweep→confirm, no "
        "config.yaml output)",
    )
    cp_sub = cp.add_subparsers(dest="comparative_cmd", required=True)

    cpg = cp_sub.add_parser(
        "gen",
        help="trial loop: spot cross-paper matches from outlines, write a question "
        "needing every match, until target_p items or max_trials trials",
    )
    cpg.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    cpg.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: cap the pool to the first N papers (own fingerprint, no clobber)",
    )
    cpg.add_argument(
        "--target-p",
        type=int,
        default=DEFAULT_TARGET_P,
        help=f"target comparative item pool size, dev+test combined "
        f"(default {DEFAULT_TARGET_P}, placeholder — see comparative-eval-spec.md)",
    )
    cpg.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help=f"hard cap on trials regardless of hit rate (default {DEFAULT_MAX_TRIALS})",
    )
    cpg.add_argument(
        "--n-papers-min",
        type=int,
        default=DEFAULT_N_PAPERS_MIN,
        help=f"fewest papers offered per trial (default {DEFAULT_N_PAPERS_MIN})",
    )
    cpg.add_argument(
        "--n-papers-max",
        type=int,
        default=DEFAULT_N_PAPERS_MAX,
        help=f"most papers offered per trial (default {DEFAULT_N_PAPERS_MAX})",
    )
    cpg.add_argument(
        "--seed",
        type=int,
        default=0,
        help="trial paper/size draws + the dev/test split coin flip",
    )
    cpg.set_defaults(func=cmd_comparative_gen)

    cps = cp_sub.add_parser(
        "sweep",
        help="exploratory: per_paper on (production + budget-matched) vs off, paired "
        "across a candidates grid, scored on the comparative dev split",
    )
    cps.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    cps.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    cps.add_argument(
        "--candidates",
        default=None,
        help="comma-separated candidates grid (default 10,20,30,50)",
    )
    cps.add_argument("--fresh", action="store_true", help=_FRESH_HELP)
    cps.set_defaults(func=cmd_comparative_sweep)

    cpc = cp_sub.add_parser(
        "confirm",
        help="one candidates value, one variant, scored on the held-out comparative "
        "test split — the only comparative mode allowed to recommend",
    )
    cpc.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    cpc.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    cpc.add_argument(
        "--candidates", type=int, required=True, help="the candidates value to confirm"
    )
    cpc.add_argument(
        "--variant",
        required=True,
        choices=["production", "budget-matched"],
        help="which claim to confirm: production's max(k, ...) floor, or the budget-matched "
        "no-floor allocation",
    )
    cpc.set_defaults(func=cmd_comparative_confirm)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
