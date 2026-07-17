"""paperlens-eval — per-pool config-optimization harness.

Phase 1: ``gen`` builds the span-anchored QA eval set from the loaded pool and writes
it, keyed on the corpus fingerprint, under ``evals/`` at the project root::

    paperlens-eval gen                 # discover config.yaml, build the set
    paperlens-eval gen --config path/to/config.yaml

Later phases add ``screen`` / ``sweep`` / ``confirm``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.config import load_config
from rag.llm import build_llm

from .fingerprint import corpus_fingerprint, load_pool
from .harness import format_report, run
from .optimizer import format_screen_report, screen_tier_a
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
    test_ids = held_out_paper_ids(pool, gen)
    out_dir = Path(cfg.root) / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / f"{fingerprint}.dev.jsonl"
    test_path = out_dir / f"{fingerprint}.test.jsonl"

    smoke = "  [smoke: --limit]" if args.limit else ""
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}{smoke}")
    print(f"Generating with {cfg.llm.tagging.model} (one question per section)...")

    llm = build_llm(cfg.llm.tagging)
    n_dev = n_test = 0
    # Stream to disk as questions are produced: a crash keeps partial progress, and a
    # per-section failure in iter_queryset is skipped rather than losing the batch.
    with (
        open(dev_path, "w", encoding="utf-8") as fdev,
        open(test_path, "w", encoding="utf-8") as ftest,
    ):
        for it in iter_queryset(pool, llm, gen):
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
            },
            indent=2,
        )
    )
    print(f"Wrote {n_dev} dev / {n_test} test questions to {out_dir}")


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


def cmd_run(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    print("Running default config on the dev split...")
    print(format_report(run(cfg, items)))


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


def cmd_screen(args: argparse.Namespace) -> None:
    cfg, pool, fingerprint, items = _load_dev_set(args)
    grid = _parse_grid(args.candidates)
    print(f"Pool: {len(pool)} papers  fingerprint={fingerprint}  dev={len(items)} questions")
    print("Screening Tier-A knobs (reranker on/off, candidates depth) on the dev split...")
    print(format_screen_report(screen_tier_a(cfg, items, candidate_grid=grid)))


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
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("run", help="score one config on the dev set (recall@candidates + nDCG@k)")
    r.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    r.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the pool to the first N papers (must match the fingerprint gen used)",
    )
    r.set_defaults(func=cmd_run)

    s = sub.add_parser(
        "screen", help="Tier-A OFAT screen of reranker on/off + candidates depth (no re-index)"
    )
    s.add_argument("--config", default=None, help="path to config.yaml (else discovery)")
    s.add_argument(
        "--tier",
        default="a",
        choices=["a"],
        help="only Tier A (no re-index) is available; Tier B lands in Phase 5",
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
        help="comma-separated candidates grid (default 10,20,30,50; default arm always added)",
    )
    s.set_defaults(func=cmd_screen)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
