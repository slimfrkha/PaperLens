"""Faithfulness checker calibration — precision/recall/F1 and a threshold sweep
against a hand-labeled golden set.

Not part of the pytest gate: it loads the real checker's model, and this repo's
tests stay offline (see CLAUDE.md). Run manually when the golden set grows, the
model or revision changes, or before flipping ``faithfulness.enabled: true`` on
a real config.

Golden set format — one JSON object per line in the fixture file:

    {"premise": "<passage sentence>", "hypothesis": "<citing sentence>", "label": "entailment"}

``label`` is your judgment of whether ``hypothesis`` is supported by ``premise``
— one of "entailment" / "neutral" / "contradiction", the same three labels
``rag.faithfulness`` produces at answer time.

Usage:

    uv run python scripts/calibrate_faithfulness.py
    uv run python scripts/calibrate_faithfulness.py --fixture path/to/pairs.jsonl \\
        --config configs/my-setup.yaml
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from rag.config import load_config
from rag.faithfulness import HFFaithfulnessChecker, _to_verdict, build_faithfulness_checker

_LABELS = ("entailment", "neutral", "contradiction")
_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "faithfulness_pairs.jsonl"
)
# Coarse enough to be readable, fine enough to place the checked-in defaults
# (0.05 / 0.3) accurately; sweeping is pure Python over already-computed scores,
# not a model call, so a finer grid costs nothing but console width.
_THRESHOLD_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No golden set at {path}.\n"
            "Create it as one JSON object per line:\n"
            '  {"premise": "<passage sentence>", "hypothesis": "<citing sentence>", '
            '"label": "entailment"}\n'
            f"label is one of {', '.join(_LABELS)}."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty — add at least one labeled pair.")
    bad = sorted({r["label"] for r in rows if r.get("label") not in _LABELS})
    if bad:
        raise SystemExit(f"Unknown label(s) {bad} — must be one of {_LABELS}.")
    return rows


def confusion_matrix(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {g: dict.fromkeys(_LABELS, 0) for g in _LABELS}
    for g, p in zip(gold, pred, strict=True):
        matrix[g][p] += 1
    return matrix


def per_label_prf(matrix: dict[str, dict[str, int]]) -> dict[str, tuple[float, float, float]]:
    out = {}
    for label in _LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in _LABELS if p != label)
        fp = sum(matrix[g][label] for g in _LABELS if g != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = (precision, recall, f1)
    return out


def macro_f1(matrix: dict[str, dict[str, int]]) -> float:
    prf = per_label_prf(matrix)
    return sum(f1 for _, _, f1 in prf.values()) / len(_LABELS)


def format_report(title: str, gold: list[str], pred: list[str]) -> str:
    matrix = confusion_matrix(gold, pred)
    prf = per_label_prf(matrix)
    lines = [title, "  confusion matrix (rows=gold, cols=predicted):"]
    lines.append(" " * 13 + "  ".join(f"{lbl:>13}" for lbl in _LABELS))
    for g in _LABELS:
        row = "  ".join(f"{matrix[g][p]:>13d}" for p in _LABELS)
        lines.append(f"  {g:>9}  {row}")
    lines.append("  precision / recall / F1:")
    for label in _LABELS:
        p, r, f1 = prf[label]
        lines.append(f"    {label:<13} P={p:.2f}  R={r:.2f}  F1={f1:.2f}")
    lines.append(f"  macro F1: {macro_f1(matrix):.3f}")
    return "\n".join(lines)


def sweep(scores: list[float], gold: list[str]) -> list[tuple[float, float, float]]:
    """Every valid (contradiction_max, entailment_min) combo on the grid, ranked
    by macro F1 against the already-computed scores — no extra model calls."""
    results = []
    for cmax, emin in product(_THRESHOLD_GRID, repeat=2):
        if cmax >= emin:  # same invariant as HFFaithfulnessCfg.__post_init__
            continue
        pred = [_to_verdict(s, cmax, emin).label for s in scores]
        results.append((cmax, emin, macro_f1(confusion_matrix(gold, pred))))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", default=str(_DEFAULT_FIXTURE), help="path to golden JSONL set")
    parser.add_argument("--config", default=None, help="path to config.yaml (else discovered)")
    parser.add_argument("--top", type=int, default=5, help="how many top threshold combos to print")
    args = parser.parse_args()

    rows = load_pairs(Path(args.fixture))
    gold = [r["label"] for r in rows]
    pairs = [(r["premise"], r["hypothesis"]) for r in rows]

    cfg = load_config(args.config)
    checker = build_faithfulness_checker(cfg.faithfulness)
    if not isinstance(checker, HFFaithfulnessChecker):
        raise SystemExit(
            f"Calibration only supports the 'hf' faithfulness backend, got "
            f"{type(checker).__name__}."
        )

    scores = [float(s) for s in checker.model.predict(pairs)]
    current_pred = [
        _to_verdict(s, checker.contradiction_max, checker.entailment_min).label for s in scores
    ]

    print(f"golden set: {len(rows)} pairs from {args.fixture}")
    print(f"model: {checker.model_name} (revision={checker.revision})\n")
    print(
        format_report(
            f"Current config (contradiction_max={checker.contradiction_max}, "
            f"entailment_min={checker.entailment_min}):",
            gold,
            current_pred,
        )
    )

    print(f"\nTop {args.top} threshold combos by macro F1:")
    for cmax, emin, f1 in sweep(scores, gold)[: args.top]:
        print(f"  contradiction_max={cmax:.2f}  entailment_min={emin:.2f}  macro F1={f1:.3f}")


if __name__ == "__main__":
    main()
