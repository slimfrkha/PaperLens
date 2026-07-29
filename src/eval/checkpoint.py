"""Resume support for long-running ``paperlens-eval`` commands.

A generic envelope, not a cache: one header line recording the parameters a checkpoint
is only valid under, then one JSON line per completed unit of work, appended and
flushed immediately so a kill mid-unit never leaves a partial line — the unit in
progress at interruption time is simply absent and gets redone whole, never partially
credited. This module knows nothing about ``QueryScore``/``QueryCache``/``Sample``; each
call site serializes its own unit into a plain dict.

Deleted on success (:meth:`CheckpointWriter.finish`) — a checkpoint file's mere
existence *is* the "this command didn't finish last time" signal, so ``evals/`` never
accumulates stale ``.ckpt.jsonl`` files from completed runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

# Bump when a unit record's shape changes (e.g. a new QueryScore field) — this
# invalidates every existing checkpoint on purpose, a clean cache miss rather than an
# old record silently deserializing into something missing the new field.
CHECKPOINT_SCHEMA_VERSION = 1

# A malformed checkpoint line (missing "header"/"id"/"record", not valid JSON at all, or
# valid JSON that isn't an object — e.g. a truncated write that happens to end on a bare
# token) raises one of these; named so no except clause needs an inline tuple literal.
_ParseError = (json.JSONDecodeError, KeyError, TypeError)


def _full_header(header: dict[str, Any]) -> dict[str, Any]:
    return {**header, "schema_version": CHECKPOINT_SCHEMA_VERSION}


def _header_diff(stored: dict[str, Any], expected: dict[str, Any]) -> str:
    keys = sorted(set(stored) | set(expected))
    diffs = [
        f"{k}: {stored.get(k)!r} -> {expected.get(k)!r}"
        for k in keys
        if stored.get(k) != expected.get(k)
    ]
    return ", ".join(diffs)


def resume_units(path: Path, header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load ``{unit_id: record}`` for already-completed units in ``path``.

    Returns ``{}`` (nothing to resume) when: the file doesn't exist; its stored header
    doesn't exactly match ``header`` (plus the current :data:`CHECKPOINT_SCHEMA_VERSION`)
    — printed as *what* diverged, e.g. ``"candidates: 20 -> 30"`` — in which case the
    stale file is deleted so a later invocation doesn't trip over it; or the header line
    itself is missing/corrupt (an empty or truncated file). A corrupt/truncated *unit*
    line (a kill mid-write) is skipped rather than failing the whole resume — the
    header still validates, and every unit line before the bad one is still usable.
    """
    if not path.exists():
        return {}
    expected = _full_header(header)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        path.unlink()
        return {}
    try:
        stored = json.loads(lines[0])["header"]
    except _ParseError:
        print(f"  checkpoint at {path} has a corrupt header — discarding, starting fresh")
        path.unlink()
        return {}
    if stored != expected:
        print(f"  stale checkpoint at {path} ({_header_diff(stored, expected)}) — starting fresh")
        path.unlink()
        return {}

    done: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        try:
            obj = json.loads(line)
            done[obj["id"]] = obj["record"]
        except _ParseError:
            continue  # partial trailing line from a kill mid-write — the unit gets redone
    return done


class CheckpointWriter:
    """Appends one JSON line per completed unit; deletes the file once everything's done.

    Call :func:`resume_units` for this same ``path``/``header`` *first* — it deletes the
    file on anything that invalidates it (no match, stale params, a corrupt header), so by
    construction time ``path.exists()`` is already the true answer to "is there a valid
    checkpoint to append to," and this reads that itself rather than trusting a caller-
    supplied flag (a wrong flag here would silently open a nonexistent path in append mode,
    producing a headerless file that looks corrupt to the next resume — a footgun with no
    upside, since nothing in this codebase ever wants ``resuming=False`` while a valid
    checkpoint sits on disk: forcing a fresh start is always "delete the file, then call
    this" — see ``--fresh`` — never "call this with the file still there and ignore it").
    """

    def __init__(self, path: Path, header: dict[str, Any]) -> None:
        self.path = path
        self._f: TextIO
        if path.exists():
            self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115 — held open across append() calls
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, "w", encoding="utf-8")  # noqa: SIM115 — held open across append() calls
            self._f.write(json.dumps({"header": _full_header(header)}) + "\n")
            self._f.flush()

    def append(self, unit_id: str, record: dict[str, Any]) -> None:
        self._f.write(json.dumps({"id": unit_id, "record": record}) + "\n")
        self._f.flush()

    def close(self) -> None:
        """Close the file handle without deleting it — for a caller that owns several
        checkpoint files across one command (``sweep``, one per ``max_tokens`` cell) and
        wants to defer cleanup until *every* one of them is done, not just this one."""
        self._f.close()

    def finish(self) -> None:
        """Call only after every unit for this command has been written or loaded from
        a prior run — there is nothing left to resume, so the file is deleted."""
        self.close()
        self.path.unlink(missing_ok=True)
