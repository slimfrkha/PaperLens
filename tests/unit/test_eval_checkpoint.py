"""Checkpoint envelope: header-gated resume, append-and-flush durability, clean-up on success.

The interruption story this module exists for — a kill mid-unit never leaves a partial
line, a header that no longer matches invalidates instead of silently misreading — is
what these tests pin down directly; the per-command wiring is exercised end-to-end
elsewhere (test_eval_cli.py, test_eval_optimizer.py, test_eval_harness.py).
"""

from __future__ import annotations

import json

from eval.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointWriter, resume_units


def test_resume_units_no_file_returns_empty(tmp_path):
    assert resume_units(tmp_path / "missing.ckpt.jsonl", header={"k": 1}) == {}


def test_write_then_resume_round_trip(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    header = {"candidates": 20, "k": 5}
    w = CheckpointWriter(path, header)
    w.append("0", {"score": 0.5})
    w.append("1", {"score": 0.9})

    done = resume_units(path, header)
    assert done == {"0": {"score": 0.5}, "1": {"score": 0.9}}
    # The file is still there — resume_units only reads, it never deletes on a match.
    assert path.exists()


def test_second_writer_appends_rather_than_truncates(tmp_path):
    """Constructing a 2nd CheckpointWriter over an existing, still-valid checkpoint must
    append to it, not silently truncate an already-completed unit's line — the writer
    decides append-vs-fresh from ``path.exists()`` itself, not a caller-supplied flag."""
    path = tmp_path / "x.ckpt.jsonl"
    header = {"candidates": 20}
    w1 = CheckpointWriter(path, header)
    w1.append("0", {"v": "a"})

    done = resume_units(path, header)
    assert "0" in done
    w2 = CheckpointWriter(path, header)  # path already exists -> must append, not truncate
    w2.append("1", {"v": "b"})

    done_again = resume_units(path, header)
    assert done_again == {"0": {"v": "a"}, "1": {"v": "b"}}


def test_finish_deletes_the_checkpoint(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    w = CheckpointWriter(path, {"k": 1})
    w.append("0", {})
    assert path.exists()
    w.finish()
    assert not path.exists()


def test_close_leaves_the_checkpoint_on_disk(tmp_path):
    """``close()`` (unlike ``finish()``) doesn't delete — for a caller (sweep) that owns
    several checkpoint files across one command and defers cleanup until all are done."""
    path = tmp_path / "x.ckpt.jsonl"
    header = {"k": 1}
    w = CheckpointWriter(path, header)
    w.append("0", {"v": 1})
    w.close()
    assert path.exists()
    assert resume_units(path, header) == {"0": {"v": 1}}


def test_header_mismatch_discards_and_starts_fresh(tmp_path, capsys):
    path = tmp_path / "x.ckpt.jsonl"
    w = CheckpointWriter(path, {"candidates": 20})
    w.append("0", {"v": 1})

    done = resume_units(path, {"candidates": 30})
    assert done == {}
    assert not path.exists()  # stale checkpoint removed so a later run doesn't trip on it
    assert "candidates: 20 -> 30" in capsys.readouterr().out


def test_schema_version_bump_invalidates_old_checkpoints(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    stale_header = {"candidates": 20, "schema_version": CHECKPOINT_SCHEMA_VERSION - 1}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": stale_header}) + "\n")
        f.write(json.dumps({"id": "0", "record": {"v": 1}}) + "\n")

    done = resume_units(path, {"candidates": 20})
    assert done == {}
    assert not path.exists()


def test_corrupt_trailing_line_is_dropped_not_fatal(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    header = {"candidates": 20}
    w = CheckpointWriter(path, header)
    w.append("0", {"v": 1})
    w.append("1", {"v": 2})
    # Simulate a kill mid-write: append a truncated, non-JSON trailing line by hand.
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"id": "2", "record": {"v": 3')  # no closing braces, no trailing newline

    done = resume_units(path, header)
    assert done == {"0": {"v": 1}, "1": {"v": 2}}  # unit "2" silently dropped, not raised


def test_valid_json_non_object_trailing_line_is_dropped_not_fatal(tmp_path):
    """A truncated write can land on a byte boundary that happens to parse as valid JSON
    while still being nonsense (a bare list/number instead of the expected object) —
    ``json.loads(line)["id"]`` on that raises TypeError, not KeyError; must still be
    treated as corrupt, not crash the whole resume."""
    path = tmp_path / "x.ckpt.jsonl"
    header = {"candidates": 20}
    w = CheckpointWriter(path, header)
    w.append("0", {"v": 1})
    with open(path, "a", encoding="utf-8") as f:
        f.write("[1, 2, 3]\n")

    done = resume_units(path, header)
    assert done == {"0": {"v": 1}}


def test_valid_json_non_object_header_is_treated_as_no_checkpoint(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    path.write_text("[1, 2, 3]\n")
    assert resume_units(path, {"k": 1}) == {}
    assert not path.exists()


def test_empty_file_is_treated_as_no_checkpoint(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    path.write_text("")
    assert resume_units(path, {"k": 1}) == {}
    assert not path.exists()


def test_corrupt_header_is_treated_as_no_checkpoint(tmp_path):
    path = tmp_path / "x.ckpt.jsonl"
    path.write_text("not json at all\n")
    assert resume_units(path, {"k": 1}) == {}
    assert not path.exists()
