"""File-backed paper annotation store."""

from __future__ import annotations

import pytest

from server.annotations import AnnotationStore


def test_create_then_list_roundtrip(tmp_path):
    store = AnnotationStore(str(tmp_path))
    a = store.create("paper1", "some passage", "2.1 Attention", "2-1-attention", "a note")

    assert store.list_all("paper1") == [a]
    assert store.list_all("missing") == []


def test_create_appends_without_clobbering_prior_annotations(tmp_path):
    store = AnnotationStore(str(tmp_path))
    first = store.create("paper1", "passage one", "Intro", "intro", "")
    second = store.create("paper1", "passage two", "Intro", "intro", "note")

    assert [a["id"] for a in store.list_all("paper1")] == [first["id"], second["id"]]


def test_update_changes_note_and_bumps_updated_at(tmp_path):
    store = AnnotationStore(str(tmp_path))
    a = store.create("paper1", "passage", "Intro", "intro", "old note")

    updated = store.update("paper1", a["id"], "new note")
    assert updated["note"] == "new note"
    assert updated["updated_at"] >= a["updated_at"]
    assert store.list_all("paper1")[0]["note"] == "new note"


def test_update_returns_none_for_missing_annotation(tmp_path):
    store = AnnotationStore(str(tmp_path))
    store.create("paper1", "passage", "Intro", "intro", "")
    assert store.update("paper1", "missing-id", "note") is None


def test_update_returns_none_for_missing_paper(tmp_path):
    store = AnnotationStore(str(tmp_path))
    assert store.update("missing-paper", "some-id", "note") is None


def test_delete_removes_only_the_target_annotation(tmp_path):
    store = AnnotationStore(str(tmp_path))
    keep = store.create("paper1", "passage one", "Intro", "intro", "")
    drop = store.create("paper1", "passage two", "Intro", "intro", "")

    assert store.delete("paper1", drop["id"]) is True
    assert [a["id"] for a in store.list_all("paper1")] == [keep["id"]]


def test_delete_returns_false_for_missing_annotation(tmp_path):
    store = AnnotationStore(str(tmp_path))
    store.create("paper1", "passage", "Intro", "intro", "")
    assert store.delete("paper1", "missing-id") is False


def test_remove_paper_drops_all_its_annotations(tmp_path):
    store = AnnotationStore(str(tmp_path))
    store.create("paper1", "passage one", "Intro", "intro", "")
    store.create("paper1", "passage two", "Intro", "intro", "")
    store.create("paper2", "unrelated", "Intro", "intro", "")

    store.remove_paper("paper1")

    assert store.list_all("paper1") == []
    assert len(store.list_all("paper2")) == 1  # untouched


def test_remove_paper_is_a_noop_for_a_paper_with_no_annotations(tmp_path):
    store = AnnotationStore(str(tmp_path))
    store.remove_paper("never-had-any")  # must not raise


def test_write_failure_leaves_prior_file_intact(tmp_path, monkeypatch):
    import server.annotations as annotations_mod

    store = AnnotationStore(str(tmp_path))
    a = store.create("paper1", "passage", "Intro", "intro", "note")

    # A serialization failure mid-write must not corrupt the already-saved file: the
    # temp file takes the hit, the real file is only swapped in on success.
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(annotations_mod.json, "dumps", boom)
    with pytest.raises(OSError):
        store.create("paper1", "another passage", "Intro", "intro", "")

    monkeypatch.undo()
    assert store.list_all("paper1") == [a]  # unchanged
    assert list(tmp_path.glob("*.tmp")) == []  # no orphaned temp left behind
