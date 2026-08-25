"""Paper metadata store (papers.json)."""

from __future__ import annotations

import threading

from rag.manifest import Manifest


def _rec(paper_id: str, tags: list[str], n_chunks: int = 1) -> dict:
    return {"paper_id": paper_id, "title": paper_id.title(), "tags": tags, "n_chunks": n_chunks}


def test_upsert_get_and_is_ingested(tmp_path):
    m = Manifest(str(tmp_path))
    assert not m.is_ingested("a")
    assert m.get("a") is None

    m.upsert(_rec("a", ["moe"]))
    assert m.is_ingested("a")
    assert m.get("a")["title"] == "A"
    assert [p["paper_id"] for p in m.papers()] == ["a"]


def test_upsert_overwrites_same_id(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["old"]))
    m.upsert(_rec("a", ["new"]))
    assert len(m.papers()) == 1
    assert m.get("a")["tags"] == ["new"]


def test_all_tags_counts_and_ordering(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["moe", "rl"]))
    m.upsert(_rec("b", ["moe"]))
    tags = m.all_tags()
    # Most common first; ties broken alphabetically.
    assert tags == [{"tag": "moe", "count": 2}, {"tag": "rl", "count": 1}]


def test_discriminating_tags_drops_universal(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["llm", "moe"]))
    m.upsert(_rec("b", ["llm", "rl"]))
    # "llm" is on every paper -> useless as a filter -> dropped.
    assert m.discriminating_tags() == [{"tag": "moe", "count": 1}, {"tag": "rl", "count": 1}]


def test_discriminating_tags_keeps_all_for_single_paper(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["llm", "moe"]))
    # Every tag is trivially universal with one paper; don't hide them all.
    assert m.discriminating_tags() == m.all_tags()


def test_remove_deletes_entry_and_leaves_others(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["moe"]))
    m.upsert(_rec("b", ["rl"]))

    assert m.remove("a") is True
    assert m.get("a") is None
    assert not m.is_ingested("a")
    assert [p["paper_id"] for p in m.papers()] == ["b"]


def test_remove_unknown_id_returns_false(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["moe"]))
    assert m.remove("missing") is False
    assert [p["paper_id"] for p in m.papers()] == ["a"]


def test_paper_ids_for_tags_is_or_semantics(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["moe"]))
    m.upsert(_rec("b", ["rl"]))
    m.upsert(_rec("c", ["quant"]))
    got = set(m.paper_ids_for_tags(["moe", "rl"]))
    assert got == {"a", "b"}
    assert m.paper_ids_for_tags(["missing"]) == []


def test_upsert_and_remove_leave_no_stray_tmp_file(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["moe"]))
    m.remove("a")
    # filelock unlinks its own lock file on release, so only papers.json should remain.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["papers.json"]


def test_concurrent_upserts_from_separate_manifest_instances(tmp_path):
    # Each thread gets its own Manifest instance (own object, same papers.json/lock
    # path) to exercise the cross-process-style FileLock guard, not a shared instance.
    n = 20

    def upsert_one(i: int) -> None:
        Manifest(str(tmp_path)).upsert(_rec(f"p{i}", ["moe"]))

    threads = [threading.Thread(target=upsert_one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    m = Manifest(str(tmp_path))
    assert {p["paper_id"] for p in m.papers()} == {f"p{i}" for i in range(n)}
