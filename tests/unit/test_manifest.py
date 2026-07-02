"""Paper metadata store (papers.json)."""

from __future__ import annotations

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


def test_paper_ids_for_tags_is_or_semantics(tmp_path):
    m = Manifest(str(tmp_path))
    m.upsert(_rec("a", ["moe"]))
    m.upsert(_rec("b", ["rl"]))
    m.upsert(_rec("c", ["quant"]))
    got = set(m.paper_ids_for_tags(["moe", "rl"]))
    assert got == {"a", "b"}
    assert m.paper_ids_for_tags(["missing"]) == []
