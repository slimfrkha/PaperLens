"""File-backed chat session store + session naming."""

from __future__ import annotations

import pytest

from server.chats import ChatStore, generate_name


def test_create_then_get_roundtrip(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    assert store.get(chat["id"])["id"] == chat["id"]
    assert store.get("missing") is None


def test_append_turn_keeps_parallel_arrays_aligned(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    citations = [{"ref": "r1"}]
    trace = [{"type": "action", "query": "q"}]
    saved = store.append_turn(chat["id"], "hi", "hello [r1]", citations, trace)

    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]
    # citations/traces run parallel to messages: null for the user turn.
    assert saved["citations"] == [None, citations]
    assert saved["traces"] == [None, trace]


def test_append_turn_pads_legacy_sessions_without_traces(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    # Simulate a pre-traces session: messages present, traces missing.
    chat["messages"] = [{"role": "user", "content": "old"}]
    chat["citations"] = [None]
    del chat["traces"]
    store._write(chat)

    saved = store.append_turn(chat["id"], "q", "a", [], [])
    assert len(saved["traces"]) == len(saved["messages"])


def test_delete_and_list_all_sorted_by_updated(tmp_path):
    store = ChatStore(str(tmp_path))
    a = store.append_turn(store.create()["id"], "q", "a", [], [], name="Alpha")
    b = store.append_turn(store.create()["id"], "q", "a", [], [], name="Beta")

    listed = store.list_all()
    assert {row["name"] for row in listed} == {"Alpha", "Beta"}

    store.delete(a["id"])
    assert store.get(a["id"]) is None
    assert [row["id"] for row in store.list_all()] == [b["id"]]


def test_write_failure_leaves_prior_file_intact(tmp_path, monkeypatch):
    import server.chats as chats_mod

    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [], name="Alpha")

    # A serialization failure mid-write must not corrupt the already-saved chat:
    # the temp file takes the hit, the real file is only swapped in on success.
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(chats_mod.json, "dumps", boom)
    with pytest.raises(OSError):
        store.append_turn(chat["id"], "q2", "a2", [], [])

    monkeypatch.undo()
    reloaded = store.get(chat["id"])
    assert [m["content"] for m in reloaded["messages"]] == ["q", "a"]  # unchanged
    assert list(tmp_path.glob("*.tmp")) == []  # no orphaned temp left behind


def test_generate_name_falls_back_when_llm_errors(monkeypatch):
    from rag.config import AnthropicSpec

    # No api_base + no key -> build_llm(...).complete raises -> graceful fallback.
    spec = AnthropicSpec(api_key_env="DEFINITELY_UNSET_KEY_XYZ")
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_XYZ", raising=False)
    assert generate_name("How does MLA work?", spec) == "New chat"
