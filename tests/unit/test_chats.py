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


def test_set_feedback_on_assistant_turn(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])

    saved = store.set_feedback(chat["id"], 1, "up", "great citation")
    assert saved["feedback"] == [
        None,
        {"vote": "up", "note": "great citation", "updated_at": saved["feedback"][1]["updated_at"]},
    ]


def test_set_feedback_rejects_user_message_index(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])
    with pytest.raises(ValueError):
        store.set_feedback(chat["id"], 0, "up", None)


def test_set_feedback_rejects_out_of_range_index(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])
    with pytest.raises(ValueError):
        store.set_feedback(chat["id"], 5, "up", None)


def test_set_feedback_rejects_note_without_vote(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])
    with pytest.raises(ValueError):
        store.set_feedback(chat["id"], 1, None, "a note with no vote")


def test_set_feedback_rejects_invalid_vote_value(tmp_path):
    # Pydantic's Literal["up", "down"] gates the HTTP route, but ChatStore is the
    # persistence layer and shouldn't trust an arbitrary direct caller either.
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])
    with pytest.raises(ValueError):
        store.set_feedback(chat["id"], 1, "sideways", None)


def test_set_feedback_clears_vote(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])
    store.set_feedback(chat["id"], 1, "down", "wrong section")

    cleared = store.set_feedback(chat["id"], 1, None, None)
    assert cleared["feedback"][1] is None


def test_set_feedback_returns_none_for_missing_chat(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.set_feedback("missing", 0, "up", None) is None


def test_set_feedback_pads_legacy_sessions_without_feedback(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], "q", "a", [], [])
    # Simulate a pre-feedback session: feedback key missing entirely.
    del chat["feedback"]
    store._write(chat)

    saved = store.set_feedback(chat["id"], 1, "up", None)
    assert len(saved["feedback"]) == len(saved["messages"])
    assert saved["feedback"][1] == {
        "vote": "up",
        "note": None,
        "updated_at": saved["feedback"][1]["updated_at"],
    }


def test_generate_name_falls_back_when_llm_errors(monkeypatch):
    from rag.config import AnthropicSpec

    # No api_base + no key -> build_llm(...).complete raises -> graceful fallback.
    spec = AnthropicSpec(api_key_env="DEFINITELY_UNSET_KEY_XYZ")
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_XYZ", raising=False)
    assert generate_name("How does MLA work?", spec) == "New chat"
