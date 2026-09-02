"""File-backed chat turns + session naming."""

from __future__ import annotations

import threading

import pytest

from server.chats import ChatStore, StoredTurn, generate_name


def turn(question: str = "q", answer: str = "a", **overrides) -> StoredTurn:
    return {
        "question": question,
        "answer": answer,
        "citations": [],
        "trace": [],
        "usage": None,
        "feedback": None,
        "per_paper": False,
        "compare": False,
        "compare_results": None,
        "auto": False,
        **overrides,
    }


def test_create_then_get_roundtrip_uses_turns(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()

    assert chat["turns"] == []
    assert store.get(chat["id"])["id"] == chat["id"]
    assert store.get("missing") is None


def test_append_turn_keeps_one_complete_record(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    saved = store.append_turn(chat["id"], turn("hi", "hello [r1]", citations=[{"ref": "r1"}]))

    assert saved["turns"] == [turn("hi", "hello [r1]", citations=[{"ref": "r1"}])]
    assert "messages" not in saved
    assert "citations" not in saved


def test_append_turn_creates_a_missing_chat(tmp_path):
    store = ChatStore(str(tmp_path))

    saved = store.append_turn("new-chat", turn("question", "answer"))

    assert saved["id"] == "new-chat"
    assert saved["turns"] == [turn("question", "answer")]
    assert store.get("new-chat") == saved


def test_feedback_targets_a_turn(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], turn())
    saved = store.set_feedback(chat["id"], 0, "up", "great citation")

    assert saved["turns"][0]["feedback"]["vote"] == "up"
    with pytest.raises(ValueError, match="turn_index"):
        store.set_feedback(chat["id"], 1, "up", None)


def test_feedback_rejects_invalid_direct_calls_and_can_be_cleared(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], turn())

    with pytest.raises(ValueError, match="note requires"):
        store.set_feedback(chat["id"], 0, None, "orphan note")
    with pytest.raises(ValueError, match="vote must"):
        store.set_feedback(chat["id"], 0, "sideways", None)

    store.set_feedback(chat["id"], 0, "down", "wrong section")
    assert store.set_feedback(chat["id"], 0, None, None)["turns"][0]["feedback"] is None
    assert store.set_feedback("missing", 0, "up", None) is None


def test_truncate_before_drops_the_selected_turn_and_successors(tmp_path):
    store = ChatStore(str(tmp_path))
    chat_id = store.create()["id"]
    store.append_turn(chat_id, turn("q1", "a1", citations=[{"ref": "r1"}]))
    store.append_turn(chat_id, turn("q2", "a2", citations=[{"ref": "r2"}]))

    saved = store.truncate_before(chat_id, 1)
    assert [item["question"] for item in saved["turns"]] == ["q1"]
    assert store.citation_count(chat_id) == 1


def test_truncate_before_accepts_zero_and_rejects_invalid_turns(tmp_path):
    store = ChatStore(str(tmp_path))
    chat_id = store.create()["id"]
    store.append_turn(chat_id, turn())

    assert store.truncate_before(chat_id, 0)["turns"] == []
    assert store.truncate_before("missing", 0) is None
    with pytest.raises(ValueError, match="turn_index"):
        store.truncate_before(chat_id, 1)


def test_list_and_delete_sessions(tmp_path):
    store = ChatStore(str(tmp_path))
    first = store.append_turn(store.create()["id"], turn(), name="First")
    second = store.append_turn(store.create()["id"], turn(), name="Second")

    assert {row["id"] for row in store.list_all()} == {first["id"], second["id"]}
    store.delete(first["id"])
    assert store.get(first["id"]) is None
    assert [row["id"] for row in store.list_all()] == [second["id"]]


def test_write_failure_leaves_prior_file_intact(tmp_path, monkeypatch):
    import server.chats as chats_mod

    store = ChatStore(str(tmp_path))
    chat = store.append_turn(store.create()["id"], turn("q", "a"), name="Alpha")

    monkeypatch.setattr(
        chats_mod.json, "dumps", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("full"))
    )
    with pytest.raises(OSError, match="full"):
        store.append_turn(chat["id"], turn("q2", "a2"))

    assert [item["question"] for item in store.get(chat["id"])["turns"]] == ["q"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_inflight_stop_lifecycle(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.try_acquire("c1") is True
    assert store.try_acquire("c1") is False
    assert isinstance(store.stop_event("c1"), threading.Event)
    assert store.request_stop("c1") is True
    assert store.stop_event("c1").is_set()
    store.release("c1")
    assert store.stop_event("c1") is None


def test_inflight_events_are_isolated_and_fresh(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.stop_event("missing") is None
    assert store.request_stop("missing") is False
    assert store.try_acquire("first") is True
    assert store.try_acquire("second") is True
    first_event = store.stop_event("first")
    store.release("first")
    assert store.try_acquire("first") is True
    assert store.stop_event("first") is not first_event


def test_generate_name_falls_back_when_llm_errors(monkeypatch):
    from rag.config import AnthropicSpec

    spec = AnthropicSpec(api_key_env="DEFINITELY_UNSET_KEY_XYZ")
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_XYZ", raising=False)
    assert generate_name("How does MLA work?", spec) == "New chat"
