"""File-backed chat session store + session naming."""

from __future__ import annotations

import threading

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


def test_append_turn_stores_usage_parallel_to_messages(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    usage = {"input_tokens": 100, "output_tokens": 20, "latency_ms": 1500}
    saved = store.append_turn(chat["id"], "hi", "hello", [], [], usage)

    assert saved["usage"] == [None, usage]


def test_append_turn_pads_legacy_sessions_without_usage(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    # Simulate a pre-usage session: messages present, usage key missing entirely.
    chat["messages"] = [{"role": "user", "content": "old"}]
    chat["citations"] = [None]
    del chat["usage"]
    store._write(chat)

    saved = store.append_turn(chat["id"], "q", "a", [], [])
    assert len(saved["usage"]) == len(saved["messages"])


def test_append_turn_stores_per_paper_parallel_to_messages(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    saved = store.append_turn(chat["id"], "hi", "hello", [], [], per_paper=True)

    assert saved["per_paper"] == [None, True]


def test_append_turn_defaults_per_paper_to_false(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    saved = store.append_turn(chat["id"], "hi", "hello", [], [])

    assert saved["per_paper"] == [None, False]


def test_append_turn_pads_legacy_sessions_without_per_paper(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    # Simulate a pre-per_paper session: messages present, per_paper key missing entirely.
    chat["messages"] = [{"role": "user", "content": "old"}]
    chat["citations"] = [None]
    del chat["per_paper"]
    store._write(chat)

    saved = store.append_turn(chat["id"], "q", "a", [], [])
    assert len(saved["per_paper"]) == len(saved["messages"])


def test_append_turn_stores_compare_and_compare_results_parallel_to_messages(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    rows = [
        {"paper_id": "p", "title": "P", "arxiv_id": None, "text": "x", "citations": [], "trace": []}
    ]
    saved = store.append_turn(chat["id"], "hi", "hello", [], [], compare=True, compare_results=rows)

    assert saved["compare"] == [None, True]
    assert saved["compare_results"] == [None, rows]


def test_append_turn_defaults_compare_to_false_and_results_to_none(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    saved = store.append_turn(chat["id"], "hi", "hello", [], [])

    assert saved["compare"] == [None, False]
    assert saved["compare_results"] == [None, None]


def test_append_turn_pads_legacy_sessions_without_compare(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    # Simulate a pre-compare session: messages present, compare/compare_results missing.
    chat["messages"] = [{"role": "user", "content": "old"}]
    chat["citations"] = [None]
    del chat["compare"]
    del chat["compare_results"]
    store._write(chat)

    saved = store.append_turn(chat["id"], "q", "a", [], [])
    assert len(saved["compare"]) == len(saved["messages"])
    assert len(saved["compare_results"]) == len(saved["messages"])


def test_append_turn_stores_auto_parallel_to_messages(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    saved = store.append_turn(chat["id"], "hi", "hello", [], [], auto=True)

    assert saved["auto"] == [None, True]


def test_append_turn_defaults_auto_to_false(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    saved = store.append_turn(chat["id"], "hi", "hello", [], [])

    assert saved["auto"] == [None, False]


def test_append_turn_pads_legacy_sessions_without_auto(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    # Simulate a pre-auto session: messages present, auto key missing entirely.
    chat["messages"] = [{"role": "user", "content": "old"}]
    chat["citations"] = [None]
    del chat["auto"]
    store._write(chat)

    saved = store.append_turn(chat["id"], "q", "a", [], [])
    assert len(saved["auto"]) == len(saved["messages"])


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


def test_truncate_at_drops_tail_across_all_parallel_arrays(tmp_path):
    store = ChatStore(str(tmp_path))
    chat_id = store.create()["id"]
    store.append_turn(chat_id, "q1", "a1 [r1]", [{"ref": "r1"}], [{"type": "action"}])
    store.append_turn(chat_id, "q2", "a2 [r2]", [{"ref": "r2"}], [{"type": "action"}])
    store.set_feedback(chat_id, 3, "up", "good")

    truncated = store.truncate_at(chat_id, 2)  # drop the second exchange (index 2 = "q2")
    assert [m["content"] for m in truncated["messages"]] == ["q1", "a1 [r1]"]
    assert truncated["citations"] == [None, [{"ref": "r1"}]]
    assert truncated["traces"] == [None, [{"type": "action"}]]
    assert truncated["feedback"] == [None, None]
    assert len(truncated["usage"]) == 2
    assert len(truncated["per_paper"]) == 2
    assert len(truncated["compare"]) == 2
    assert len(truncated["compare_results"]) == 2
    assert len(truncated["auto"]) == 2


def test_truncate_at_index_zero_drops_everything(tmp_path):
    store = ChatStore(str(tmp_path))
    chat_id = store.create()["id"]
    store.append_turn(chat_id, "q1", "a1", [], [])

    truncated = store.truncate_at(chat_id, 0)
    assert truncated["messages"] == []
    assert truncated["citations"] == []


def test_truncate_at_rejects_out_of_range_index(tmp_path):
    store = ChatStore(str(tmp_path))
    chat_id = store.create()["id"]
    store.append_turn(chat_id, "q", "a", [], [])
    with pytest.raises(ValueError):
        store.truncate_at(chat_id, 5)


def test_truncate_at_rejects_assistant_turn_index(tmp_path):
    store = ChatStore(str(tmp_path))
    chat_id = store.create()["id"]
    store.append_turn(chat_id, "q", "a", [], [])
    with pytest.raises(ValueError):
        store.truncate_at(chat_id, 1)  # index 1 is the assistant turn


def test_truncate_at_returns_none_for_missing_chat(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.truncate_at("missing", 0) is None


def test_try_acquire_then_release_allows_reacquire(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.try_acquire("c1") is True
    assert store.try_acquire("c1") is False  # already in flight
    store.release("c1")
    assert store.try_acquire("c1") is True  # free again after release


def test_try_acquire_is_independent_per_chat(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.try_acquire("c1") is True
    assert store.try_acquire("c2") is True  # unrelated chat, not blocked


def test_stop_event_is_none_when_not_in_flight(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.stop_event("c1") is None


def test_stop_event_available_after_try_acquire(tmp_path):
    store = ChatStore(str(tmp_path))
    store.try_acquire("c1")
    assert isinstance(store.stop_event("c1"), threading.Event)


def test_try_acquire_gives_a_fresh_event_each_turn(tmp_path):
    # A stale stop from a previous (finished) turn on this chat must not leak into the
    # next one — otherwise a fresh turn would look pre-stopped from its first checkpoint.
    store = ChatStore(str(tmp_path))
    store.try_acquire("c1")
    first = store.stop_event("c1")
    store.release("c1")
    store.try_acquire("c1")
    assert store.stop_event("c1") is not first


def test_request_stop_sets_the_inflight_event(tmp_path):
    store = ChatStore(str(tmp_path))
    store.try_acquire("c1")
    assert store.request_stop("c1") is True
    assert store.stop_event("c1").is_set()


def test_request_stop_returns_false_when_not_in_flight(tmp_path):
    store = ChatStore(str(tmp_path))
    assert store.request_stop("c1") is False


def test_release_clears_the_stop_event(tmp_path):
    store = ChatStore(str(tmp_path))
    store.try_acquire("c1")
    store.release("c1")
    assert store.stop_event("c1") is None


def test_generate_name_falls_back_when_llm_errors(monkeypatch):
    from rag.config import AnthropicSpec

    # No api_base + no key -> build_llm(...).complete raises -> graceful fallback.
    spec = AnthropicSpec(api_key_env="DEFINITELY_UNSET_KEY_XYZ")
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_XYZ", raising=False)
    assert generate_name("How does MLA work?", spec) == "New chat"
