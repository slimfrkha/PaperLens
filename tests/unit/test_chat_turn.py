"""run_turn: turn orchestration extracted from /api/chat's route closure (offline: real
ChatStore over tmp_path, a fake ChatAgent, no HTTP layer, no monkeypatching)."""

from __future__ import annotations

import json

from rag.config import OpenAISpec
from rag.llm import Usage
from server.chat_turn import run_turn
from server.chats import ChatStore
from server.schemas import ChatMessage, ChatRequest

_TAGGING = OpenAISpec(api_base="http://x")  # generate_name falls back to "New chat" on failure


class _TwoTokenAgent:
    def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0, per_paper=False):
        on_text("foo")
        on_text("bar")
        return "foobar", [{"ref": "r1", "paper_id": "p"}], Usage(10, 5)


class _RefStartAgent:
    """Echoes ref_start into the returned citation's ref, so tests can assert numbering
    continues correctly across turns/edits."""

    def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0, per_paper=False):
        ref = f"r{ref_start + 1}"
        text = f"See [{ref}]."
        on_text(text)
        return text, [{"ref": ref, "paper_id": "p"}], Usage(10, 5)


class _RecordingAgent:
    """Records the per_paper it was called with, so tests can assert run_turn threads
    ChatRequest.per_paper through to the agent."""

    def __init__(self):
        self.received_per_paper = None

    def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0, per_paper=False):
        self.received_per_paper = per_paper
        on_text("ok")
        return "ok", [], Usage(10, 5)


def test_run_turn_streams_tokens_then_citations_then_usage_then_done(tmp_path):
    events: list[tuple[str, str]] = []
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])

    run_turn(
        lambda: _TwoTokenAgent(),
        ChatStore(str(tmp_path)),
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
    )

    assert [e for e, _ in events] == ["token", "token", "citations", "usage", "done"]
    assert events[0][1] + events[1][1] == "foobar"


def test_run_turn_threads_per_paper_to_agent(tmp_path):
    agent = _RecordingAgent()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], per_paper=True)

    run_turn(lambda: agent, ChatStore(str(tmp_path)), req, lambda *a: None, _TAGGING)

    assert agent.received_per_paper is True


def test_run_turn_persists_per_paper_on_the_saved_turn(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")], chat_id=chat["id"], per_paper=True
    )

    run_turn(lambda: _TwoTokenAgent(), store, req, lambda *a: None, _TAGGING)

    saved = store.get(chat["id"])
    assert saved["per_paper"][-1] is True


def test_run_turn_persists_usage_and_citations(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], chat_id=chat["id"])

    run_turn(lambda: _TwoTokenAgent(), store, req, lambda *a: None, _TAGGING)

    saved = store.get(chat["id"])
    assert saved["messages"][-1] == {"role": "assistant", "content": "foobar"}
    assert saved["citations"][-1] == [{"ref": "r1", "paper_id": "p"}]
    assert saved["usage"][-1]["input_tokens"] == 10


def test_run_turn_continues_citation_numbering_across_turns(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    events1: list[tuple[str, str]] = []
    req1 = ChatRequest(messages=[ChatMessage(role="user", content="q1")], chat_id=chat["id"])
    run_turn(lambda: _RefStartAgent(), store, req1, lambda e, d: events1.append((e, d)), _TAGGING)

    # meta fires with a fresh, non-empty name on a chat's first turn.
    meta1 = json.loads(dict(events1)["meta"])
    assert meta1 == {"chat_id": chat["id"], "name": store.get(chat["id"])["name"]}
    assert meta1["name"]

    events2: list[tuple[str, str]] = []
    req2 = ChatRequest(
        messages=[
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="See [r1]."),
            ChatMessage(role="user", content="q2"),
        ],
        chat_id=chat["id"],
    )
    run_turn(lambda: _RefStartAgent(), store, req2, lambda e, d: events2.append((e, d)), _TAGGING)

    # meta still fires on a later turn, but the name isn't clobbered back to a fresh one.
    meta2 = json.loads(dict(events2)["meta"])
    assert meta2["name"] == meta1["name"]

    saved = store.get(chat["id"])
    assert saved["citations"][1][0]["ref"] == "r1"
    assert saved["citations"][3][0]["ref"] == "r2"


def test_run_turn_edit_index_truncates_before_computing_ref_start(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    req1 = ChatRequest(messages=[ChatMessage(role="user", content="q1")], chat_id=chat["id"])
    run_turn(lambda: _RefStartAgent(), store, req1, lambda *a: None, _TAGGING)
    req2 = ChatRequest(
        messages=[
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="See [r1]."),
            ChatMessage(role="user", content="q2"),
        ],
        chat_id=chat["id"],
    )
    run_turn(lambda: _RefStartAgent(), store, req2, lambda *a: None, _TAGGING)

    # Editing turn 0 must truncate back to just that turn *before* ref_start is computed —
    # otherwise the edited turn would wrongly continue at r2 instead of restarting at r1.
    edit_req = ChatRequest(
        messages=[ChatMessage(role="user", content="q1 edited")],
        chat_id=chat["id"],
        edit_index=0,
    )
    run_turn(lambda: _RefStartAgent(), store, edit_req, lambda *a: None, _TAGGING)

    saved = store.get(chat["id"])
    assert [m["content"] for m in saved["messages"]] == ["q1 edited", "See [r1]."]


def test_run_turn_agent_run_error_still_emits_done(tmp_path):
    class _RaisingAgent:
        def run(self, *a, **k):
            raise RuntimeError("boom")

    events: list[tuple[str, str]] = []
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])

    run_turn(
        lambda: _RaisingAgent(),
        ChatStore(str(tmp_path)),
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
    )

    assert [e for e, _ in events] == ["error", "done"]
    assert events[0][1] == "RuntimeError: boom"


def test_run_turn_get_agent_failure_emits_error_and_done(tmp_path):
    def _raising_get_agent():
        raise RuntimeError("cold start failed")

    events: list[tuple[str, str]] = []
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])

    run_turn(
        _raising_get_agent,
        ChatStore(str(tmp_path)),
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
    )

    assert [e for e, _ in events] == ["error", "done"]
    assert events[0][1] == "RuntimeError: cold start failed"
