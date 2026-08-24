"""run_turn: turn orchestration extracted from /api/chat's route closure (offline: real
ChatStore over tmp_path, a fake ChatAgent, no HTTP layer, no monkeypatching)."""

from __future__ import annotations

import json
import threading
import time

from rag.config import OpenAISpec
from rag.llm import Usage
from server.chat_turn import _run_in_thread_with_abandon, run_turn
from server.chats import ChatStore
from server.schemas import ChatMessage, ChatRequest

_TAGGING = OpenAISpec(api_base="http://x")  # generate_name falls back to "New chat" on failure


def test_run_in_thread_with_abandon_returns_true_when_thread_finishes():
    done: list[int] = []
    _run_in_thread_with_abandon(lambda: done.append(1), stop_check=None)
    assert done == [1]


def test_run_in_thread_with_abandon_returns_false_when_stopped_first():
    release = threading.Event()

    def target():
        release.wait()  # blocks until the test lets it go

    stop_flag = threading.Event()
    stop_flag.set()  # already stopped, for a deterministic test

    result = _run_in_thread_with_abandon(target, stop_flag.is_set)
    assert result is False
    release.set()  # let the background thread finish so it doesn't leak past the test


class _TwoTokenAgent:
    def run(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_trace=None,
        ref_start=0,
        per_paper=False,
        stop_check=None,
    ):
        on_text("foo")
        on_text("bar")
        return "foobar", [{"ref": "r1", "paper_id": "p"}], Usage(10, 5)


class _RefStartAgent:
    """Echoes ref_start into the returned citation's ref, so tests can assert numbering
    continues correctly across turns/edits."""

    def run(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_trace=None,
        ref_start=0,
        per_paper=False,
        stop_check=None,
    ):
        ref = f"r{ref_start + 1}"
        text = f"See [{ref}]."
        on_text(text)
        return text, [{"ref": ref, "paper_id": "p"}], Usage(10, 5)


class _RecordingAgent:
    """Records the per_paper/stop_check it was called with, so tests can assert run_turn
    threads ChatRequest.per_paper and the caller's stop_check through to the agent."""

    def __init__(self):
        self.received_per_paper = None
        self.received_stop_check = None

    def run(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_trace=None,
        ref_start=0,
        per_paper=False,
        stop_check=None,
    ):
        self.received_per_paper = per_paper
        self.received_stop_check = stop_check
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


def test_run_turn_threads_stop_check_to_agent(tmp_path):
    agent = _RecordingAgent()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    stop_check = lambda: False  # noqa: E731 — trivial sentinel, identity is what's asserted

    run_turn(lambda: agent, ChatStore(str(tmp_path)), req, lambda *a: None, _TAGGING, stop_check)

    assert agent.received_stop_check is stop_check


def test_run_turn_persists_partial_text_from_a_stopped_agent(tmp_path):
    # A stopped generation isn't an error: run_turn has no special-casing for it — the
    # agent just returns whatever text streamed before the stop, and it's persisted like
    # any other completed turn.
    class _StoppedAgent:
        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            on_text("partial answ")
            return "partial answ", [], Usage(10, 5)

    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], chat_id=chat["id"])

    events: list[tuple[str, str]] = []
    run_turn(lambda: _StoppedAgent(), store, req, lambda e, d: events.append((e, d)), _TAGGING)

    assert [e for e, _ in events] == ["token", "citations", "usage", "meta", "done"]
    saved = store.get(chat["id"])
    assert saved["messages"][-1] == {"role": "assistant", "content": "partial answ"}


def test_run_turn_abandons_an_agent_that_never_returns_once_stopped(tmp_path):
    # The agent's run() here never returns on its own — simulates being stuck deep
    # inside a blocking call (e.g. slow local-model prefill, or retrieval) that a stop
    # signal alone can't interrupt. run_turn must give up waiting and persist whatever
    # streamed so far rather than hanging forever.
    class _NeverReturnsAgent:
        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            on_text("partial")
            threading.Event().wait()  # blocks forever; run_turn must not wait for this

    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], chat_id=chat["id"])
    stop_flag = threading.Event()
    stop_flag.set()  # already stopped before the turn starts, for a deterministic test

    events: list[tuple[str, str]] = []
    run_turn(
        lambda: _NeverReturnsAgent(),
        store,
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
        stop_flag.is_set,
    )

    assert [e for e, _ in events] == ["token", "citations", "usage", "meta", "done"]
    saved = store.get(chat["id"])
    assert saved["messages"][-1] == {"role": "assistant", "content": "partial"}
    assert saved["usage"][-1]["input_tokens"] is None
    assert saved["usage"][-1]["output_tokens"] is None


def test_run_turn_never_persists_the_abandoned_agents_late_result(tmp_path):
    # The agent DOES eventually return here — just after run_turn already gave up
    # waiting on it and persisted the partial answer. Proves the late result is truly
    # discarded rather than racing (or overwriting) what was already saved: the whole
    # point of "stop waiting" instead of "force the call to actually stop".
    finished = threading.Event()

    class _LateFinisherAgent:
        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            on_text("partial")
            time.sleep(0.3)  # past run_turn's 0.1s abandon-poll interval
            finished.set()
            return "full late answer", [{"ref": "r1", "paper_id": "p"}], Usage(10, 5)

    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], chat_id=chat["id"])
    stop_flag = threading.Event()
    stop_flag.set()  # already stopped before the turn starts, for a deterministic test

    run_turn(lambda: _LateFinisherAgent(), store, req, lambda *a: None, _TAGGING, stop_flag.is_set)

    # run_turn already returned (having abandoned the agent thread) at this point — now
    # let the agent's late return actually happen, and confirm it changed nothing.
    assert finished.wait(timeout=5), "test agent never finished"
    saved = store.get(chat["id"])
    assert saved["messages"][-1] == {"role": "assistant", "content": "partial"}
    assert saved["citations"][-1] == []  # not the late [{"ref": "r1", ...}]


def test_run_turn_stops_forwarding_tokens_and_trace_after_abandonment(tmp_path):
    # The agent keeps calling on_text/on_trace after run_turn gave up waiting on it —
    # those calls must not keep reaching the SSE stream (`emit`), which run_turn has
    # already moved on from (straight to persisting + "done").
    finished = threading.Event()

    class _KeepsTalkingAfterAbandonAgent:
        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            on_text("before")
            time.sleep(0.3)  # past run_turn's 0.1s abandon-poll interval
            on_text("after")  # must not reach the SSE stream
            if on_trace:
                on_trace({"type": "action", "query": "late"})  # must not reach it either
            finished.set()
            return "ignored", [], Usage(10, 5)

    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], chat_id=chat["id"])
    stop_flag = threading.Event()
    stop_flag.set()  # already stopped before the turn starts, for a deterministic test

    events: list[tuple[str, str]] = []
    run_turn(
        lambda: _KeepsTalkingAfterAbandonAgent(),
        store,
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
        stop_flag.is_set,
    )

    assert finished.wait(timeout=5), "test agent never finished"
    assert [d for e, d in events if e == "token"] == ["before"]  # "after" never forwarded
    assert not any(e == "trace" for e, _ in events)  # the late trace step never forwarded


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


class _TwoRowCompareAgent:
    """Fake agent implementing `compare` only (not `run`) — proves run_turn dispatches to
    `agent.compare` for a Compare turn rather than `agent.run`, since calling the latter on
    this fake would raise AttributeError."""

    def compare(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_row,
        on_trace=None,
        ref_start=0,
        stop_check=None,
    ):
        row1 = {
            "paper_id": "p1",
            "title": "P1",
            "arxiv_id": None,
            "text": "answer 1",
            "citations": [{"ref": "r1", "paper_id": "p1"}],
            "trace": [],
        }
        row2 = {
            "paper_id": "p2",
            "title": "P2",
            "arxiv_id": None,
            "text": "answer 2",
            "citations": [{"ref": "r2", "paper_id": "p2"}],
            "trace": [],
        }
        on_row(row1)
        on_row(row2)
        on_text("Synth")
        on_text("esized")
        citations = row1["citations"] + row2["citations"]
        return "Synthesized", [row1, row2], citations, Usage(20, 10)


def test_run_turn_dispatches_to_agent_compare_and_emits_compare_row_events(tmp_path):
    events: list[tuple[str, str]] = []
    req = ChatRequest(messages=[ChatMessage(role="user", content="compare them")], compare=True)

    run_turn(
        lambda: _TwoRowCompareAgent(),
        ChatStore(str(tmp_path)),
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
    )

    kinds = [e for e, _ in events]
    assert kinds == ["compare_row", "compare_row", "token", "token", "citations", "usage", "done"]
    rows = [json.loads(d) for e, d in events if e == "compare_row"]
    assert [r["paper_id"] for r in rows] == ["p1", "p2"]


def test_run_turn_persists_compare_and_compare_results(tmp_path):
    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="compare them")],
        chat_id=chat["id"],
        compare=True,
    )

    run_turn(lambda: _TwoRowCompareAgent(), store, req, lambda *a: None, _TAGGING)

    saved = store.get(chat["id"])
    assert saved["compare"][-1] is True
    assert saved["messages"][-1] == {"role": "assistant", "content": "Synthesized"}
    assert [r["paper_id"] for r in saved["compare_results"][-1]] == ["p1", "p2"]
    assert saved["citations"][-1] == [
        {"ref": "r1", "paper_id": "p1"},
        {"ref": "r2", "paper_id": "p2"},
    ]


def test_run_turn_never_persists_per_paper_true_when_compare_true(tmp_path):
    # The secondary knob is hidden/unsendable client-side under Compare, but the backend
    # doesn't trust the client alone: never persist both as true on the same turn.
    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="compare them")],
        chat_id=chat["id"],
        compare=True,
        per_paper=True,
    )

    run_turn(lambda: _TwoRowCompareAgent(), store, req, lambda *a: None, _TAGGING)

    saved = store.get(chat["id"])
    assert saved["per_paper"][-1] is False


def test_run_turn_abandons_a_compare_agent_that_never_finishes(tmp_path):
    row = {
        "paper_id": "p1",
        "title": "P1",
        "arxiv_id": None,
        "text": "partial row",
        "citations": [{"ref": "r1", "paper_id": "p1"}],
        "trace": [],
    }

    class _NeverFinishesCompareAgent:
        def compare(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_row,
            on_trace=None,
            ref_start=0,
            stop_check=None,
        ):
            on_row(row)
            threading.Event().wait()  # blocks forever; run_turn must not wait for this

    store = ChatStore(str(tmp_path))
    chat = store.create()
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="compare")], chat_id=chat["id"], compare=True
    )
    stop_flag = threading.Event()
    stop_flag.set()  # already stopped before the turn starts, for a deterministic test

    events: list[tuple[str, str]] = []
    run_turn(
        lambda: _NeverFinishesCompareAgent(),
        store,
        req,
        lambda e, d: events.append((e, d)),
        _TAGGING,
        stop_flag.is_set,
    )

    assert [e for e, _ in events] == ["compare_row", "citations", "usage", "meta", "done"]
    saved = store.get(chat["id"])
    # abandon-path fallback flattens whatever rows completed, via _flatten_compare_rows —
    # the same helper agent.compare() uses for its own internal stop-before-synthesis path.
    assert saved["messages"][-1]["content"] == "## P1\n\npartial row\n\n"
    assert saved["citations"][-1] == [row["citations"][0]]
    assert saved["compare_results"][-1] == [row]


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
