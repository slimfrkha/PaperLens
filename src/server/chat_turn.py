"""Turn orchestration for ``/api/chat``: get-or-build the agent, run it, persist, stream."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from rag.config import LLMSpec
from rag.llm import Usage

from .agent import ChatAgent, InsufficientScopeError, _flatten_compare_rows
from .chats import ChatStore, StoredTurn, UsagePayload, generate_name
from .schemas import ChatRequest

# How often _run_agent/_run_agent_compare check whether they should give up waiting on the
# agent thread — the bound on how long Stop takes to unlock the chat, regardless of what
# the agent thread is actually blocked on.
_ABANDON_POLL_INTERVAL_S = 0.1


@dataclass
class _InvocationResult:
    """The one result shape ``run_turn`` needs from either answer mode."""

    text: str
    citations: list[dict]
    usage: Usage
    compare_results: list[dict] | None = None


def _run_in_thread_with_abandon(
    target: Callable[[], None], stop_check: Callable[[], bool] | None
) -> bool:
    """Runs `target` (which stores its own result via closure) in a daemon thread, polling
    until it finishes or `stop_check` fires first. Returns True if the thread finished
    normally (its result is ready to read), False if it was abandoned instead.

    A stop signal can't always make the blocked call return promptly on its own — whether
    closing the LLM connection actually interrupts a thread stuck on a network read
    (mid-prefill on a local model) or on retrieval (embedding/reranking) depends on the
    SDK/platform, and isn't guaranteed. So once `stop_check` fires this stops *waiting* on
    the thread rather than depending on it to return: the abandoned thread — which may
    still be running, possibly for a long time — is left to finish or error out on its own,
    and its eventual result is simply never read by the caller.

    Shared by both `_run_agent` and `_run_agent_compare` so this abandon race lives in one
    place — only the fallback *value* each builds when this returns False differs.
    """
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=_ABANDON_POLL_INTERVAL_S)
        if thread.is_alive() and stop_check and stop_check():
            return False
    return True


class _TurnInvocation:
    """Owns one agent call's abandon race and late-event suppression."""

    def __init__(self, stop_check: Callable[[], bool] | None):
        self._stop_check = stop_check
        self._abandoned = threading.Event()
        self._lock = threading.Lock()

    def forward(self, callback: Callable[[], None]) -> None:
        """Run an SSE or trace callback only while this invocation owns the stream."""
        # The callback can mutate a partial-result accumulator as well as emit SSE. Hold
        # this lock through both the active check and the callback so abandonment cannot
        # snapshot the accumulator (or emit ``done``) halfway through either operation.
        with self._lock:
            if not self._abandoned.is_set():
                callback()

    def run(
        self,
        target: Callable[[], _InvocationResult],
        abandoned_result: Callable[[], _InvocationResult],
    ) -> _InvocationResult:
        result_value: _InvocationResult | None = None
        result_error: Exception | None = None

        def run_target() -> None:
            nonlocal result_value, result_error
            try:
                result_value = target()
            except Exception as e:  # re-raised only if this invocation waited for it
                result_error = e

        if not _run_in_thread_with_abandon(run_target, self._stop_check):
            # Serialize with `forward`: a callback already in progress completes before
            # this captures the partial result; callbacks that arrive after this point are
            # ignored altogether.
            with self._lock:
                self._abandoned.set()
                return abandoned_result()

        if result_error is not None:
            raise result_error
        assert result_value is not None  # target returned without a result or an error
        return result_value


def _run_agent(
    agent: ChatAgent,
    req: ChatRequest,
    on_trace: Callable[[dict], None],
    emit: Callable[[str, str], None],
    ref_start: int,
    stop_check: Callable[[], bool] | None,
) -> _InvocationResult:
    """Invoke Ask, retaining its streamed-text partial-result rule."""
    parts: list[str] = []
    invocation = _TurnInvocation(stop_check)

    def on_text(t: str) -> None:
        def forward() -> None:
            parts.append(t)
            emit("token", t)

        invocation.forward(forward)

    def guarded_on_trace(e: dict) -> None:
        invocation.forward(lambda: on_trace(e))

    def target() -> _InvocationResult:
        text, citations, usage = agent.run(
            [m.model_dump() for m in req.messages],
            req.tags,
            req.papers,
            on_text=on_text,
            on_trace=guarded_on_trace,
            ref_start=ref_start,
            per_paper=req.per_paper,
            stop_check=stop_check,
        )
        return _InvocationResult(text, citations, usage)

    return invocation.run(target, lambda: _InvocationResult("".join(parts), [], Usage(None, None)))


def _run_agent_compare(
    agent: ChatAgent,
    req: ChatRequest,
    on_trace: Callable[[dict], None],
    emit: Callable[[str, str], None],
    ref_start: int,
    stop_check: Callable[[], bool] | None,
) -> _InvocationResult:
    """Invoke Compare, retaining its completed-row partial-result rule."""
    rows: list[dict] = []
    invocation = _TurnInvocation(stop_check)

    def on_text(t: str) -> None:
        invocation.forward(lambda: emit("token", t))

    def guarded_on_trace(e: dict) -> None:
        invocation.forward(lambda: on_trace(e))

    def on_row(row: dict) -> None:
        def forward() -> None:
            rows.append(row)
            emit("compare_row", json.dumps(row))

        invocation.forward(forward)

    def target() -> _InvocationResult:
        text, completed_rows, citations, usage = agent.compare(
            [m.model_dump() for m in req.messages],
            req.tags,
            req.papers,
            on_text=on_text,
            on_row=on_row,
            on_trace=guarded_on_trace,
            ref_start=ref_start,
            stop_check=stop_check,
        )
        return _InvocationResult(text, citations, usage, completed_rows)

    def abandoned_result() -> _InvocationResult:
        text, citations = _flatten_compare_rows(rows)
        return _InvocationResult(text, citations, Usage(None, None), rows)

    return invocation.run(target, abandoned_result)


def run_turn(
    get_agent: Callable[[], ChatAgent],
    chats: ChatStore,
    req: ChatRequest,
    emit: Callable[[str, str], None],
    tagging_spec: LLMSpec,
    stop_check: Callable[[], bool] | None = None,
) -> None:
    """Run one chat turn, streaming events via ``emit`` and persisting the result.

    Always emits a terminal ``"done"`` event, even on failure — ``emit``'s caller relies on
    it to end the SSE stream. Doesn't manage the chat's single-flight guard
    (``ChatStore.try_acquire``/``release``): that's route-level state tied to the early
    ``409`` response, owned by the caller.

    ``stop_check``, if given, drives ``_run_agent``'s stop handling (see there) — a
    stopped turn's partial text is persisted exactly like a normal completed turn (no
    special-casing below).
    """
    trace_entries: list = []

    def on_trace(e):
        trace_entries.append(e)
        emit("trace", json.dumps(e))

    try:
        # Built here, not before this call: get_agent() is the first-touch lazy model build
        # and can throw (bad key, cold cloud client) — the caller's guard-release still runs
        # in its own finally regardless, since it wraps this whole call.
        agent = get_agent()
        # An edit-and-resume truncates the stored tail before this turn is computed, so
        # ref_start below reads only the retained history — the caller's single-flight guard
        # ensures no concurrent request can interleave.
        if req.edit_turn is not None and req.chat_id:
            chats.truncate_before(req.chat_id, req.edit_turn)
        # Existing chats already carry ref-numbered citations for prior turns — offset this
        # turn's numbering past them so a follow-up question continues (r4, r5, ...) instead
        # of restarting at r1 and colliding with refs already shown for a different paper
        # earlier in the chat.
        existing = chats.get(req.chat_id) if req.chat_id else None
        ref_start = chats.citation_count(req.chat_id) if req.chat_id else 0
        # Spans the whole turn (retrieval + rerank + LLM + faithfulness check), not just LLM
        # think time — that's the wait the user actually feels.
        t0 = time.perf_counter()
        # Tracks what actually ran, which may differ from req.compare on the fallback below —
        # used for persistence instead of req.compare so a degraded turn isn't saved as if it
        # were a real Compare turn with no rows.
        ran_compare = req.compare
        if req.compare:
            try:
                result = _run_agent_compare(agent, req, on_trace, emit, ref_start, stop_check)
            except InsufficientScopeError:
                # Classification (Auto mode's own pre-flight round trip) or the user's click
                # can both race a manifest change between "compare was decided" and this turn
                # actually running — a paper removed mid-flight can drop resolved scope below
                # 2 and trip compare()'s own guard. Fall back to a plain Ask turn instead of
                # surfacing that race as a 500; the guard raises before any streaming starts,
                # so nothing has been emitted yet. Narrowed to this specific exception (not a
                # bare ValueError) so a genuine failure elsewhere in compare() — e.g. a
                # synthesis retry that fails twice, after rows have already streamed — surfaces
                # as a real error instead of being silently rerun as an unrelated Ask turn.
                result = _run_agent(agent, req, on_trace, emit, ref_start, stop_check)
                ran_compare = False
        else:
            result = _run_agent(agent, req, on_trace, emit, ref_start, stop_check)
        text = result.text
        citations = result.citations
        usage = result.usage
        compare_results = result.compare_results
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage_payload: UsagePayload = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": latency_ms,
        }
        emit("citations", json.dumps(citations))
        emit("usage", json.dumps(usage_payload))
        # Persist the turn (append user + assistant) and name new sessions.
        if req.chat_id and req.messages:
            name = None
            if not existing or not existing.get("name"):
                name = generate_name(req.messages[0].content, tagging_spec)
            turn: StoredTurn = {
                "question": req.messages[-1].content,
                "answer": text,
                "citations": citations,
                "trace": trace_entries,
                "usage": usage_payload,
                "feedback": None,
                # The secondary knob is hidden/unsendable under Compare (see ChatPage), but
                # the backend doesn't trust the client alone: never persist both as true.
                "per_paper": req.per_paper and not ran_compare,
                "compare": ran_compare,
                "compare_results": compare_results,
                "auto": req.auto,
            }
            saved = chats.append_turn(req.chat_id, turn, name=name)
            emit("meta", json.dumps({"chat_id": saved["id"], "name": saved["name"]}))
    except Exception as e:  # surface errors to the client
        emit("error", f"{type(e).__name__}: {e}")
    finally:
        emit("done", "")
