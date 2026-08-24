"""Turn orchestration for ``/api/chat``: get-or-build the agent, run it, persist, stream."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

from rag.config import LLMSpec
from rag.llm import Usage

from .agent import ChatAgent, _flatten_compare_rows
from .chats import ChatStore, generate_name
from .schemas import ChatRequest

# How often _run_agent/_run_agent_compare check whether they should give up waiting on the
# agent thread — the bound on how long Stop takes to unlock the chat, regardless of what
# the agent thread is actually blocked on.
_ABANDON_POLL_INTERVAL_S = 0.1


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


def _run_agent(
    agent: ChatAgent,
    req: ChatRequest,
    on_trace: Callable[[dict], None],
    emit: Callable[[str, str], None],
    ref_start: int,
    stop_check: Callable[[], bool] | None,
) -> tuple[str, list[dict], Usage]:
    """Runs ``agent.run()`` via `_run_in_thread_with_abandon` and returns its
    ``(text, citations, usage)`` — or, if abandoned, whatever text streamed so far with
    empty citations (see `_run_in_thread_with_abandon`'s docstring for why).

    Once abandoned, `on_text`/`on_trace` stop forwarding to `emit` — the SSE stream this
    turn owns is about to end (`run_turn` moves straight on to persisting), so any further
    tokens/trace steps the abandoned thread produces would otherwise queue up for a
    stream nobody's reading anymore.
    """
    parts: list[str] = []
    abandoned = threading.Event()

    def on_text(t: str) -> None:
        if abandoned.is_set():
            return
        parts.append(t)
        emit("token", t)

    def guarded_on_trace(e: dict) -> None:
        if abandoned.is_set():
            return
        on_trace(e)

    result_value: tuple[str, list[dict], Usage] | None = None
    result_error: Exception | None = None

    def _target() -> None:
        nonlocal result_value, result_error
        try:
            result_value = agent.run(
                [m.model_dump() for m in req.messages],
                req.tags,
                req.papers,
                on_text=on_text,
                on_trace=guarded_on_trace,
                ref_start=ref_start,
                per_paper=req.per_paper,
                stop_check=stop_check,
            )
        except Exception as e:  # re-raised below, but only in the thread that waited it out
            result_error = e

    if not _run_in_thread_with_abandon(_target, stop_check):
        abandoned.set()
        return "".join(parts), [], Usage(None, None)

    if result_error is not None:
        raise result_error
    assert result_value is not None  # thread finished without setting either result slot
    return result_value


def _run_agent_compare(
    agent: ChatAgent,
    req: ChatRequest,
    on_trace: Callable[[dict], None],
    emit: Callable[[str, str], None],
    ref_start: int,
    stop_check: Callable[[], bool] | None,
) -> tuple[str, list[dict], list[dict], Usage]:
    """Compare-mode counterpart to `_run_agent` — same `_run_in_thread_with_abandon` race,
    but accumulates `rows` (parallel to `_run_agent`'s `parts`) so an abandoned turn still
    returns a usable `compare_results`/`citations` pair via `_flatten_compare_rows` — the
    same helper `ChatAgent.compare` uses for its own internal stop-before-synthesis
    fallback, so both stop paths degrade the same way."""
    rows: list[dict] = []
    abandoned = threading.Event()

    def on_text(t: str) -> None:
        if abandoned.is_set():
            return
        emit("token", t)

    def guarded_on_trace(e: dict) -> None:
        if abandoned.is_set():
            return
        on_trace(e)

    def on_row(row: dict) -> None:
        if abandoned.is_set():
            return
        rows.append(row)
        emit("compare_row", json.dumps(row))

    result_value: tuple[str, list[dict], list[dict], Usage] | None = None
    result_error: Exception | None = None

    def _target() -> None:
        nonlocal result_value, result_error
        try:
            result_value = agent.compare(
                [m.model_dump() for m in req.messages],
                req.tags,
                req.papers,
                on_text=on_text,
                on_row=on_row,
                on_trace=guarded_on_trace,
                ref_start=ref_start,
                stop_check=stop_check,
            )
        except Exception as e:
            result_error = e

    if not _run_in_thread_with_abandon(_target, stop_check):
        abandoned.set()
        text, citations = _flatten_compare_rows(rows)
        return text, rows, citations, Usage(None, None)

    if result_error is not None:
        raise result_error
    assert result_value is not None
    return result_value


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
        if req.edit_index is not None and req.chat_id:
            chats.truncate_at(req.chat_id, req.edit_index)
        # Existing chats already carry ref-numbered citations for prior turns — offset this
        # turn's numbering past them so a follow-up question continues (r4, r5, ...) instead
        # of restarting at r1 and colliding with refs already shown for a different paper
        # earlier in the chat.
        existing = chats.get(req.chat_id) if req.chat_id else None
        ref_start = sum(len(c) for c in existing.get("citations", []) if c) if existing else 0
        # Spans the whole turn (retrieval + rerank + LLM + faithfulness check), not just LLM
        # think time — that's the wait the user actually feels.
        t0 = time.perf_counter()
        if req.compare:
            text, compare_results, citations, usage = _run_agent_compare(
                agent, req, on_trace, emit, ref_start, stop_check
            )
        else:
            text, citations, usage = _run_agent(agent, req, on_trace, emit, ref_start, stop_check)
            compare_results = None
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage_payload = {
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
            saved = chats.append_turn(
                req.chat_id,
                req.messages[-1].content,
                text,
                citations,
                trace_entries,
                usage_payload,
                name=name,
                # The secondary knob is hidden/unsendable under Compare (see ChatPage), but
                # the backend doesn't trust the client alone: never persist both as true.
                per_paper=req.per_paper and not req.compare,
                compare=req.compare,
                compare_results=compare_results,
            )
            emit("meta", json.dumps({"chat_id": saved["id"], "name": saved["name"]}))
    except Exception as e:  # surface errors to the client
        emit("error", f"{type(e).__name__}: {e}")
    finally:
        emit("done", "")
