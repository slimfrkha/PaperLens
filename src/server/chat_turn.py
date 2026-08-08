"""Turn orchestration for ``/api/chat``: get-or-build the agent, run it, persist, stream."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from rag.config import LLMSpec

from .agent import ChatAgent
from .chats import ChatStore, generate_name
from .schemas import ChatRequest


def run_turn(
    get_agent: Callable[[], ChatAgent],
    chats: ChatStore,
    req: ChatRequest,
    emit: Callable[[str, str], None],
    tagging_spec: LLMSpec,
) -> None:
    """Run one chat turn, streaming events via ``emit`` and persisting the result.

    Always emits a terminal ``"done"`` event, even on failure — ``emit``'s caller relies on
    it to end the SSE stream. Doesn't manage the chat's single-flight guard
    (``ChatStore.try_acquire``/``release``): that's route-level state tied to the early
    ``409`` response, owned by the caller.
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
        text, citations, usage = agent.run(
            [m.model_dump() for m in req.messages],
            req.tags,
            req.papers,
            on_text=lambda t: emit("token", t),
            on_trace=on_trace,
            ref_start=ref_start,
            per_paper=req.per_paper,
        )
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
                per_paper=req.per_paper,
            )
            emit("meta", json.dumps({"chat_id": saved["id"], "name": saved["name"]}))
    except Exception as e:  # surface errors to the client
        emit("error", f"{type(e).__name__}: {e}")
    finally:
        emit("done", "")
