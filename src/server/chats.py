"""File-backed chat session store.

Each session is one JSON file in the configured `chat_history` directory:

    {
      "id": "...", "name": "KV cache in DeepSeek-V3",
      "created_at": "...", "updated_at": "...",
      "messages":  [{"role": "user", "content": "..."}, ...],   # ChatML
      "citations": [null, [<citation>, ...], ...],              # parallel to messages
      "feedback":  [null, {"vote": "up", "note": "...", "updated_at": "..."}, ...],
      "usage":     [null, {"input_tokens": 512, "output_tokens": 128, "latency_ms": 2100}, ...],
      "per_paper": [null, true, ...],
      "compare":   [null, false, ...],
      "compare_results": [null, null, ...]   # or [<row>, ...] on a Compare turn
    }

`messages` is a plain ChatML list; `citations[i]` holds the grounded sources for
`messages[i]` (null for user turns) so the UI can re-link them after reload. `feedback[i]`
holds the 👍/👎 + optional note left on that assistant turn, set via `set_feedback`
(null until a vote is cast, and always null for user turns). `usage[i]` holds the token
counts + latency for that assistant turn (null for user turns; token counts may also be
null if the LLM backend never reported usage). `per_paper[i]` holds the per-paper-retrieval
toggle's value for that assistant turn (null for user turns) — the UI reads the last entry
to restore the toggle to what the conversation's latest message actually used, rather than
resetting it to a default that may not match. `compare[i]` is the same pattern for the
Compare-mode toggle; `compare_results[i]` holds that turn's per-paper carousel data
(`[{paper_id, title, arxiv_id, text, citations, trace}, ...]`, one entry per paper) when
`compare[i]` is true, `null` otherwise — this is what lets a reloaded conversation restore
the carousel instead of just the final synthesized answer with no drill-down. `auto[i]` is
`true` when the message's answer shape (Ask vs Compare) was resolved by Auto mode's
classification rather than picked directly by the user — badge-only, doesn't change how
`compare[i]` is interpreted.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from rag.config import LLMSpec
from rag.llm import build_llm


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ChatStore:
    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}

    def try_acquire(self, chat_id: str) -> bool:
        """Claim a chat for one in-flight turn. Returns False if one is already running,
        so an edit-and-resume's truncate + append reads a history no concurrent request
        can be mutating."""
        with self._lock:
            if chat_id in self._inflight:
                return False
            self._inflight[chat_id] = threading.Event()
            return True

    def release(self, chat_id: str) -> None:
        with self._lock:
            self._inflight.pop(chat_id, None)

    def stop_event(self, chat_id: str) -> threading.Event | None:
        """The in-flight turn's stop signal, or None if no turn is currently running for
        this chat. The route handler grabs this right after a successful `try_acquire`
        and threads it down into the LLM backend's streaming loop as `stop_check`;
        `request_stop` (below) is what sets it."""
        with self._lock:
            return self._inflight.get(chat_id)

    def request_stop(self, chat_id: str) -> bool:
        """Signal the in-flight turn for `chat_id` to stop generating at its next
        checkpoint. Returns False if no turn is currently running (e.g. it already
        finished) — the caller can treat that as a harmless no-op, not an error."""
        event = self.stop_event(chat_id)
        if event is None:
            return False
        event.set()
        return True

    def _path(self, chat_id: str) -> Path:
        return self.dir / f"{chat_id}.json"

    def list_all(self) -> list[dict]:
        out = []
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
                out.append(
                    {
                        "id": d["id"],
                        "name": d.get("name") or "New chat",
                        "updated_at": d.get("updated_at", ""),
                    }
                )
            except Exception:
                continue
        out.sort(key=lambda x: x["updated_at"], reverse=True)
        return out

    def get(self, chat_id: str) -> dict | None:
        p = self._path(chat_id)
        return json.loads(p.read_text()) if p.exists() else None

    def create(self) -> dict:
        chat = {
            "id": uuid.uuid4().hex[:12],
            "name": "",
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
            "citations": [],
            "traces": [],
            "feedback": [],
            "usage": [],
            "per_paper": [],
            "compare": [],
            "compare_results": [],
            "auto": [],
        }
        self._write(chat)
        return chat

    def append_turn(
        self,
        chat_id: str,
        user_content: str,
        assistant_content: str,
        citations: list,
        trace: list,
        usage: dict | None = None,
        name: str | None = None,
        per_paper: bool = False,
        compare: bool = False,
        compare_results: list | None = None,
        auto: bool = False,
    ) -> dict:
        """Append one user+assistant exchange, preserving prior turns."""
        with self._lock:
            chat = self.get(chat_id) or {
                "id": chat_id,
                "name": "",
                "created_at": _now(),
                "messages": [],
                "citations": [],
                "traces": [],
            }
            chat.setdefault("traces", [])
            chat.setdefault("usage", [])
            chat.setdefault("per_paper", [])
            chat.setdefault("compare", [])
            chat.setdefault("compare_results", [])
            chat.setdefault("auto", [])
            # Keep parallel arrays aligned even for sessions created before traces/usage.
            while len(chat["traces"]) < len(chat["messages"]):
                chat["traces"].append(None)
            while len(chat["usage"]) < len(chat["messages"]):
                chat["usage"].append(None)
            while len(chat["per_paper"]) < len(chat["messages"]):
                chat["per_paper"].append(None)
            while len(chat["compare"]) < len(chat["messages"]):
                chat["compare"].append(None)
            while len(chat["compare_results"]) < len(chat["messages"]):
                chat["compare_results"].append(None)
            while len(chat["auto"]) < len(chat["messages"]):
                chat["auto"].append(None)
            chat["messages"].append({"role": "user", "content": user_content})
            chat["citations"].append(None)
            chat["traces"].append(None)
            chat["usage"].append(None)
            chat["per_paper"].append(None)
            chat["compare"].append(None)
            chat["compare_results"].append(None)
            chat["auto"].append(None)
            chat["messages"].append({"role": "assistant", "content": assistant_content})
            chat["citations"].append(citations)
            chat["traces"].append(trace)
            chat["usage"].append(usage)
            chat["per_paper"].append(per_paper)
            chat["compare"].append(compare)
            chat["compare_results"].append(compare_results)
            chat["auto"].append(auto)
            if name is not None:
                chat["name"] = name
            chat["updated_at"] = _now()
            self._write(chat)
            return chat

    def delete(self, chat_id: str) -> None:
        p = self._path(chat_id)
        if p.exists():
            p.unlink()

    def set_feedback(
        self, chat_id: str, index: int, vote: str | None, note: str | None
    ) -> dict | None:
        """Set or clear 👍/👎 + note feedback on one assistant turn.

        Returns `None` if `chat_id` doesn't exist; raises `ValueError` for an invalid
        `index`/role/vote-note combination.
        """
        with self._lock:
            chat = self.get(chat_id)
            if chat is None:
                return None
            if not (0 <= index < len(chat["messages"])):
                raise ValueError("index out of range")
            if chat["messages"][index]["role"] != "assistant":
                raise ValueError("feedback only applies to assistant turns")
            if vote not in ("up", "down", None):
                raise ValueError("vote must be 'up', 'down', or None")
            if note is not None and vote is None:
                raise ValueError("note requires a vote")
            chat.setdefault("feedback", [])
            # Keep the array aligned even for sessions created before this field existed.
            while len(chat["feedback"]) < len(chat["messages"]):
                chat["feedback"].append(None)
            chat["feedback"][index] = (
                {"vote": vote, "note": note, "updated_at": _now()} if vote else None
            )
            chat["updated_at"] = _now()
            self._write(chat)
            return chat

    def truncate_at(self, chat_id: str, index: int) -> dict | None:
        """Drop messages[index:] and the parallel arrays' tails (an edit-and-resume).

        Returns `None` if `chat_id` doesn't exist; raises `ValueError` for an
        out-of-range index or one that doesn't land on a user turn.
        """
        with self._lock:
            chat = self.get(chat_id)
            if chat is None:
                return None
            if not (0 <= index < len(chat["messages"])):
                raise ValueError("index out of range")
            if chat["messages"][index]["role"] != "user":
                raise ValueError("edit index must be a user turn")
            for key in (
                "messages",
                "citations",
                "traces",
                "feedback",
                "usage",
                "per_paper",
                "compare",
                "compare_results",
                "auto",
            ):
                chat[key] = chat.get(key, [])[:index]
            chat["updated_at"] = _now()
            self._write(chat)
            return chat

    def _write(self, chat: dict) -> None:
        # Write-temp-then-rename so a crash or full disk mid-write can't leave a
        # truncated JSON file that get() would then fail to parse forever.
        p = self._path(chat["id"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(chat, indent=2))
        tmp.replace(p)


def generate_name(first_user_msg: str, spec: LLMSpec) -> str:
    """Ask the LLM for a short session title. Best-effort; falls back gracefully."""
    prompt = (
        "Give a concise 3-6 word title (no quotes, no trailing punctuation) for a "
        f"conversation that begins with:\n\n{first_user_msg}\n\nReply with only the title."
    )
    try:
        # Budget generously: reasoning models (e.g. gpt-oss) spend tokens on a
        # hidden reasoning channel first and return empty content if capped too low.
        raw = build_llm(spec).complete(
            system="You write concise chat titles.", user=prompt, max_tokens=256
        )
        title = raw.strip().strip('"').splitlines()[0].strip()
        return title[:60] or "New chat"
    except Exception:
        return "New chat"
