"""File-backed chat session store.

Each session is one JSON file in the configured `chat_history` directory:

    {
      "id": "...", "name": "KV cache in DeepSeek-V3",
      "created_at": "...", "updated_at": "...",
      "messages":  [{"role": "user", "content": "..."}, ...],   # ChatML
      "citations": [null, [<citation>, ...], ...],              # parallel to messages
      "feedback":  [null, {"vote": "up", "note": "...", "updated_at": "..."}, ...],
      "usage":     [null, {"input_tokens": 512, "output_tokens": 128, "latency_ms": 2100}, ...]
    }

`messages` is a plain ChatML list; `citations[i]` holds the grounded sources for
`messages[i]` (null for user turns) so the UI can re-link them after reload. `feedback[i]`
holds the 👍/👎 + optional note left on that assistant turn, set via `set_feedback`
(null until a vote is cast, and always null for user turns). `usage[i]` holds the token
counts + latency for that assistant turn (null for user turns; token counts may also be
null if the LLM backend never reported usage).
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
            # Keep parallel arrays aligned even for sessions created before traces/usage.
            while len(chat["traces"]) < len(chat["messages"]):
                chat["traces"].append(None)
            while len(chat["usage"]) < len(chat["messages"]):
                chat["usage"].append(None)
            chat["messages"].append({"role": "user", "content": user_content})
            chat["citations"].append(None)
            chat["traces"].append(None)
            chat["usage"].append(None)
            chat["messages"].append({"role": "assistant", "content": assistant_content})
            chat["citations"].append(citations)
            chat["traces"].append(trace)
            chat["usage"].append(usage)
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
