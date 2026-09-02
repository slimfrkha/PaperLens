"""File-backed chat sessions, stored as coherent user/assistant turns."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, TypedDict

from rag.config import LLMSpec
from rag.llm import build_llm

type Payload = dict[str, Any]
"""JSON object returned by the agent and persisted in a chat turn."""


class UsagePayload(TypedDict):
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


class FeedbackPayload(TypedDict):
    vote: Literal["up", "down"]
    note: str | None
    updated_at: str


class StoredTurn(TypedDict):
    """Everything persisted for one completed user/assistant exchange."""

    question: str
    answer: str
    citations: list[Payload]
    trace: list[Payload]
    usage: UsagePayload | None
    feedback: FeedbackPayload | None
    per_paper: bool
    compare: bool
    compare_results: list[Payload] | None
    auto: bool


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ChatStore:
    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}

    def try_acquire(self, chat_id: str) -> bool:
        """Claim a chat for one in-flight turn."""
        with self._lock:
            if chat_id in self._inflight:
                return False
            self._inflight[chat_id] = threading.Event()
            return True

    def release(self, chat_id: str) -> None:
        with self._lock:
            self._inflight.pop(chat_id, None)

    def stop_event(self, chat_id: str) -> threading.Event | None:
        with self._lock:
            return self._inflight.get(chat_id)

    def request_stop(self, chat_id: str) -> bool:
        event = self.stop_event(chat_id)
        if event is None:
            return False
        event.set()
        return True

    def _path(self, chat_id: str) -> Path:
        return self.dir / f"{chat_id}.json"

    def _read(self, chat_id: str) -> dict | None:
        path = self._path(chat_id)
        return json.loads(path.read_text()) if path.exists() else None

    def list_all(self) -> list[dict]:
        out = []
        for path in self.dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                out.append(
                    {
                        "id": data["id"],
                        "name": data.get("name") or "New chat",
                        "updated_at": data.get("updated_at", ""),
                    }
                )
            except Exception:
                continue
        out.sort(key=lambda row: row["updated_at"], reverse=True)
        return out

    def get(self, chat_id: str) -> dict | None:
        return self._read(chat_id)

    def create(self) -> dict:
        chat = {
            "id": uuid.uuid4().hex[:12],
            "name": "",
            "created_at": _now(),
            "updated_at": _now(),
            "turns": [],
        }
        self._write(chat)
        return chat

    def append_turn(self, chat_id: str, turn: StoredTurn, name: str | None = None) -> dict:
        """Append a completed exchange."""
        with self._lock:
            chat = self._read(chat_id) or {
                "id": chat_id,
                "name": "",
                "created_at": _now(),
                "turns": [],
            }
            chat["turns"].append(turn)
            if name is not None:
                chat["name"] = name
            chat["updated_at"] = _now()
            self._write(chat)
            return chat

    def citation_count(self, chat_id: str) -> int:
        chat = self.get(chat_id)
        return sum(len(turn["citations"]) for turn in chat["turns"]) if chat else 0

    def delete(self, chat_id: str) -> None:
        path = self._path(chat_id)
        if path.exists():
            path.unlink()

    def set_feedback(
        self, chat_id: str, turn_index: int, vote: str | None, note: str | None
    ) -> dict | None:
        with self._lock:
            chat = self._read(chat_id)
            if chat is None:
                return None
            if not (0 <= turn_index < len(chat["turns"])):
                raise ValueError("turn_index out of range")
            if vote not in ("up", "down", None):
                raise ValueError("vote must be 'up', 'down', or None")
            if note is not None and vote is None:
                raise ValueError("note requires a vote")
            chat["turns"][turn_index]["feedback"] = (
                {"vote": vote, "note": note, "updated_at": _now()} if vote else None
            )
            chat["updated_at"] = _now()
            self._write(chat)
            return chat

    def truncate_before(self, chat_id: str, turn_index: int) -> dict | None:
        """Drop this turn and its successors before an edit-and-resend."""
        with self._lock:
            chat = self._read(chat_id)
            if chat is None:
                return None
            if not (0 <= turn_index < len(chat["turns"])):
                raise ValueError("turn_index out of range")
            chat["turns"] = chat["turns"][:turn_index]
            chat["updated_at"] = _now()
            self._write(chat)
            return chat

    def _write(self, chat: dict) -> None:
        path = self._path(chat["id"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(chat, indent=2))
        temporary.replace(path)


def generate_name(first_user_msg: str, spec: LLMSpec) -> str:
    """Ask the LLM for a short session title. Best-effort; falls back gracefully."""
    prompt = (
        "Give a concise 3-6 word title (no quotes, no trailing punctuation) for a "
        f"conversation that begins with:\n\n{first_user_msg}\n\nReply with only the title."
    )
    try:
        raw = build_llm(spec).complete(
            system="You write concise chat titles.", user=prompt, max_tokens=256
        )
        title = raw.strip().strip('"').splitlines()[0].strip()
        return title[:60] or "New chat"
    except Exception:
        return "New chat"
