"""Request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    tags: list[str] = []
    papers: list[str] = []  # restrict search to these paper_ids (empty = all)
    chat_id: str | None = None
