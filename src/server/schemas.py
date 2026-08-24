"""Request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    tags: list[str] = []
    papers: list[str] = []  # restrict search to these paper_ids (empty = all)
    per_paper: bool = False  # recall once per paper, pool flat, instead of once over the scope
    compare: bool = False  # Compare mode: guaranteed per-paper search+answer, then synthesized
    # Auto mode was selected client-side and resolved (ask/compare) via /api/chat/classify
    # before this request was sent — badge-only, never re-classified server-side: `compare`
    # above already carries the resolved dispatch decision, this field exists purely so
    # persistence/reload can show "Auto -> ..." instead of looking like the user picked the
    # mode manually.
    auto: bool = False
    chat_id: str | None = None
    edit_index: int | None = None  # if set, truncate stored history to this index first


class ClassifyModeRequest(BaseModel):
    messages: list[ChatMessage]
    tags: list[str] = []
    papers: list[str] = []


class ClassifyModeResponse(BaseModel):
    mode: Literal["ask", "compare"]
    scope_size: int


class FeedbackRequest(BaseModel):
    index: int  # position in the session's messages/citations/traces arrays
    vote: Literal["up", "down"] | None = None
    note: str | None = None


class AnnotationCreate(BaseModel):
    snippet: str
    section_title: str = ""
    section_slug: str = ""
    note: str = ""


class AnnotationUpdate(BaseModel):
    note: str


class AddPapersRequest(BaseModel):
    arxiv_ids_or_urls: list[str]
