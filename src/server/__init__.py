"""PaperLens server: FastAPI backend + in-process ingestion worker.

Composes the ``rag`` core (retrieval, ingestion pipeline, LLM) behind an HTTP API.
Import the public API from this package (e.g. ``from server import create_app``)
rather than the leaf modules. ``server`` depends on ``rag``, never the reverse.
"""

from __future__ import annotations

from .agent import ChatAgent
from .chats import ChatStore, generate_name
from .main import create_app, main
from .schemas import ChatMessage, ChatRequest
from .worker import IngestionWorker

__all__ = [
    "ChatAgent",
    "ChatMessage",
    "ChatRequest",
    "ChatStore",
    "IngestionWorker",
    "create_app",
    "generate_name",
    "main",
]
