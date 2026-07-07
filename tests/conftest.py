"""Shared fixtures and offline seams.

Everything here runs without the network or the local embedding/reranking models:

* ``FakeEmbedder`` — a deterministic bag-of-words embedder that satisfies the
  Chroma embedding-function protocol, so retrieval is exercised against a real
  temp Chroma DB without downloading ``bge-m3``.
* ``FakeLLM`` — an :class:`rag.llm.LLMBackend` that records calls and replays a
  scripted answer / tool call, so the chat agent and tagger run with no API.

The ``make_*`` fixtures are factories: each test overrides only what it cares
about and gets a fresh, temp-backed object.
"""

from __future__ import annotations

import hashlib

import pytest

from rag.config import (
    AnthropicSpec,
    Config,
    HFEmbeddingCfg,
    HFRerankerCfg,
    LLMCfg,
    LLMSpec,
    OpenAISpec,
    Paths,
)
from rag.index import open_collection
from rag.llm import LLMBackend

_EMBED_DIM = 32


class FakeEmbedder:
    """Deterministic bag-of-words embedder (no model, no network).

    Words hash into fixed dimensions; a constant bias dim keeps every vector
    non-zero so cosine distance is always defined. Texts that share words land
    closer together, which is enough to assert retrieval ordering.
    """

    def __init__(self, name: str = "fake") -> None:
        self._name = name

    def name(self) -> str:  # Chroma namespaces the collection by this
        return f"fake:{self._name}"

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * _EMBED_DIM
        vec[-1] = 1.0  # bias dim -> never a zero vector
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % (_EMBED_DIM - 1)] += 1.0
        return vec

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in input]


class FakeLLM(LLMBackend):
    """Scripted, call-recording LLM backend.

    ``complete`` returns ``answer``. ``run_tools`` optionally emits reasoning,
    executes each scripted ``(tool_name, args)`` call (driving the real tool),
    then streams ``answer`` through ``on_text``.
    """

    def __init__(
        self,
        spec: LLMSpec | None = None,
        answer: str = "ok",
        tool_calls: list[tuple[str, dict]] | None = None,
        reasoning: str | None = None,
    ) -> None:
        super().__init__(spec or AnthropicSpec())
        self.answer = answer
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning
        self.complete_calls: list[dict] = []
        self.run_tools_calls: list[dict] = []
        self.executed: list[tuple[str, dict]] = []

    def complete(self, system, user, max_tokens=None):
        self.complete_calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.answer

    def run_tools(
        self, system, messages, tools, execute, on_text=None, on_reasoning=None, max_rounds=8
    ):
        self.run_tools_calls.append({"system": system, "messages": messages, "tools": tools})
        if self.reasoning and on_reasoning:
            on_reasoning(self.reasoning)
        for name, args in self.tool_calls:
            self.executed.append((name, args))
            execute(name, args)
        if on_text:
            on_text(self.answer)
        return self.answer


@pytest.fixture
def make_config(tmp_path):
    """Factory for a :class:`Config` with all runtime paths under ``tmp_path``."""

    def _make(**overrides) -> Config:
        paths = Paths(
            rag_db=str(tmp_path / "rag_db"),
            pdf_dir=str(tmp_path / "pdf"),
            markdown_dir=str(tmp_path / "md"),
            chat_history=str(tmp_path / "chats"),
            web_dist=str(tmp_path / "web_dist"),
        )
        cfg = Config(
            paths=paths,
            collection="test_papers",
            embedding=HFEmbeddingCfg(),
            reranker=HFRerankerCfg(enabled=False),
            llm=LLMCfg(
                tagging=OpenAISpec(api_base="http://x"),
                chat=OpenAISpec(api_base="http://x"),
            ),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    return _make


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    """The FakeLLM class, so tests can script an answer / tool calls per case."""
    return FakeLLM


@pytest.fixture
def make_searcher(make_config, fake_embedder):
    """Factory: a temp Chroma collection seeded with ``docs`` + a Searcher over it.

    ``docs`` is a list of ``(id, text, metadata)``; metadata must carry the
    ``paper_id`` / ``breadcrumb`` / ``section_title`` keys Searcher reads back.
    """
    from types import SimpleNamespace

    from rag.chunking import Chunk
    from rag.search import Searcher

    def _make(docs):
        cfg = make_config()
        collection = open_collection(
            cfg.paths.rag_db, cfg.collection, embedder_name=fake_embedder.name()
        )
        chunks = [
            Chunk(text=text, body=text.split("\n\n", 1)[-1], metadata=meta)
            for _, text, meta in docs
        ]
        # upsert_chunks recomputes ids from metadata; override with explicit ids.
        collection.upsert(
            ids=[doc_id for doc_id, _, _ in docs],
            embeddings=fake_embedder([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        searcher = Searcher(
            db_dir=cfg.paths.rag_db,
            collection=cfg.collection,
            embedder=fake_embedder,
        )
        return SimpleNamespace(searcher=searcher, cfg=cfg, collection=collection)

    return _make


@pytest.fixture
def seed_chunks():
    """A helper to build the ``(id, text, metadata)`` tuples make_searcher wants."""

    def _make(paper_id: str, section: str, body: str, doc_id: str | None = None):
        breadcrumb = f"Paper > {section}"
        text = f"{breadcrumb}\n\n{body}"
        meta = {
            "paper_id": paper_id,
            "paper_title": "Paper",
            "section_number": "1",
            "section_title": section,
            "breadcrumb": breadcrumb,
            "body": body,
            "part": 0,
            "n_parts": 1,
        }
        return (doc_id or f"{paper_id}-{section}", text, meta)

    return _make
