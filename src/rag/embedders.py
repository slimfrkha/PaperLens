"""Pluggable embedding backends for indexing.

One backend per config ``embedding.type`` variant (see ``EmbeddingCfg`` in
``config.py``):

* ``hf``     — any sentence-transformers / HuggingFace model, run locally
               (default; uses Apple MPS when available).
* ``openai`` — any OpenAI-compatible embeddings endpoint (OpenAI itself, or
               a local/other server via ``api_base``).
* ``gemini`` — Google GenAI embeddings (asymmetric: document vs query task type).
* ``ollama`` — Ollama's native ``/api/embed`` endpoint.

All expose the Chroma ``EmbeddingFunction`` protocol: ``__call__(input) -> list[list[float]]``.
Add a backend by registering an ``EmbeddingCfg`` variant in ``config.py`` and adding
a match arm to ``build_embedder`` below.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .config import (
    EmbeddingCfg,
    GeminiEmbeddingCfg,
    HFEmbeddingCfg,
    OllamaEmbeddingCfg,
    OpenAIEmbeddingCfg,
)


class Embedder(ABC):
    """Chroma-compatible embedding function.

    ``build_embedder`` constructs the right concrete embedder from the matching
    ``EmbeddingCfg`` variant.
    """

    @abstractmethod
    def name(self) -> str:
        """Stable id Chroma uses to namespace/validate the collection."""

    @abstractmethod
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed documents (index-time)."""

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed search queries. Symmetric models reuse ``__call__``; asymmetric
        ones (e.g. Gemini/Cohere/Voyage, which want a query-vs-document hint)
        override this. Kept off the Chroma protocol, so ``Searcher`` calls it
        only when present."""
        return self(input)


class HFEmbedder(Embedder):
    """Local sentence-transformers embedder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int = 1024,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        from sentence_transformers import SentenceTransformer

        if device is None:
            try:
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        # Cap sequence length: models like bge-m3 default to 8192, which at a
        # normal batch size overflows Metal's 2**32-byte per-tensor limit on MPS.
        # Our chunks are <=~750 tokens, so 1024 is headroom without truncation.
        cur = self.model.max_seq_length or max_seq_length
        self.model.max_seq_length = min(cur, max_seq_length)

    def name(self) -> str:  # Chroma calls this to namespace/validate the collection
        return f"hf:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        texts = [self.document_prefix + t for t in input] if self.document_prefix else input
        return self._encode(texts)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        texts = [self.query_prefix + t for t in input] if self.query_prefix else input
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine-ready
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()


class OpenAIEmbedder(Embedder):
    """OpenAI-compatible API embedder (OpenAI, or any compatible base_url)."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        batch_size: int = 128,
    ):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env}. "
                f"Export it or pass --api-key-env pointing at the right variable."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def name(self) -> str:
        return f"openai:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


class GeminiEmbedder(Embedder):
    """Google GenAI embeddings (e.g. ``text-embedding-004``, ``gemini-embedding-001``).

    Asymmetric: documents are embedded with ``RETRIEVAL_DOCUMENT`` and queries with
    ``RETRIEVAL_QUERY`` (via :meth:`embed_query`), which the model uses to place a
    query near the passages that answer it.
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "GEMINI_API_KEY",
        batch_size: int = 100,
    ):
        from google import genai

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env}. Export it (or add it to .env)."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.client = genai.Client(api_key=api_key)

    def name(self) -> str:
        return f"gemini:{self.model_name}"

    def _embed(self, input: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,  # ty: ignore[invalid-argument-type]  # SDK accepts list[str] at runtime; stub union omits it
                config=types.EmbedContentConfig(task_type=task_type),
            )
            out.extend(e.values for e in (resp.embeddings or []) if e.values)
        return out

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, "RETRIEVAL_DOCUMENT")

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, "RETRIEVAL_QUERY")


class OllamaEmbedder(Embedder):
    """Ollama's native ``/api/embed`` endpoint (batched). Needs no API key."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        batch_size: int = 64,
    ):
        import httpx

        self.model_name = model_name
        self.api_base = (api_base or "http://localhost:11434").rstrip("/")
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=120)

    def name(self) -> str:
        return f"ollama:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = self.client.post(
                f"{self.api_base}/api/embed",
                json={"model": self.model_name, "input": batch},
            )
            resp.raise_for_status()
            out.extend(resp.json()["embeddings"])
        return out


def build_embedder(cfg: EmbeddingCfg) -> Embedder:
    """Construct the embedder described by ``config.embedding`` (its variant)."""
    match cfg:
        case HFEmbeddingCfg():
            return HFEmbedder(
                cfg.model,
                batch_size=cfg.batch_size,
                max_seq_length=cfg.max_seq_length,
                query_prefix=cfg.query_prefix,
                document_prefix=cfg.document_prefix,
            )
        case OpenAIEmbeddingCfg():
            return OpenAIEmbedder(
                cfg.model,
                api_base=cfg.api_base or None,
                api_key_env=cfg.api_key_env,
                batch_size=cfg.batch_size,
            )
        case GeminiEmbeddingCfg():
            return GeminiEmbedder(cfg.model, api_key_env=cfg.api_key_env, batch_size=cfg.batch_size)
        case OllamaEmbeddingCfg():
            return OllamaEmbedder(
                cfg.model, api_base=cfg.api_base or None, batch_size=cfg.batch_size
            )
        case _:
            raise ValueError(f"Unknown embedding config: {type(cfg).__name__}")
