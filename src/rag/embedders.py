"""Pluggable embedding backends for indexing.

One backend per config ``embedding.type`` variant (see ``EmbeddingCfg`` in
``config.py``):

* ``hf``     — any sentence-transformers / HuggingFace model, run locally
               (default; uses Apple MPS when available).
* ``openai`` — any OpenAI-compatible embeddings endpoint (OpenAI itself, or
               a local/other server via ``api_base``), via LiteLLM.
* ``gemini`` — Google GenAI embeddings (asymmetric: document vs query task type),
               via LiteLLM.
* ``voyage`` — Voyage AI embeddings (asymmetric), Anthropic's recommended
               embedding partner — Anthropic has no embeddings API of its own.
               Called directly, not via LiteLLM: LiteLLM's Voyage integration
               doesn't forward the `input_type` param, which would silently
               drop the asymmetric query/document distinction.
* ``ollama`` — Ollama's native ``/api/embed`` endpoint, via LiteLLM.

All expose the Chroma ``EmbeddingFunction`` protocol: ``__call__(input) -> list[list[float]]``.
Add a backend by registering an ``EmbeddingCfg`` variant in ``config.py`` and adding
a match arm to ``build_embedder`` below.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import litellm

from .config import (
    EmbeddingCfg,
    GeminiEmbeddingCfg,
    HFEmbeddingCfg,
    OllamaEmbeddingCfg,
    OpenAIEmbeddingCfg,
    VoyageEmbeddingCfg,
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
        ones (e.g. Gemini/Voyage, which want a query-vs-document hint)
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
    """OpenAI-compatible API embedder (OpenAI, or any compatible base_url), via LiteLLM."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        batch_size: int = 128,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env}. "
                f"Export it or pass --api-key-env pointing at the right variable."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self._kwargs: dict = {"model": f"openai/{model_name}", "api_key": api_key}
        if api_base:
            self._kwargs["api_base"] = api_base
            self._kwargs["custom_llm_provider"] = "openai"

    def name(self) -> str:
        return f"openai:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = litellm.embedding(input=batch, **self._kwargs)
            out.extend(d["embedding"] for d in resp.data)
        return out


class GeminiEmbedder(Embedder):
    """Google GenAI embeddings (e.g. ``text-embedding-004``, ``gemini-embedding-001``), via LiteLLM.

    Asymmetric: documents are embedded with ``RETRIEVAL_DOCUMENT`` and queries with
    ``RETRIEVAL_QUERY`` (via :meth:`embed_query`), which the model uses to place a
    query near the passages that answer it. LiteLLM forwards this via its
    provider-agnostic `input_type` param (Vertex/Gemini's `map_openai_params` reads
    it straight off the raw call kwargs, unlike most providers' `non_default_params`
    path — verified against the installed litellm's source).
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "GEMINI_API_KEY",
        batch_size: int = 100,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env}. Export it (or add it to .env)."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self._kwargs: dict = {"model": f"gemini/{model_name}", "api_key": api_key}

    def name(self) -> str:
        return f"gemini:{self.model_name}"

    def _embed(self, input: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = litellm.embedding(input=batch, input_type=task_type, **self._kwargs)
            out.extend(d["embedding"] for d in resp.data)
        return out

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, "RETRIEVAL_DOCUMENT")

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, "RETRIEVAL_QUERY")


class VoyageEmbedder(Embedder):
    """Voyage AI embeddings (e.g. ``voyage-3.5``) — Anthropic's recommended embedding
    partner; Anthropic has no embeddings API of its own.

    Asymmetric like Gemini's, via Voyage's own `input_type: "query" | "document"`.
    Called directly over HTTP rather than through LiteLLM: LiteLLM's
    `VoyageEmbeddingConfig.map_openai_params` doesn't read `input_type` at all (unlike
    Gemini's, it only receives the pre-filtered `non_default_params`, and `input_type`
    isn't an OpenAI-standard embedding param — verified against the installed
    litellm's source), which would silently drop the asymmetric distinction.
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "VOYAGE_API_KEY",
        batch_size: int = 100,
    ):
        import httpx

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env}. Export it (or add it to .env)."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.client = httpx.Client(
            base_url="https://api.voyageai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    def name(self) -> str:
        return f"voyage:{self.model_name}"

    def _embed(self, input: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = self.client.post(
                "/embeddings",
                json={"model": self.model_name, "input": batch, "input_type": input_type},
            )
            resp.raise_for_status()
            out.extend(d["embedding"] for d in resp.json()["data"])
        return out

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, "document")

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, "query")


class OllamaEmbedder(Embedder):
    """Ollama's native ``/api/embed`` endpoint (batched), via LiteLLM. Needs no API key."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.api_base = (api_base or "http://localhost:11434").rstrip("/")
        self.batch_size = batch_size

    def name(self) -> str:
        return f"ollama:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            resp = litellm.embedding(
                model=f"ollama/{self.model_name}", input=batch, api_base=self.api_base
            )
            out.extend(d["embedding"] for d in resp.data)
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
        case VoyageEmbeddingCfg():
            return VoyageEmbedder(cfg.model, api_key_env=cfg.api_key_env, batch_size=cfg.batch_size)
        case OllamaEmbeddingCfg():
            return OllamaEmbedder(
                cfg.model, api_base=cfg.api_base or None, batch_size=cfg.batch_size
            )
        case _:
            raise ValueError(f"Unknown embedding config: {type(cfg).__name__}")
