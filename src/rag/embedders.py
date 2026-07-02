"""Pluggable embedding backends for indexing.

Two backends, registered under a config ``type`` string via ``@register_embedder``:

* ``hf``     — any sentence-transformers / HuggingFace model, run locally
               (default; uses Apple MPS when available).
* ``openai`` — any OpenAI-compatible embeddings endpoint (OpenAI itself, or
               a local/other server via ``--api-base``).

Both expose the Chroma ``EmbeddingFunction`` protocol: ``__call__(input) -> list[list[float]]``.
Add a backend by dropping a ``@register_embedder("name")`` class here — ``build_embedder``
discovers it via the registry, no other wiring needed.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Chroma-compatible embedding function.

    Concrete embedders register under a config ``type`` string and expose a
    uniform ``build`` classmethod so ``build_embedder`` can construct any of
    them from the same config fields (ignoring the ones a given backend doesn't use).
    """

    @abstractmethod
    def name(self) -> str:
        """Stable id Chroma uses to namespace/validate the collection."""

    @abstractmethod
    def __call__(self, input: list[str]) -> list[list[float]]: ...

    @classmethod
    @abstractmethod
    def build(
        cls,
        model_name: str,
        *,
        batch_size: int,
        api_base: str | None,
        api_key_env: str,
        max_seq_length: int,
    ) -> Embedder:
        """Construct from the shared config fields."""


_EMBEDDERS: dict[str, type[Embedder]] = {}


def register_embedder(name: str):
    """Register an :class:`Embedder` subclass under a config ``type`` string."""

    def deco(cls: type[Embedder]) -> type[Embedder]:
        _EMBEDDERS[name] = cls
        return cls

    return deco


@register_embedder("hf")
class HFEmbedder(Embedder):
    """Local sentence-transformers embedder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int = 1024,
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
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        # Cap sequence length: models like bge-m3 default to 8192, which at a
        # normal batch size overflows Metal's 2**32-byte per-tensor limit on MPS.
        # Our chunks are <=~750 tokens, so 1024 is headroom without truncation.
        cur = self.model.max_seq_length or max_seq_length
        self.model.max_seq_length = min(cur, max_seq_length)

    def name(self) -> str:  # Chroma calls this to namespace/validate the collection
        return f"hf:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            input,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine-ready
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    @classmethod
    def build(
        cls,
        model_name: str,
        *,
        batch_size: int,
        api_base: str | None,
        api_key_env: str,
        max_seq_length: int,
    ) -> HFEmbedder:
        return cls(model_name, batch_size=batch_size, max_seq_length=max_seq_length)


@register_embedder("openai")
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

    @classmethod
    def build(
        cls,
        model_name: str,
        *,
        batch_size: int,
        api_base: str | None,
        api_key_env: str,
        max_seq_length: int,
    ) -> OpenAIEmbedder:
        return cls(model_name, api_base=api_base, api_key_env=api_key_env, batch_size=batch_size)


def build_embedder(
    embedder: str,
    embedder_type: str,
    *,
    batch_size: int,
    api_base: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    max_seq_length: int = 1024,
) -> Embedder:
    """Factory used by the CLI/config: dispatch to the registered embedder type."""
    try:
        cls = _EMBEDDERS[embedder_type]
    except KeyError:
        raise ValueError(
            f"Unknown embedder-type: {embedder_type!r} (expected one of {sorted(_EMBEDDERS)})"
        ) from None
    return cls.build(
        embedder,
        batch_size=batch_size,
        api_base=api_base,
        api_key_env=api_key_env,
        max_seq_length=max_seq_length,
    )
