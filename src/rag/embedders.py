"""Pluggable embedding backends for indexing.

Two backends, selected by `--embedder-type`:

* ``hf``     — any sentence-transformers / HuggingFace model, run locally
               (default; uses Apple MPS when available).
* ``openai`` — any OpenAI-compatible embeddings endpoint (OpenAI itself, or
               a local/other server via ``--api-base``).

Both expose the Chroma ``EmbeddingFunction`` protocol: ``__call__(input) -> list[list[float]]``.
"""

from __future__ import annotations

import os


class HFEmbedder:
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


class OpenAIEmbedder:
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


def build_embedder(
    embedder: str,
    embedder_type: str,
    *,
    batch_size: int,
    api_base: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    max_seq_length: int = 1024,
):
    """Factory used by the CLI."""
    if embedder_type == "hf":
        return HFEmbedder(embedder, batch_size=batch_size, max_seq_length=max_seq_length)
    if embedder_type == "openai":
        return OpenAIEmbedder(
            embedder, api_base=api_base, api_key_env=api_key_env, batch_size=batch_size
        )
    raise ValueError(f"Unknown embedder-type: {embedder_type!r} (expected 'hf' or 'openai')")
