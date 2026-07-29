"""Embedder registry + the two new API backends (offline: no server/network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config import EmbeddingCfg, HFEmbeddingCfg, OllamaEmbeddingCfg
from rag.embedders import build_embedder


def test_build_embedder_unknown_variant_raises():
    # The base EmbeddingCfg is not a registered variant.
    with pytest.raises(ValueError, match="Unknown embedding config"):
        build_embedder(EmbeddingCfg())


# ---- HF: asymmetric query/document prefixing, faked SentenceTransformer ----


class _FakeSentenceTransformer:
    def __init__(self, model_name, device=None, trust_remote_code=None):
        self.max_seq_length = 8192
        self.seen: list[list[str]] = []

    def encode(self, input, **kwargs):
        self.seen.append(list(input))
        import numpy as np

        return np.zeros((len(input), 2))


def test_hf_embedder_prefixes_are_asymmetric(monkeypatch):
    st = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(st, "SentenceTransformer", _FakeSentenceTransformer)

    e = build_embedder(
        HFEmbeddingCfg(model="fake-model", query_prefix="query: ", document_prefix="passage: ")
    )
    e(["doc"])
    e.embed_query(["q"])

    assert e.model.seen == [["passage: doc"], ["query: q"]]


def test_hf_embedder_empty_prefix_is_noop(monkeypatch):
    st = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(st, "SentenceTransformer", _FakeSentenceTransformer)

    e = build_embedder(HFEmbeddingCfg(model="fake-model"))
    e(["doc"])
    e.embed_query(["q"])

    assert e.model.seen == [["doc"], ["q"]]


# ---- Ollama: native /api/embed, faked httpx client -------------------------


class _FakeResp:
    def __init__(self, n: int):
        self._n = n

    def raise_for_status(self):
        pass

    def json(self):
        return {"embeddings": [[0.1, 0.2]] * self._n}


class _FakeHTTP:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json):
        self.calls.append((url, json))
        return _FakeResp(len(json["input"]))


def test_ollama_embedder_batches_and_parses():
    e = build_embedder(
        OllamaEmbeddingCfg(model="nomic-embed-text", batch_size=64, api_base="http://h:11434/")
    )
    e.client = _FakeHTTP()

    out = e(["x", "y"])

    assert out == [[0.1, 0.2], [0.1, 0.2]]
    assert e.name() == "ollama:nomic-embed-text"
    url, payload = e.client.calls[0]
    assert url == "http://h:11434/api/embed"  # trailing slash stripped
    assert payload == {"model": "nomic-embed-text", "input": ["x", "y"]}


# ---- Gemini: asymmetric document vs query task type ------------------------


def test_gemini_embedder_is_asymmetric(monkeypatch):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    seen: list[str] = []

    class FakeModels:
        def embed_content(self, model, contents, config):
            seen.append(config.task_type)
            embs = [SimpleNamespace(values=[1.0, 2.0]) for _ in contents]
            return SimpleNamespace(embeddings=embs)

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)

    from rag.embedders import GeminiEmbedder

    e = GeminiEmbedder("text-embedding-004")
    docs = e(["a", "b"])
    query = e.embed_query(["c"])

    assert len(docs) == 2 and len(query) == 1
    assert seen == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert e.name() == "gemini:text-embedding-004"
