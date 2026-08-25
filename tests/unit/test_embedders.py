"""Embedder registry + the API-backed embedders (offline: no server/network)."""

from __future__ import annotations

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


# ---- OpenAI / Gemini / Ollama: fake litellm.embedding() --------------------


def _embedding_response(n: int):
    from types import SimpleNamespace

    return SimpleNamespace(data=[{"embedding": [0.1, 0.2], "index": i} for i in range(n)])


class _FakeLiteLLMEmbedding:
    """Records each litellm.embedding() call, returns a fixed-shape response."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _embedding_response(len(kwargs["input"]))


def test_openai_embedder_calls_litellm_with_model_prefix(monkeypatch):
    fake = _FakeLiteLLMEmbedding()
    monkeypatch.setattr("rag.embedders.litellm.embedding", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    from rag.config import OpenAIEmbeddingCfg

    e = build_embedder(OpenAIEmbeddingCfg(model="text-embedding-3-small"))
    out = e(["a", "b"])

    assert out == [[0.1, 0.2], [0.1, 0.2]]
    assert e.name() == "openai:text-embedding-3-small"
    assert fake.calls[0]["model"] == "openai/text-embedding-3-small"
    assert fake.calls[0]["api_key"] == "k"
    assert "api_base" not in fake.calls[0]


def test_openai_embedder_local_server_sets_api_base_and_custom_provider(monkeypatch):
    fake = _FakeLiteLLMEmbedding()
    monkeypatch.setattr("rag.embedders.litellm.embedding", fake)
    monkeypatch.setenv("UNUSED_LOCAL_KEY", "local-no-key")

    from rag.config import OpenAIEmbeddingCfg

    e = build_embedder(
        OpenAIEmbeddingCfg(
            model="nomic-embed-text",
            api_base="http://localhost:1234/v1",
            api_key_env="UNUSED_LOCAL_KEY",
        )
    )
    e(["a"])

    assert fake.calls[0]["api_base"] == "http://localhost:1234/v1"
    assert fake.calls[0]["custom_llm_provider"] == "openai"


def test_gemini_embedder_is_asymmetric(monkeypatch):
    fake = _FakeLiteLLMEmbedding()
    monkeypatch.setattr("rag.embedders.litellm.embedding", fake)
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    from rag.embedders import GeminiEmbedder

    e = GeminiEmbedder("text-embedding-004")
    docs = e(["a", "b"])
    query = e.embed_query(["c"])

    assert len(docs) == 2 and len(query) == 1
    assert [c["input_type"] for c in fake.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert fake.calls[0]["model"] == "gemini/text-embedding-004"
    assert e.name() == "gemini:text-embedding-004"


def test_ollama_embedder_batches_via_litellm(monkeypatch):
    fake = _FakeLiteLLMEmbedding()
    monkeypatch.setattr("rag.embedders.litellm.embedding", fake)

    e = build_embedder(
        OllamaEmbeddingCfg(model="nomic-embed-text", batch_size=1, api_base="http://h:11434")
    )
    out = e(["x", "y"])

    assert out == [[0.1, 0.2], [0.1, 0.2]]
    assert e.name() == "ollama:nomic-embed-text"
    assert len(fake.calls) == 2  # batch_size=1 -> one call per input
    assert fake.calls[0]["model"] == "ollama/nomic-embed-text"
    assert fake.calls[0]["api_base"] == "http://h:11434"


def test_ollama_embedder_strips_trailing_slash_from_api_base(monkeypatch):
    # litellm's ollama embedding handler does raw string concatenation
    # (api_base += "/api/embed") with no normalization of its own — a trailing
    # slash here would produce a malformed double-slash URL.
    fake = _FakeLiteLLMEmbedding()
    monkeypatch.setattr("rag.embedders.litellm.embedding", fake)

    e = build_embedder(OllamaEmbeddingCfg(model="nomic-embed-text", api_base="http://h:11434/"))
    e(["x"])

    assert fake.calls[0]["api_base"] == "http://h:11434"


# ---- Voyage: asymmetric, called directly (not via litellm) -----------------


class _FakeVoyageResp:
    def __init__(self, n: int):
        self._n = n

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"embedding": [0.3, 0.4], "index": i} for i in range(self._n)]}


class _FakeVoyageHTTP:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def post(self, path, json):
        self.calls.append((path, json))
        return _FakeVoyageResp(len(json["input"]))


def test_voyage_embedder_is_asymmetric_and_bypasses_litellm(monkeypatch):
    # Assert litellm.embedding is never called for Voyage — it doesn't forward
    # input_type, which would silently drop the asymmetric distinction.
    def _boom(**kwargs):
        raise AssertionError("VoyageEmbedder must not go through litellm.embedding")

    monkeypatch.setattr("rag.embedders.litellm.embedding", _boom)
    monkeypatch.setenv("VOYAGE_API_KEY", "k")

    from rag.embedders import VoyageEmbedder

    e = VoyageEmbedder("voyage-3.5")
    e.client = _FakeVoyageHTTP()

    docs = e(["a", "b"])
    query = e.embed_query(["c"])

    assert docs == [[0.3, 0.4], [0.3, 0.4]]
    assert query == [[0.3, 0.4]]
    assert e.name() == "voyage:voyage-3.5"
    doc_path, doc_payload = e.client.calls[0]
    query_path, query_payload = e.client.calls[1]
    assert doc_path == query_path == "/embeddings"
    assert doc_payload["input_type"] == "document"
    assert query_payload["input_type"] == "query"


def test_voyage_embedder_missing_key_raises(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    from rag.embedders import VoyageEmbedder

    with pytest.raises(RuntimeError, match="No API key"):
        VoyageEmbedder("voyage-3.5")
