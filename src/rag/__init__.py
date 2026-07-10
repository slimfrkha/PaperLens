"""PaperLens RAG core: config-driven ingestion + two-stage retrieval.

Import the public API from this package rather than the leaf modules
(e.g. ``from rag import Searcher, load_config``); the module layout below is an
internal detail and may change.

Module layering — imports flow one way (top -> bottom); there are no cycles::

    config  chunking  extract  manifest              (leaves: no intra-rag deps)
       |        |         |         |
    embedders(config)   llm(config)   index(chunking, embedders)   reranker(config, llm)
       |                                              |
    tagger(llm)   search(embedders, reranker)   pipeline(extract, index, manifest, tagger)
                              |
                         ingest(pipeline, index, manifest, tagger)

The embedder/reranker/llm backends are selected by ``draccus.ChoiceRegistry``
config variants (``embedding.type`` / ``reranker.type`` / ``llm.*.type``); the
``build_*`` functions match on the variant. ``server`` composes ``rag``; ``rag``
never imports ``server``.
"""

from __future__ import annotations

from .chunking import Chunk, chunk_markdown
from .config import (
    AnthropicSpec,
    ChunkingCfg,
    Config,
    EmbeddingCfg,
    GeminiEmbeddingCfg,
    GeminiSpec,
    HFEmbeddingCfg,
    HFRerankerCfg,
    IngestConfig,
    LLMCfg,
    LLMRerankerCfg,
    LLMSpec,
    OllamaEmbeddingCfg,
    OpenAIEmbeddingCfg,
    OpenAISpec,
    Paper,
    RerankerCfg,
    SGLangSpec,
    VLLMSpec,
    load_config,
    parse_config,
)
from .embedders import (
    Embedder,
    GeminiEmbedder,
    HFEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    build_embedder,
)
from .index import index_markdown, open_collection
from .llm import LLMBackend, build_llm
from .manifest import Manifest
from .pipeline import build_embedder_from_config, ingest_paper, pending_papers
from .reranker import Reranker, build_reranker
from .search import Result, Searcher
from .tagger import generate_tags

__all__ = [
    "AnthropicSpec",
    "Chunk",
    "ChunkingCfg",
    "Config",
    "Embedder",
    "EmbeddingCfg",
    "GeminiEmbedder",
    "GeminiEmbeddingCfg",
    "GeminiSpec",
    "HFEmbedder",
    "HFEmbeddingCfg",
    "HFRerankerCfg",
    "IngestConfig",
    "LLMBackend",
    "LLMCfg",
    "LLMRerankerCfg",
    "LLMSpec",
    "Manifest",
    "OllamaEmbedder",
    "OllamaEmbeddingCfg",
    "OpenAIEmbedder",
    "OpenAIEmbeddingCfg",
    "OpenAISpec",
    "Paper",
    "Reranker",
    "RerankerCfg",
    "Result",
    "SGLangSpec",
    "Searcher",
    "VLLMSpec",
    "build_embedder",
    "build_embedder_from_config",
    "build_llm",
    "build_reranker",
    "chunk_markdown",
    "generate_tags",
    "index_markdown",
    "ingest_paper",
    "load_config",
    "open_collection",
    "parse_config",
    "pending_papers",
]
