"""PaperLens RAG core: config-driven ingestion + two-stage retrieval.

Import the public API from this package rather than the leaf modules
(e.g. ``from rag import Searcher, load_config``); the module layout below is an
internal detail and may change.

Module layering — imports flow one way (top -> bottom); there are no cycles::

    config  chunking  extract  manifest  sparse       (leaves: no intra-rag deps)
       |        |         |         |        |
    embedders(config)   llm(config)   index(chunking, embedders)   reranker(config, llm)
       |                                              |
    tagger(llm)   query_expansion(llm)   search(embedders, reranker, sparse, query_expansion)
                              |                        |
                         pipeline(extract, index, manifest, tagger)
                              |
                         ingest(pipeline, index, manifest, tagger)

``faithfulness(config)`` is a sibling leaf-plus-config module like ``embedders``/
``llm`` (depends only on ``config``), but isn't part of the retrieval flow above —
it's composed directly by ``server.agent``, not by ``search``/``pipeline``.

The embedder/reranker/llm backends are selected by ``draccus.ChoiceRegistry``
config variants (``embedding.type`` / ``reranker.type`` / ``llm.*.type``); the
``build_*`` functions match on the variant. ``server`` composes ``rag``; ``rag``
never imports ``server``.
"""

from __future__ import annotations

from .chunking import Chunk, chunk_markdown
from .config import (
    AnthropicSpec,
    BM25Cfg,
    ChunkingCfg,
    Config,
    EmbeddingCfg,
    FaithfulnessCfg,
    GeminiEmbeddingCfg,
    GeminiSpec,
    HFEmbeddingCfg,
    HFFaithfulnessCfg,
    HFRerankerCfg,
    IngestConfig,
    LLMCfg,
    LLMRerankerCfg,
    LLMSpec,
    MultiQueryCfg,
    OllamaEmbeddingCfg,
    OpenAIEmbeddingCfg,
    OpenAISpec,
    Paper,
    RerankerCfg,
    SGLangSpec,
    SparseCfg,
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
from .faithfulness import (
    FaithfulnessChecker,
    HFFaithfulnessChecker,
    Verdict,
    best_support,
    build_faithfulness_checker,
)
from .index import index_markdown, open_collection
from .llm import LLMBackend, build_llm
from .manifest import Manifest
from .pipeline import build_embedder_from_config, ingest_paper, pending_papers
from .query_expansion import generate_paraphrases
from .reranker import Reranker, build_reranker
from .search import Result, Searcher
from .sparse import BM25Index, build_sparse_index, reciprocal_rank_fusion, rrf_scores
from .tagger import generate_tags

__all__ = [
    "AnthropicSpec",
    "BM25Cfg",
    "BM25Index",
    "Chunk",
    "ChunkingCfg",
    "Config",
    "Embedder",
    "EmbeddingCfg",
    "FaithfulnessCfg",
    "FaithfulnessChecker",
    "GeminiEmbedder",
    "GeminiEmbeddingCfg",
    "GeminiSpec",
    "HFEmbedder",
    "HFEmbeddingCfg",
    "HFFaithfulnessCfg",
    "HFFaithfulnessChecker",
    "HFRerankerCfg",
    "IngestConfig",
    "LLMBackend",
    "LLMCfg",
    "LLMRerankerCfg",
    "LLMSpec",
    "Manifest",
    "MultiQueryCfg",
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
    "SparseCfg",
    "VLLMSpec",
    "Verdict",
    "best_support",
    "build_embedder",
    "build_embedder_from_config",
    "build_faithfulness_checker",
    "build_llm",
    "build_reranker",
    "build_sparse_index",
    "chunk_markdown",
    "generate_paraphrases",
    "generate_tags",
    "index_markdown",
    "ingest_paper",
    "load_config",
    "open_collection",
    "parse_config",
    "pending_papers",
    "reciprocal_rank_fusion",
    "rrf_scores",
]
