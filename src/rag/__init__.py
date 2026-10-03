"""PaperLens RAG core: config-driven ingestion + two-stage retrieval.

Import the public API from this package rather than the leaf modules
(e.g. ``from rag import Searcher, load_config``); the module layout below is an
internal detail and may change.

Module layering — imports flow one way (top -> bottom); there are no cycles::

    config  chunking  extract  manifest  sparse  config_writer  cited_papers  (leaves)
       |        |         |         |        |         |
    embedders(config)   llm(config)   index(chunking, embedders)   reranker(config, llm)
       |                                              |
    tagger(llm)   query_expansion(llm)   search(embedders, reranker, sparse, query_expansion)
                              |                        |
                         pipeline(extract, index, manifest, tagger)
                              |
                         ingest(pipeline, manifest, tagger)

``faithfulness(config)`` and ``web_search(config)`` are sibling leaf-plus-config modules
like ``embedders``/``llm`` (each depends only on ``config``), but aren't part of the
retrieval flow above — they're composed directly by ``server.agent`` (faithfulness scores
citations; web_search backs the agent's out-of-scope ``web_search`` tool), not by
``search``/``pipeline``.

``config_writer`` is a leaf like ``sparse`` (no intra-rag deps — it round-trips
config.yaml directly, not through the ``Config`` dataclass), composed only by the
admin add/remove-paper routes in ``server.main``. Accessed as
``rag.config_writer.add_paper`` / ``.remove_paper`` rather than flattened into this
package's namespace: those names would collide with the identically-named HTTP
route handlers that call them.

The embedder/reranker/llm backends are selected by ``draccus.ChoiceRegistry``
config variants (``embedding.type`` / ``reranker.type`` / ``llm.*.type``); the
``build_*`` functions match on the variant. ``server`` composes ``rag``; ``rag``
never imports ``server``.
"""

from __future__ import annotations

from . import config_writer
from .chunking import Chunk, chunk_markdown
from .cited_papers import CitedPaper, extract_cited_arxiv_ids
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
    WebSearchCfg,
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
from .llm import LLMBackend, Usage, build_llm
from .manifest import Manifest
from .pipeline import build_embedder_from_config, ingest_paper, pending_papers
from .query_expansion import generate_paraphrases
from .reranker import Reranker, build_reranker
from .search import Result, Searcher
from .sparse import BM25Index, build_sparse_index, reciprocal_rank_fusion, rrf_scores
from .tagger import generate_tags
from .web_search import WebResult, WebSearcher, build_web_searcher

__all__ = [
    "AnthropicSpec",
    "BM25Cfg",
    "BM25Index",
    "Chunk",
    "ChunkingCfg",
    "CitedPaper",
    "Config",
    "Embedder",
    "config_writer",
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
    "Usage",
    "VLLMSpec",
    "Verdict",
    "WebResult",
    "WebSearchCfg",
    "WebSearcher",
    "best_support",
    "build_embedder",
    "build_embedder_from_config",
    "build_faithfulness_checker",
    "build_llm",
    "build_reranker",
    "build_sparse_index",
    "build_web_searcher",
    "chunk_markdown",
    "extract_cited_arxiv_ids",
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
