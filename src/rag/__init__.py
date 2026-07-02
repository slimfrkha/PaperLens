"""PaperLens RAG core: config-driven ingestion + two-stage retrieval.

Import the public API from this package rather than the leaf modules
(e.g. ``from rag import Searcher, load_config``); the module layout below is an
internal detail and may change.

Module layering — imports flow one way (top -> bottom); there are no cycles::

    config  chunking  embedders  extract  manifest        (leaves: no intra-rag deps)
       |        |          |                    |
      llm      index(chunking, embedders)     search(embedders)
       |
    tagger(llm)          pipeline(extract, index, manifest, tagger)
                              |
                         ingest(pipeline, index, manifest, tagger)

The ``server`` package composes ``rag``; ``rag`` never imports ``server``.
"""

from __future__ import annotations

from .chunking import Chunk, chunk_markdown
from .config import Config, LLMSpec, Paper, load_config
from .embedders import Embedder, HFEmbedder, OpenAIEmbedder, build_embedder, register_embedder
from .index import index_markdown, open_collection
from .llm import LLMBackend, build_llm, register_llm
from .manifest import Manifest
from .pipeline import build_embedder_from_config, ingest_paper, pending_papers
from .search import Result, Searcher
from .tagger import generate_tags

__all__ = [
    "Chunk",
    "Config",
    "Embedder",
    "HFEmbedder",
    "LLMBackend",
    "LLMSpec",
    "Manifest",
    "OpenAIEmbedder",
    "Paper",
    "Result",
    "Searcher",
    "build_embedder",
    "build_embedder_from_config",
    "build_llm",
    "chunk_markdown",
    "generate_tags",
    "index_markdown",
    "ingest_paper",
    "load_config",
    "open_collection",
    "pending_papers",
    "register_embedder",
    "register_llm",
]
