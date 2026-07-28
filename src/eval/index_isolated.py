"""Guard 1: isolated re-index cells for chunking sweeps — never touch the prod collection.

Changing ``chunking.max_tokens`` and re-ingesting into the **same** collection contaminates
it: ``_chunk_id`` is a content hash of ``paper_id|section_number|section_title|part`` and
``upsert_chunks`` only upserts, so re-cut chunks get *new* ids and pile up alongside the old
ones — every retrieval number after that is garbage (``src/rag/index.py:40-91``). The embedder
is protected (``embedder.name()`` namespaces the collection); chunking is not.

So each cell gets its **own** collection, named from a hash of the chunking (+ embedding)
config, built fresh (``reset=True``) in a throwaway temp dir — the prod ``paths.rag_db`` is
never opened. After upsert we assert ``count() == len(chunks)``: a fresh collection holds
exactly what we just put in it, so any shortfall means two chunks collided on ``_chunk_id``
— a latent *prod* bug (the second silently overwrites the first on disk too), surfaced here,
not a harness fault.

We reuse ``rag.index`` (``open_collection`` + ``upsert_chunks``) rather than reimplement
indexing, and return the ordinary :class:`~rag.search.Searcher` over the cell's collection —
so the whole retrieval-screen scoring/caching path (``harness.score_items``,
``optimizer.build_cache``) runs over an ablation cell unchanged.
"""

from __future__ import annotations

import hashlib

from rag.chunking import Chunk, chunk_markdown
from rag.config import ChunkingCfg, EmbeddingCfg
from rag.embedders import build_embedder
from rag.index import open_collection, upsert_chunks
from rag.reranker import Reranker
from rag.search import Searcher


def cell_signature(chunking: ChunkingCfg, embedding: EmbeddingCfg) -> str:
    """A collection name unique to a (chunking, embedding) cell — Guard 1's isolation key.

    Embedding identity is folded in even though the default sweep varies only chunking: it
    costs nothing and keeps cells from colliding the day a user does vary the embedder.
    """
    key = (
        f"mt={chunking.max_tokens}|ov={chunking.overlap_tokens}|mn={chunking.min_tokens}|"
        f"nr={chunking.noise_ratio}|skip={sorted(chunking.extra_skip_titles)}|"
        f"emb={type(embedding).__name__}:{embedding.model}"
    )
    return "cell_" + hashlib.md5(key.encode()).hexdigest()[:16]


def chunks_for(pool: dict[str, str], chunking: ChunkingCfg) -> list[Chunk]:
    """Re-chunk every paper's markdown at ``chunking`` — in memory, prod collection untouched."""
    chunks: list[Chunk] = []
    for paper_id, md in pool.items():
        chunks.extend(
            chunk_markdown(
                md,
                paper_id=paper_id,
                max_tokens=chunking.max_tokens,
                overlap_tokens=chunking.overlap_tokens,
                min_tokens=chunking.min_tokens,
                noise_ratio=chunking.noise_ratio,
                extra_skip_titles=chunking.extra_skip_titles,
            )
        )
    return chunks


def build_isolated_searcher(
    pool: dict[str, str],
    chunking: ChunkingCfg,
    embedding: EmbeddingCfg,
    *,
    db_dir: str,
    embedder=None,
    reranker: Reranker | None = None,
) -> Searcher:
    """Re-index ``pool`` at ``(chunking, embedding)`` into a fresh cell collection under
    ``db_dir`` and return a :class:`Searcher` over it.

    ``db_dir`` is a throwaway directory owned by the caller (a ``TemporaryDirectory`` in
    prod, ``tmp_path`` in tests) — never ``paths.rag_db``. ``embedder`` is injectable so
    offline tests skip the model download; production builds it from ``embedding``.
    """
    embedder = embedder or build_embedder(embedding)
    chunks = chunks_for(pool, chunking)
    name = cell_signature(chunking, embedding)
    # reset=True: a *fresh* collection every call, so a re-chunk can never pile up on a
    # previous cell's ids (Guard 1). The count assertion below then has exact ground truth.
    collection = open_collection(db_dir, name, embedder_name=embedder.name(), reset=True)
    upsert_chunks(collection, embedder, chunks)
    if collection.count() != len(chunks):
        raise AssertionError(
            f"cell {name}: collection.count()={collection.count()} != {len(chunks)} chunks — "
            f"two chunks collided on _chunk_id (a latent prod bug: the second overwrites the "
            f"first on disk too), not a harness fault."
        )
    return Searcher(db_dir=db_dir, collection=name, embedder=embedder, reranker=reranker)
