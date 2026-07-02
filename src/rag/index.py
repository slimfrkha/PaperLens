"""Build the local RAG vector DB by indexing extracted arXiv papers into Chroma.

Chunking is section-aware with hierarchical breadcrumbs (see chunking.py).
Embeddings are computed here and stored directly, so the collection is
portable and independent of Chroma's embedding-function API.

Examples
--------
Default (best local model that fits a 64GB M3 Max, fast):
    python -m rag.index

Swap the embedder for another HF model:
    python -m rag.index --embedder BAAI/bge-large-en-v1.5

Use an OpenAI-compatible API instead:
    python -m rag.index --embedder-type openai --embedder text-embedding-3-large
    python -m rag.index --embedder-type openai --embedder my-model \
        --api-base http://localhost:1234/v1 --api-key-env LOCAL_API_KEY
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time

from .chunking import Chunk, chunk_markdown
from .embedders import build_embedder

# Best quality-per-speed local default: BGE-M3 is battle-tested with
# sentence-transformers, has an 8192-token context (won't truncate our
# ~512-token chunks), 1024-dim dense vectors, and indexes this corpus in
# seconds-to-minutes on Apple MPS while using a fraction of 64GB.
DEFAULT_EMBEDDER = "BAAI/bge-m3"


def _chunk_id(c: Chunk) -> str:
    m = c.metadata
    key = f"{m['paper_id']}|{m['section_number']}|{m['section_title']}|{m['part']}"
    return hashlib.md5(key.encode()).hexdigest()


def collect_chunks(docs_dir: str, **chunk_kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        paper_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            md = f.read()
        doc_chunks = chunk_markdown(md, paper_id=paper_id, **chunk_kwargs)
        print(f"  {paper_id:32s} -> {len(doc_chunks):3d} chunks")
        chunks.extend(doc_chunks)
    return chunks


def open_collection(db_dir: str, collection: str, *, embedder_name: str | None = None,
                    reset: bool = False):
    """Open (or create) the persistent Chroma collection, cosine space."""
    import chromadb

    client = chromadb.PersistentClient(path=db_dir)
    if reset:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
    meta = {"hnsw:space": "cosine"}
    if embedder_name:
        meta["embedder"] = embedder_name
    return client.get_or_create_collection(name=collection, metadata=meta)


def upsert_chunks(collection, embedder, chunks: list[Chunk], batch_size: int = 32,
                  progress=None) -> int:
    """Embed and upsert chunks in batches. `progress(done, total)` is optional."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embedder([c.text for c in batch])
        collection.upsert(
            ids=[_chunk_id(c) for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        if progress:
            progress(min(i + batch_size, len(chunks)), len(chunks))
    return len(chunks)


def index_markdown(collection, embedder, md_path: str, paper_id: str, *,
                   max_tokens: int = 512, overlap_tokens: int = 64,
                   batch_size: int = 32, progress=None) -> int:
    """Chunk one markdown file and upsert it. Returns the chunk count."""
    with open(md_path) as f:
        md = f.read()
    chunks = chunk_markdown(
        md, paper_id=paper_id, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )
    upsert_chunks(collection, embedder, chunks, batch_size=batch_size, progress=progress)
    return len(chunks)


def build(args) -> None:
    print(f"== Chunking docs in {args.docs_dir} ==")
    chunks = collect_chunks(
        args.docs_dir,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
    )
    if not chunks:
        raise SystemExit(f"No chunks produced from {args.docs_dir} (any *.md files?)")
    print(f"  total: {len(chunks)} chunks")

    print(f"== Loading embedder [{args.embedder_type}] {args.embedder} ==")
    embedder = build_embedder(
        args.embedder,
        args.embedder_type,
        batch_size=args.batch_size,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        max_seq_length=args.max_seq_length,
    )

    if args.reset:
        print(f"  reset: dropping existing collection {args.collection!r}")
    collection = open_collection(
        args.db_dir, args.collection, embedder_name=embedder.name(), reset=args.reset
    )

    print(f"== Embedding + upserting into {args.db_dir} :: {args.collection} ==")
    t0 = time.time()
    upsert_chunks(
        collection, embedder, chunks, batch_size=args.batch_size,
        progress=lambda done, total: print(f"  {done:4d}/{total} embedded", end="\r"),
    )
    dt = time.time() - t0

    print()
    print(f"Done. {collection.count()} chunks indexed in {dt:.1f}s "
          f"({len(chunks) / dt:.0f} chunks/s) using {embedder.name()}.")
    print(f"DB: {os.path.abspath(args.db_dir)}  collection: {args.collection!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs-dir", default="papers/text", help="dir of extracted *.md papers")
    p.add_argument("--db-dir", default="rag_db", help="Chroma persistent directory")
    p.add_argument("--collection", default="arxiv_papers")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                   help="HF model id, or API model name when --embedder-type openai")
    p.add_argument("--embedder-type", choices=["hf", "openai"], default="hf")
    p.add_argument("--api-base", default=None, help="base URL for OpenAI-compatible endpoint")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="env var holding the API key (openai type)")
    p.add_argument("--max-tokens", type=int, default=512, help="target max tokens per chunk")
    p.add_argument("--overlap", type=int, default=64, help="overlap tokens between sub-chunks")
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="cap embedder input length (guards MPS 2**32 tensor limit)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--reset", action="store_true", help="drop the collection before indexing")
    build(p.parse_args())


if __name__ == "__main__":
    main()
