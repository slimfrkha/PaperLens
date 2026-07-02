"""Retrieval over the Chroma RAG DB, with optional cross-encoder reranking.

Two-stage retrieval:
  1. Dense vector search in Chroma returns the top ``candidates`` chunks.
  2. A cross-encoder reranker (default BAAI/bge-reranker-v2-m3, the sibling of
     the bge-m3 embedder) rescores each (query, chunk) pair and keeps the top
     ``k``. Reranking reorders keyword-y matches below the truly relevant ones;
     disable it with ``--no-rerank`` for pure vector results.

CLI:
    python -m rag.search "how does MLA reduce the KV cache?"
    python -m rag.search "FP8 quantization" --k 5 --candidates 30 --paper deepseek-v3
    python -m rag.search "long context" --no-rerank
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from typing import Any, cast

from .embedders import HFEmbedder

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass
class Result:
    score: float  # rerank score if reranked, else cosine similarity
    paper_id: str
    breadcrumb: str
    section_title: str
    text: str  # breadcrumb + body (what was embedded)
    body: str  # body only


def _pick_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


class Searcher:
    def __init__(
        self,
        db_dir: str = "rag_db",
        collection: str = "arxiv_papers",
        embedder_model: str = DEFAULT_EMBEDDER,
        reranker_model: str = DEFAULT_RERANKER,
        device: str | None = None,
    ):
        import chromadb

        self.device = _pick_device(device)
        self.embedder = HFEmbedder(embedder_model, device=self.device)
        self.collection = chromadb.PersistentClient(path=db_dir).get_collection(collection)
        self._reranker_model = reranker_model
        self._reranker = None  # lazy: only load when reranking is actually used

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self._reranker_model, device=self.device, max_length=512)
        return self._reranker

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 20,
        paper: str | None = None,
        paper_ids: list[str] | None = None,
        rerank: bool = True,
    ) -> list[Result]:
        # Resolve the paper filter. `paper_ids` (e.g. from a tag filter) and a
        # single `paper` intersect; an explicit empty set matches nothing.
        ids = paper_ids
        if paper is not None:
            ids = [paper] if ids is None else [p for p in ids if p == paper]
        if ids is not None and len(ids) == 0:
            return []
        where = {"paper_id": {"$in": ids}} if ids is not None else None

        qvec = self.embedder([query])
        # Chroma's type stubs narrow query_embeddings/results more tightly than
        # runtime accepts; the include= keys are always present and non-None here.
        res = cast(
            dict[str, Any],
            self.collection.query(
                query_embeddings=qvec,  # ty: ignore[invalid-argument-type]  # list invariance vs Chroma's Sequence param
                n_results=candidates,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        if not docs:
            return []

        results = [
            Result(
                score=1 - d,  # cosine distance -> similarity
                paper_id=m["paper_id"],
                breadcrumb=m["breadcrumb"],
                section_title=m["section_title"],
                text=doc,
                body=doc.split("\n\n", 1)[-1],
            )
            for doc, m, d in zip(docs, metas, dists, strict=True)
        ]

        if rerank:
            scores = self.reranker.predict([(query, r.text) for r in results])
            for r, s in zip(results, scores, strict=True):
                r.score = float(s)
            results.sort(key=lambda r: r.score, reverse=True)

        return results[:k]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("query")
    p.add_argument("--db-dir", default="rag_db")
    p.add_argument("--collection", default="arxiv_papers")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--reranker", default=DEFAULT_RERANKER)
    p.add_argument("-k", type=int, default=5, help="final results to show")
    p.add_argument("--candidates", type=int, default=20, help="vector hits to rerank")
    p.add_argument("--paper", default=None, help="restrict to one paper_id")
    p.add_argument("--no-rerank", action="store_true", help="skip cross-encoder rerank")
    args = p.parse_args()

    searcher = Searcher(
        db_dir=args.db_dir,
        collection=args.collection,
        embedder_model=args.embedder,
        reranker_model=args.reranker,
    )
    results = searcher.search(
        args.query,
        k=args.k,
        candidates=args.candidates,
        paper=args.paper,
        rerank=not args.no_rerank,
    )

    tag = "vector" if args.no_rerank else "rerank"
    print(f"\nQ: {args.query}   [{tag}, top {args.k} of {args.candidates} candidates]")
    for r in results:
        crumb = r.breadcrumb.split(" > ", 1)[-1]
        snippet = textwrap.shorten(r.body.replace("\n", " "), width=200)
        print(f"\n  [{r.score:.3f}] {r.paper_id}  ::  {crumb}")
        print(f"          {snippet}")


if __name__ == "__main__":
    main()
