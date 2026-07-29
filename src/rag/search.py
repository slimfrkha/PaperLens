"""Retrieval over the Chroma RAG DB, with optional cross-encoder reranking and hybrid BM25 fusion.

Two-stage retrieval:
  1. Dense vector search in Chroma returns the top ``candidates`` chunks. When hybrid sparse
     retrieval is enabled, a BM25 lexical search runs alongside it and the two rankings are
     fused via reciprocal rank fusion (RRF) before reranking.
  2. A cross-encoder reranker (default BAAI/bge-reranker-v2-m3, the sibling of
     the bge-m3 embedder) rescores each (query, chunk) pair and keeps the top
     ``k``. Reranking reorders keyword-y matches below the truly relevant ones;
     disable it with ``--no-rerank`` for pure vector results.

CLI:
    python -m rag.search "how does MLA reduce the KV cache?"
    python -m rag.search "FP8 quantization" --k 5 --candidates 30 --paper <paper_id>
    python -m rag.search "long context" --no-rerank
    python -m rag.search "SwiGLU activation" --sparse
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from typing import Any, cast

from .embedders import HFEmbedder
from .reranker import CrossEncoderReranker, Reranker
from .sparse import BM25Index, build_sparse_index, reciprocal_rank_fusion, rrf_scores

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass
class Result:
    score: float  # rerank score if reranked; RRF score if hybrid+no-rerank; else cosine similarity
    paper_id: str
    breadcrumb: str
    section_title: str
    text: str  # breadcrumb + body (what was embedded)
    body: str  # body only
    source: str = "dense"  # "dense" | "sparse" | "both" — which retrieval pool(s) surfaced this


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
        embedder=None,
        query_prefix: str = "",
        reranker: Reranker | None = None,
        sparse_enabled: bool = False,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        fetch_multiplier: int = 3,
        sparse: BM25Index | None = None,
    ):
        import chromadb

        self.device = _pick_device(device)
        # Query embedder: inject one to avoid loading the local model (tests); the
        # default builds the same HF model used at indexing time. query_prefix only
        # applies to this default-build path — an injected embedder already carries
        # its own prefixes (see build_embedder).
        self.embedder = embedder or HFEmbedder(
            embedder_model, device=self.device, query_prefix=query_prefix
        )
        self.collection = chromadb.PersistentClient(path=db_dir).get_collection(collection)
        self._reranker_model = reranker_model
        # Inject a Reranker (e.g. the config-built llm reranker) or leave None to
        # lazily build the default hf cross-encoder on first use.
        self._reranker = reranker
        self.sparse_enabled = sparse_enabled
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._rrf_k = rrf_k
        self._fetch_multiplier = fetch_multiplier
        # Inject a BM25Index for tests, or leave None to lazily build/rebuild on first use.
        self._sparse = sparse

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self._reranker_model, device=self.device)
        return self._reranker

    @property
    def sparse(self) -> BM25Index:
        """The BM25 index, rebuilt whenever the collection has grown since it was built.

        A BM25 index is a snapshot (its IDF needs global corpus stats), unlike the stateless
        reranker — so this checks staleness on every access rather than caching forever. That's
        what makes a live `/api/admin/rescan` (server/main.py) safe: dense search already
        queries Chroma live, and this keeps the sparse side from silently falling behind it. The
        extra cost when sparse is on is one cheap `collection.count()` call per search — a
        metadata lookup, not a corpus scan — with a full rebuild only right after a rescan
        changed the count.
        """
        if self._sparse is None or self._sparse.built_at_count != self.collection.count():
            self._sparse = build_sparse_index(self.collection, k1=self._bm25_k1, b=self._bm25_b)
        return self._sparse

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

        # Asymmetric embedders (e.g. Gemini) embed queries differently from docs;
        # symmetric ones fall through to __call__.
        embed_query = getattr(self.embedder, "embed_query", None)
        qvec = embed_query([query]) if embed_query else self.embedder([query])

        # Fusion breadth: when hybrid is on, both sides over-fetch fetch_multiplier * candidates
        # before RRF-fusing, so fusion has margin to promote a sparse-only hit that dense ranked
        # outside its own top-`candidates` — the final truncation to `candidates` happens after
        # fusion, not before, so downstream (rerank, k) sees the same pool shape either way.
        fetch_n = candidates * self._fetch_multiplier if self.sparse_enabled else candidates
        # Chroma's type stubs narrow query_embeddings/results more tightly than
        # runtime accepts; the include= keys are always present and non-None here.
        res = cast(
            dict[str, Any],
            self.collection.query(
                query_embeddings=qvec,  # ty: ignore[invalid-argument-type]  # list invariance vs Chroma's Sequence param
                n_results=fetch_n,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )
        dense_ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        if not dense_ids:
            return []

        by_id = {
            cid: Result(
                score=1 - d,  # cosine distance -> similarity
                paper_id=m["paper_id"],
                breadcrumb=m["breadcrumb"],
                section_title=m["section_title"],
                text=doc,
                body=m["body"],
            )
            for cid, doc, m, d in zip(dense_ids, docs, metas, dists, strict=True)
        }

        if self.sparse_enabled:
            allowed = set(ids) if ids is not None else None
            sparse_ids = self.sparse.search(query, n=fetch_n, allowed_ids=allowed)
            dense_set = set(dense_ids)
            sparse_set = set(sparse_ids)
            fused_ids = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self._rrf_k)[:candidates]
            missing = [cid for cid in fused_ids if cid not in by_id]
            if missing:
                got = cast(
                    dict[str, Any],
                    self.collection.get(ids=missing, include=["documents", "metadatas"]),
                )
                for cid, doc, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True):
                    by_id[cid] = Result(
                        score=0.0,
                        paper_id=m["paper_id"],
                        breadcrumb=m["breadcrumb"],
                        section_title=m["section_title"],
                        text=doc,
                        body=m["body"],
                    )
            # RRF score uniformly across the whole fused set — don't mix cosine similarity and
            # RRF scales (Result.score's docstring covers this: "rerank score if reranked, else
            # cosine similarity" stops being true for a hybrid+no-rerank query).
            scores = rrf_scores([dense_ids, sparse_ids], k=self._rrf_k)
            for cid in fused_ids:
                by_id[cid].score = scores[cid]
                in_dense, in_sparse = cid in dense_set, cid in sparse_set
                by_id[cid].source = (
                    "both" if in_dense and in_sparse else "sparse" if in_sparse else "dense"
                )
            results = [by_id[cid] for cid in fused_ids]
        else:
            results = list(by_id.values())

        if rerank:
            scores = self.reranker.score(query, [r.text for r in results])
            for r, s in zip(results, scores, strict=True):
                r.score = s
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
    p.add_argument("--query-prefix", default="", help="prepended to the query before embedding")
    p.add_argument("--reranker", default=DEFAULT_RERANKER)
    p.add_argument("-k", type=int, default=5, help="final results to show")
    p.add_argument("--candidates", type=int, default=20, help="vector hits to rerank")
    p.add_argument("--paper", default=None, help="restrict to one paper_id")
    p.add_argument("--no-rerank", action="store_true", help="skip cross-encoder rerank")
    p.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fuse in BM25 lexical search (hybrid); --no-sparse is dense-only (default)",
    )
    args = p.parse_args()

    searcher = Searcher(
        db_dir=args.db_dir,
        collection=args.collection,
        embedder_model=args.embedder,
        query_prefix=args.query_prefix,
        reranker_model=args.reranker,
        sparse_enabled=args.sparse,
    )
    results = searcher.search(
        args.query,
        k=args.k,
        candidates=args.candidates,
        paper=args.paper,
        rerank=not args.no_rerank,
    )

    mode = "hybrid" if searcher.sparse_enabled else "vector"
    tag = mode if args.no_rerank else f"{mode}+rerank"
    print(f"\nQ: {args.query}   [{tag}, top {args.k} of {args.candidates} candidates]")
    for r in results:
        crumb = r.breadcrumb.split(" > ", 1)[-1]
        snippet = textwrap.shorten(r.body.replace("\n", " "), width=200)
        print(f"\n  [{r.score:.3f}] {r.paper_id}  ::  {crumb}")
        print(f"          {snippet}")


if __name__ == "__main__":
    main()
