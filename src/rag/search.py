"""Retrieval over the Chroma RAG DB, with optional cross-encoder reranking and hybrid BM25 fusion.

Two-stage retrieval:
  1. Dense vector search in Chroma returns the top ``candidates`` chunks. When hybrid sparse
     retrieval is enabled, a BM25 lexical search runs alongside it and the two rankings are
     fused via reciprocal rank fusion (RRF) before reranking. When multi-query expansion is
     also enabled, the chat LLM paraphrases the query and every variant's dense (+ sparse)
     ranking is fused into that same flat RRF pass — a recall boost against how a question
     happens to be phrased, opt-in like hybrid (see ``MultiQueryCfg`` in ``config.py``).
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

from .config import AnthropicSpec
from .embedders import HFEmbedder
from .llm import LLMBackend, build_llm
from .query_expansion import generate_paraphrases
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
    section_number: str
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
        multi_query_enabled: bool = False,
        multi_query_n: int = 3,
        multi_query_fetch_multiplier: int = 3,
        llm: LLMBackend | None = None,
    ):
        import chromadb

        # Multi-query is opt-in with no sane default LLM (unlike reranker/sparse, which
        # always have a default backend) — require one explicitly rather than lazily
        # building one, matching build_reranker's check that the `llm` reranker needs an
        # LLM passed.
        if multi_query_enabled and llm is None:
            raise ValueError("multi_query_enabled requires an LLM; pass Searcher(..., llm=...).")

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
        self.multi_query_enabled = multi_query_enabled
        self.multi_query_n = multi_query_n
        self.multi_query_fetch_multiplier = multi_query_fetch_multiplier
        self.llm = llm

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

    def dense_recall(
        self, query: str, fetch_n: int, where: dict[str, Any] | None
    ) -> tuple[list[str], dict[str, Result]]:
        """One query's dense ranking: chunk ids in Chroma's distance order, plus their Results.

        The shared per-query-variant retrieval primitive. ``search`` calls this once per
        variant (the single-query path below is the ``len(variants) == 1`` case of the same
        call); ``eval.harness._retrieve`` and ``eval.optimizer.build_cache`` call it directly
        to get the same dense pool ``search`` would compute — chunk ids and per-id Results
        (``.text`` for reranking, ``.score`` for a no-rerank dense-only ranking) that
        ``search``'s own return value (a fused/reranked ``list[Result]``) doesn't expose.
        """
        # Asymmetric embedders (e.g. Gemini) embed queries differently from docs;
        # symmetric ones fall through to __call__.
        embed_query = getattr(self.embedder, "embed_query", None)
        qvec = embed_query([query]) if embed_query else self.embedder([query])
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
        by_id = {
            cid: Result(
                score=1 - d,  # cosine distance -> similarity
                paper_id=m["paper_id"],
                breadcrumb=m["breadcrumb"],
                section_title=m["section_title"],
                section_number=m["section_number"],
                text=doc,
                body=m["body"],
            )
            for cid, doc, m, d in zip(dense_ids, docs, metas, dists, strict=True)
        }
        return dense_ids, by_id

    def backfill_missing(self, by_id: dict[str, Result], candidate_ids: list[str]) -> None:
        """Fetch any candidate id ``dense_recall`` didn't already return — reached via BM25
        (hybrid retrieval); in multi-query this is always this variant's own BM25 hits, since
        a hit already found by another variant's dense recall is already in ``by_id`` — and
        add it with a placeholder score. Mutates ``by_id`` in place (not a return-a-new-dict
        design): ``search``'s multi-query loop and ``eval.optimizer.build_cache``'s per-variant
        loop both call this repeatedly, accumulating onto what earlier calls already backfilled
        without a reassignment at every call site. The placeholder score (0.0) is overwritten
        immediately after by ``_tag_and_collect`` in ``search``; eval callers only ever read
        ``.text`` off a backfilled Result (rerank input), never ``.score``, so the placeholder
        is never mistaken for a real one there either. Only paper_id/breadcrumb/section/text/body
        need to be real here.
        """
        missing = [cid for cid in candidate_ids if cid not in by_id]
        if not missing:
            return
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
                section_number=m["section_number"],
                text=doc,
                body=m["body"],
            )

    def _tag_and_collect(
        self,
        by_id: dict[str, Result],
        fused_ids: list[str],
        scores: dict[str, float],
        dense_set: set[str],
        sparse_set: set[str],
    ) -> list[Result]:
        """Called once, after fusion is fully resolved: stamp each fused id's RRF score and
        dense/sparse/both source into ``by_id``, then collect in fused order."""
        for cid in fused_ids:
            by_id[cid].score = scores[cid]
            in_dense, in_sparse = cid in dense_set, cid in sparse_set
            by_id[cid].source = (
                "both" if in_dense and in_sparse else "sparse" if in_sparse else "dense"
            )
        return [by_id[cid] for cid in fused_ids]

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
        allowed = set(ids) if ids is not None else None

        variants = [query]
        if self.multi_query_enabled:
            assert self.llm is not None  # enforced at construction
            variants += generate_paraphrases(query, self.llm, n=self.multi_query_n)

        if len(variants) == 1:
            # Fusion breadth: when hybrid is on, both sides over-fetch fetch_multiplier * candidates
            # before RRF-fusing, so fusion has margin to promote a sparse-only hit that dense ranked
            # outside its own top-`candidates` — the final truncation to `candidates` happens after
            # fusion, not before, so downstream (rerank, k) sees the same pool shape either way.
            fetch_n = candidates * self._fetch_multiplier if self.sparse_enabled else candidates
            dense_ids, by_id = self.dense_recall(query, fetch_n, where)
            if not dense_ids:
                return []

            if self.sparse_enabled:
                sparse_ids = self.sparse.search(query, n=fetch_n, allowed_ids=allowed)
                dense_set = set(dense_ids)
                sparse_set = set(sparse_ids)
                fused_ids = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self._rrf_k)[
                    :candidates
                ]
                self.backfill_missing(by_id, fused_ids)
                # RRF score uniformly across the whole fused set — don't mix cosine similarity and
                # RRF scales (Result.score's docstring covers this: "rerank score if reranked, else
                # cosine similarity" stops being true for a hybrid+no-rerank query).
                scores = rrf_scores([dense_ids, sparse_ids], k=self._rrf_k)
                results = self._tag_and_collect(by_id, fused_ids, scores, dense_set, sparse_set)
            else:
                results = list(by_id.values())
        else:
            # Multi-query: one flat RRF pass over every ranking from every variant at once —
            # dense_v1, sparse_v1, dense_v2, sparse_v2, ... — not a fusion of per-variant
            # fusions. RRF is already generalized to N rankings, so nesting two RRF passes
            # (fuse dense+sparse per variant, then fuse those fused results again) would
            # double-apply its rank-decay to evidence that already went through one fusion
            # round, with no clean interpretation of what that compounding means.
            # `multi_query_fetch_multiplier` governs over-fetch depth for every ranking here
            # (both a variant's dense and its sparse query), deliberately independent of
            # `sparse.fetch_multiplier` — needed even when hybrid is off, or the fuse has no
            # margin to promote a hit ranked just outside one variant's own top-`candidates`.
            variant_fetch_n = candidates * self.multi_query_fetch_multiplier
            rankings: list[list[str]] = []
            dense_ids_all: list[str] = []
            sparse_ids_all: list[str] = []
            by_id = {}
            for v in variants:
                dense_ids_v, by_id_v = self.dense_recall(v, variant_fetch_n, where)
                for cid, r in by_id_v.items():
                    by_id.setdefault(cid, r)
                rankings.append(dense_ids_v)
                dense_ids_all += dense_ids_v
                if self.sparse_enabled:
                    sparse_ids_v = self.sparse.search(v, n=variant_fetch_n, allowed_ids=allowed)
                    rankings.append(sparse_ids_v)
                    sparse_ids_all += sparse_ids_v
                    self.backfill_missing(by_id, sparse_ids_v)
            fused_ids = reciprocal_rank_fusion(rankings, k=self._rrf_k)[:candidates]
            scores = rrf_scores(rankings, k=self._rrf_k)
            dense_set = set(dense_ids_all)
            sparse_set = set(sparse_ids_all)
            results = self._tag_and_collect(by_id, fused_ids, scores, dense_set, sparse_set)

        if rerank:
            try:
                scores = self.reranker.score(query, [r.text for r in results])
                if len(scores) != len(results):
                    raise ValueError(
                        f"reranker returned {len(scores)} scores for {len(results)} docs"
                    )
                # Lengths are already checked equal above; strict=False since a mismatch is
                # handled there, not here.
                for r, s in zip(results, scores, strict=False):
                    r.score = s
                results.sort(key=lambda r: r.score, reverse=True)
            except Exception as e:
                # Reranker failure (model load, inference error, network/rate-limit/auth from an
                # LLM reranker, or a malformed score list) — degrade to the pre-rerank (dense/RRF)
                # order rather than failing the whole search.
                print(
                    f"  [warn] reranking failed for query {query!r},"
                    f" falling back to pre-rerank order: {e}"
                )

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
    p.add_argument(
        "--multi-query",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="paraphrase the query with the chat LLM and RRF-fuse each variant's results",
    )
    p.add_argument(
        "--multi-query-n", type=int, default=3, help="paraphrases to generate (--multi-query only)"
    )
    args = p.parse_args()

    searcher = Searcher(
        db_dir=args.db_dir,
        collection=args.collection,
        embedder_model=args.embedder,
        query_prefix=args.query_prefix,
        reranker_model=args.reranker,
        sparse_enabled=args.sparse,
        multi_query_enabled=args.multi_query,
        multi_query_n=args.multi_query_n,
        llm=build_llm(AnthropicSpec()) if args.multi_query else None,
    )
    results = searcher.search(
        args.query,
        k=args.k,
        candidates=args.candidates,
        paper=args.paper,
        rerank=not args.no_rerank,
    )

    mode = "hybrid" if searcher.sparse_enabled else "vector"
    mode = f"{mode}+multi-query" if searcher.multi_query_enabled else mode
    tag = mode if args.no_rerank else f"{mode}+rerank"
    print(f"\nQ: {args.query}   [{tag}, top {args.k} of {args.candidates} candidates]")
    for r in results:
        crumb = r.breadcrumb.split(" > ", 1)[-1]
        snippet = textwrap.shorten(r.body.replace("\n", " "), width=200)
        print(f"\n  [{r.score:.3f}] {r.paper_id}  ::  {crumb}")
        print(f"          {snippet}")


if __name__ == "__main__":
    main()
