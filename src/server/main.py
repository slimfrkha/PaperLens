"""FastAPI app: chat (SSE), papers, tags, admin status, and static SPA."""

from __future__ import annotations

import asyncio
import re
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from rag import config_writer
from rag.config import BM25Cfg, Config, HFEmbeddingCfg, Paper, parse_config
from rag.index import open_collection, remove_paper_chunks
from rag.llm import build_llm
from rag.manifest import Manifest
from rag.pipeline import pending_papers
from rag.reranker import build_reranker
from rag.search import Searcher

from .agent import ChatAgent
from .annotations import AnnotationStore
from .chat_turn import run_turn
from .chats import ChatStore
from .schemas import (
    AddPapersRequest,
    AnnotationCreate,
    AnnotationUpdate,
    ChatRequest,
    ClassifyModeRequest,
    ClassifyModeResponse,
    FeedbackRequest,
)
from .worker import IngestionWorker

# Modern arXiv id shape only (YYMM.NNNNN[N]); old-style slashed ids (e.g.
# hep-th/9901001) are rejected, not just unsupported — `name = arxiv_id` below
# becomes a filename stem (pipeline.py's pdf/md paths), and a `/` in that string
# would turn into a path component instead of a filename.
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _normalize_arxiv_id(raw: str) -> str | None:
    """Extract a bare arXiv id from a raw id or an arxiv.org abs/pdf URL.

    Returns None if the result isn't a modern-format id."""
    s = raw.strip()
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s)
    s = re.sub(r"\.pdf$", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s if _ARXIV_ID_RE.match(s) else None


def create_app(cfg: Config) -> FastAPI:
    web_dist = Path(cfg.paths.web_dist)
    manifest = Manifest(cfg.paths.rag_db)
    chats = ChatStore(cfg.paths.chat_history)
    annotations = AnnotationStore(cfg.paths.annotations)
    icfg = cfg.for_ingest()  # ingestion-only view for the worker + pending-paper checks
    worker = IngestionWorker(icfg, manifest)
    # Ensure the collection exists so the chat Searcher can open it even when the
    # DB is still empty (ingestion creates it too, but chat may be hit first).
    # Also reused as-is by the admin remove-paper route below — no need to build an
    # embedder just to delete chunks by a paper_id metadata filter.
    collection = open_collection(cfg.paths.rag_db, cfg.collection)

    # Chat models (embedder + reranker + LLM) are heavy — build once on first use.
    # The lock keeps the build single-shot when the startup warmer (below) and an
    # early /api/chat race for it.
    lazy: dict = {"agent": None}
    lazy_lock = threading.Lock()

    def get_agent() -> ChatAgent:
        if lazy["agent"] is None:
            with lazy_lock:
                if lazy["agent"] is None:
                    # Reused for both the reranker (if `llm` type) and multi-query
                    # paraphrase generation — one client, not two.
                    chat_llm = build_llm(cfg.llm.chat)
                    searcher = Searcher(
                        db_dir=cfg.paths.rag_db,
                        collection=cfg.collection,
                        embedder_model=cfg.embedding.model,
                        query_prefix=cfg.embedding.query_prefix
                        if isinstance(cfg.embedding, HFEmbeddingCfg)
                        else "",
                        reranker=build_reranker(cfg.reranker, llm=chat_llm),
                        sparse_enabled=cfg.sparse.enabled,
                        bm25_k1=cfg.sparse.k1 if isinstance(cfg.sparse, BM25Cfg) else 1.5,
                        bm25_b=cfg.sparse.b if isinstance(cfg.sparse, BM25Cfg) else 0.75,
                        rrf_k=cfg.sparse.rrf_k,
                        fetch_multiplier=cfg.sparse.fetch_multiplier,
                        multi_query_enabled=cfg.multi_query.enabled,
                        multi_query_n=cfg.multi_query.n_paraphrases,
                        multi_query_fetch_multiplier=cfg.multi_query.fetch_multiplier,
                        llm=chat_llm,
                    )
                    lazy["agent"] = ChatAgent(cfg, searcher, manifest)
        return lazy["agent"]

    def warm_models() -> None:
        """Preload the chat models so the first /api/chat isn't a 20-30s wait.
        Building the Searcher loads the embedder eagerly; a tiny dummy search also
        exercises the rerank path to load the cross-encoder, and (if enabled) a
        dummy check_batch loads the faithfulness checker's model too. Failures are
        non-fatal — the next /api/chat rebuilds and surfaces any real error."""
        try:
            agent = get_agent()
            agent.searcher.search("warm up", min_k=1, max_k=1, candidates=1)
            if cfg.faithfulness.enabled:
                agent.faithfulness.check_batch([("warm up", "warm up")])
        except Exception as e:
            print(f"[warn] model warmup skipped: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.ingestion.auto_start:
            worker.trigger()
        # Warm the chat models in the background: startup stays instant (the SPA,
        # papers, and admin are served immediately) while the models load.
        threading.Thread(target=warm_models, name="warm-models", daemon=True).start()
        yield

    app = FastAPI(title="PaperLens", lifespan=lifespan)

    # ---- API ----
    @app.get("/api/papers")
    def list_papers():
        return manifest.papers()

    @app.get("/api/papers/{paper_id}")
    def get_paper(paper_id: str):
        # Prefer the display markdown (figures rendered in) when it exists; fall back to
        # the plain RAG text for papers ingested before this feature, or with
        # extraction.render_images off.
        display_path = Path(cfg.paths.markdown_dir) / f"{paper_id}_display.md"
        text_path = Path(cfg.paths.markdown_dir) / f"{paper_id}.md"
        path = display_path if display_path.exists() else text_path
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        md = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.DOTALL)
        # Docling's display markdown references images as a relative `{paper_id}.assets/`
        # path; rewrite it to the assets route below so the SPA resolves it against the
        # backend instead of its own router.
        md = md.replace(f"{paper_id}.assets/", f"/api/papers/{paper_id}/assets/")
        rec = manifest.get(paper_id) or {}
        return {
            "paper_id": paper_id,
            "title": rec.get("title", paper_id),
            "tags": rec.get("tags", []),
            "arxiv_id": rec.get("arxiv_id"),
            "markdown": md,
        }

    @app.get("/api/papers/{paper_id}/assets/{filename}")
    def get_paper_asset(paper_id: str, filename: str):
        base = (Path(cfg.paths.markdown_dir) / f"{paper_id}.assets").resolve()
        path = (base / filename).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    # ---- annotations ----
    @app.get("/api/papers/{paper_id}/annotations")
    def list_annotations(paper_id: str):
        return annotations.list_all(paper_id)

    @app.post("/api/papers/{paper_id}/annotations")
    def create_annotation(paper_id: str, req: AnnotationCreate):
        return annotations.create(
            paper_id, req.snippet, req.section_title, req.section_slug, req.note
        )

    @app.patch("/api/papers/{paper_id}/annotations/{annotation_id}")
    def update_annotation(paper_id: str, annotation_id: str, req: AnnotationUpdate):
        a = annotations.update(paper_id, annotation_id, req.note)
        return a or JSONResponse({"error": "not found"}, status_code=404)

    @app.delete("/api/papers/{paper_id}/annotations/{annotation_id}")
    def delete_annotation(paper_id: str, annotation_id: str):
        ok = annotations.delete(paper_id, annotation_id)
        return {"ok": ok}

    @app.get("/api/annotations")
    def list_all_annotations():
        # Every annotation across the whole library, joined with paper title/arxiv_id —
        # the read-side aggregation behind the cross-paper Notes page. A plain loop, not a
        # new AnnotationStore method: AnnotationStore stays a pure per-paper file store.
        out = []
        for p in manifest.papers():
            for a in annotations.list_all(p["paper_id"]):
                out.append(
                    {
                        **a,
                        "paper_id": p["paper_id"],
                        "paper_title": p["title"],
                        "arxiv_id": p.get("arxiv_id"),
                    }
                )
        return out

    @app.get("/api/tags")
    def list_tags():
        return manifest.discriminating_tags()

    @app.get("/api/admin/status")
    def admin_status():
        papers = manifest.papers()
        return {
            "db": {
                "n_papers": len(papers),
                "n_chunks": sum(p.get("n_chunks", 0) for p in papers),
            },
            "tags": manifest.all_tags(),
            "pending": [p.name for p in pending_papers(icfg, manifest)],
            "ingestion": worker.snapshot(),
        }

    @app.post("/api/admin/rescan")
    def admin_rescan():
        return {"started": worker.trigger()}

    def _add_one_paper(raw: str) -> dict:
        """Normalize, dedupe, and queue a single arXiv id/URL — one line of an
        add_papers request. There's no separate single-add route: adding one paper
        is just a length-1 batch, so the UI (and any other caller) always goes
        through this same endpoint.

        config_writer's dedup is by arxiv_id (not name) against every existing
        papers: entry, so a paper already curated under a human-chosen name (e.g.
        `deepseek-v3`) is still caught even though this route's generated `name`
        won't match it textually. Only on success does cfg.papers get the new
        entry — an unconditional append here would double-append (and
        double-ingest) on a race between two near-simultaneous requests for the
        same arxiv_id.
        """
        arxiv_id = _normalize_arxiv_id(raw)
        if arxiv_id is None:
            return {"input": raw, "status": "invalid"}
        name = arxiv_id
        try:
            existing_name = config_writer.add_paper(cfg.source_path, name, arxiv_id)
        except OSError as e:
            # A failed write on one line of a batch shouldn't 500 the whole
            # request and strand already-written lines without a worker trigger.
            return {"input": raw, "status": "error", "detail": str(e)}
        if existing_name is not None:
            return {"input": raw, "status": "duplicate", "existing_name": existing_name}
        # In-place append: icfg.papers (the worker's view) is the same list object
        # by reference (see Config.for_ingest), so this is immediately visible to
        # pending_papers() without any config reload.
        cfg.papers.append(Paper(name=name, arxiv_id=arxiv_id))
        return {"input": raw, "status": "queued", "name": name}

    @app.post("/api/admin/papers")
    def add_papers(req: AddPapersRequest):
        results = [_add_one_paper(raw) for raw in req.arxiv_ids_or_urls]
        if any(r["status"] == "queued" for r in results):
            worker.trigger()
        return {"results": results}

    @app.delete("/api/admin/papers/{paper_id}")
    def remove_paper(paper_id: str):
        if manifest.get(paper_id) is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        # config_writer goes first, deliberately: it's the step most likely to fail
        # (parsing/rewriting a config.yaml a human may have hand-edited into
        # something ruamel chokes on), and it's also the step that decides whether
        # this paper can ever be re-ingested. Doing it before anything destructive
        # means a failure here leaves chunks/manifest/files untouched instead of
        # half-deleting the paper and then having it silently reappear on the next
        # rescan (config.yaml still lists it, but the manifest no longer does).
        config_writer.remove_paper(cfg.source_path, paper_id)
        # In-place slice assignment (not `cfg.papers = [...]`) — must keep the same
        # list object so icfg.papers stays aliased, or the removed paper silently
        # reappears as "pending" on the next rescan/trigger.
        cfg.papers[:] = [p for p in cfg.papers if p.name != paper_id]
        remove_paper_chunks(collection, paper_id)
        manifest.remove(paper_id)
        annotations.remove_paper(paper_id)
        Path(cfg.paths.pdf_dir, f"{paper_id}.pdf").unlink(missing_ok=True)
        Path(cfg.paths.markdown_dir, f"{paper_id}.md").unlink(missing_ok=True)
        Path(cfg.paths.markdown_dir, f"{paper_id}_display.md").unlink(missing_ok=True)
        shutil.rmtree(Path(cfg.paths.markdown_dir, f"{paper_id}.assets"), ignore_errors=True)
        return Response(status_code=204)

    # ---- chat sessions ----
    @app.get("/api/chats")
    def list_chats():
        return chats.list_all()

    @app.post("/api/chats")
    def new_chat():
        return chats.create()

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str):
        c = chats.get(chat_id)
        return c or JSONResponse({"error": "not found"}, status_code=404)

    @app.delete("/api/chats/{chat_id}")
    def delete_chat(chat_id: str):
        chats.delete(chat_id)
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/feedback")
    def post_feedback(chat_id: str, req: FeedbackRequest):
        try:
            c = chats.set_feedback(chat_id, req.turn_index, req.vote, req.note)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return c or JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/chats/{chat_id}/stop")
    def stop_chat(chat_id: str):
        # No error on a miss (turn already finished, or never started) — the frontend
        # fires this the instant it aborts its own fetch and doesn't wait on the result.
        return {"stopped": chats.request_stop(chat_id)}

    @app.post("/api/chat/classify", response_model=ClassifyModeResponse)
    async def classify_chat_mode(req: ClassifyModeRequest):
        agent = get_agent()
        loop = asyncio.get_running_loop()
        mode, scope_size = await loop.run_in_executor(
            None,
            lambda: agent.classify_mode(
                [m.model_dump() for m in req.messages], req.tags, req.papers
            ),
        )
        return {"mode": mode, "scope_size": scope_size}

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        if req.chat_id and not chats.try_acquire(req.chat_id):
            return JSONResponse(
                {"error": "a turn is already in progress for this chat"}, status_code=409
            )
        # try_acquire above created this turn's stop Event; grab it now so `work` can
        # thread it into the LLM backend's streaming loop as `stop_check`.
        stop_event = chats.stop_event(req.chat_id) if req.chat_id else None
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: str, data: str):
            loop.call_soon_threadsafe(queue.put_nowait, {"event": event, "data": data})

        def work():
            try:
                run_turn(
                    get_agent,
                    chats,
                    req,
                    emit,
                    cfg.llm.tagging,
                    stop_check=stop_event.is_set if stop_event else None,
                )
            finally:
                if req.chat_id:
                    chats.release(req.chat_id)

        loop.run_in_executor(None, work)

        async def event_stream():
            while True:
                msg = await queue.get()
                yield msg
                if msg["event"] == "done":
                    break

        return EventSourceResponse(event_stream())

    # ---- static SPA (prod build) ----
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if not web_dist.exists():
            return PlainTextResponse(
                "Frontend not built. Run `npm --prefix web run build`, "
                "or use the Vite dev server (`npm --prefix web run dev`).",
            )
        root = web_dist.resolve()
        candidate = (web_dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(root / "index.html")

    return app


def main() -> None:
    import uvicorn

    cfg = parse_config()  # config file + draccus per-field CLI overrides (--server.port=...)
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
