"""FastAPI app: chat (SSE), papers, tags, admin status, and static SPA."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
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
from .chats import ChatStore, generate_name
from .schemas import (
    AddPaperRequest,
    AnnotationCreate,
    AnnotationUpdate,
    ChatRequest,
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
            agent.searcher.search("warm up", k=1, candidates=1)
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
        path = Path(cfg.paths.markdown_dir) / f"{paper_id}.md"
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        md = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.DOTALL)
        rec = manifest.get(paper_id) or {}
        return {
            "paper_id": paper_id,
            "title": rec.get("title", paper_id),
            "tags": rec.get("tags", []),
            "arxiv_id": rec.get("arxiv_id"),
            "markdown": md,
        }

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

    @app.post("/api/admin/papers")
    def add_paper(req: AddPaperRequest):
        arxiv_id = _normalize_arxiv_id(req.arxiv_id_or_url)
        if arxiv_id is None:
            return JSONResponse({"error": "not a recognizable arXiv id or URL"}, status_code=400)
        name = arxiv_id
        # config_writer's dedup is by arxiv_id (not name) against every existing
        # papers: entry, so a paper already curated under a human-chosen name (e.g.
        # `deepseek-v3`) is still caught even though this route's generated `name`
        # won't match it textually. Only on success does cfg.papers get the new
        # entry — an unconditional append here would double-append (and
        # double-ingest) on a race between two near-simultaneous requests for the
        # same arxiv_id.
        existing_name = config_writer.add_paper(cfg.source_path, name, arxiv_id)
        if existing_name is not None:
            return JSONResponse({"error": f"already curated as {existing_name}"}, status_code=409)
        # In-place append: icfg.papers (the worker's view) is the same list object
        # by reference (see Config.for_ingest), so this is immediately visible to
        # pending_papers() without any config reload.
        cfg.papers.append(Paper(name=name, arxiv_id=arxiv_id))
        worker.trigger()
        return {"queued": True, "name": name}

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
            c = chats.set_feedback(chat_id, req.index, req.vote, req.note)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return c or JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        if req.chat_id and not chats.try_acquire(req.chat_id):
            return JSONResponse(
                {"error": "a turn is already in progress for this chat"}, status_code=409
            )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: str, data: str):
            loop.call_soon_threadsafe(queue.put_nowait, {"event": event, "data": data})

        def work():
            trace_entries: list = []

            def on_trace(e):
                trace_entries.append(e)
                emit("trace", json.dumps(e))

            try:
                # Built here, not before scheduling this thread: get_agent() is the
                # first-touch lazy model build and can throw (bad key, cold cloud
                # client) — done outside this try/finally, a failure here would never
                # release the try_acquire guard above, wedging the chat permanently.
                agent = get_agent()
                # An edit-and-resume truncates the stored tail before this turn is
                # computed, so ref_start below reads only the retained history — the
                # try_acquire guard above ensures no concurrent request can interleave.
                if req.edit_index is not None and req.chat_id:
                    chats.truncate_at(req.chat_id, req.edit_index)
                # Existing chats already carry ref-numbered citations for prior turns —
                # offset this turn's numbering past them so a follow-up question
                # continues (r4, r5, ...) instead of restarting at r1 and colliding
                # with refs already shown for a different paper earlier in the chat.
                existing = chats.get(req.chat_id) if req.chat_id else None
                ref_start = (
                    sum(len(c) for c in existing.get("citations", []) if c) if existing else 0
                )
                # Spans the whole turn (retrieval + rerank + LLM + faithfulness check),
                # not just LLM think time — that's the wait the user actually feels.
                t0 = time.perf_counter()
                text, citations, usage = agent.run(
                    [m.model_dump() for m in req.messages],
                    req.tags,
                    req.papers,
                    on_text=lambda t: emit("token", t),
                    on_trace=on_trace,
                    ref_start=ref_start,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                usage_payload = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "latency_ms": latency_ms,
                }
                emit("citations", json.dumps(citations))
                emit("usage", json.dumps(usage_payload))
                # Persist the turn (append user + assistant) and name new sessions.
                if req.chat_id and req.messages:
                    name = None
                    if not existing or not existing.get("name"):
                        name = generate_name(req.messages[0].content, cfg.llm.tagging)
                    saved = chats.append_turn(
                        req.chat_id,
                        req.messages[-1].content,
                        text,
                        citations,
                        trace_entries,
                        usage_payload,
                        name=name,
                    )
                    emit("meta", json.dumps({"chat_id": saved["id"], "name": saved["name"]}))
            except Exception as e:  # surface errors to the client
                emit("error", f"{type(e).__name__}: {e}")
            finally:
                emit("done", "")
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
