"""FastAPI app: chat (SSE), papers, tags, admin status, and static SPA."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from rag.config import BM25Cfg, Config, parse_config
from rag.index import open_collection
from rag.llm import build_llm
from rag.manifest import Manifest
from rag.pipeline import pending_papers
from rag.reranker import build_reranker
from rag.search import Searcher

from .agent import ChatAgent
from .chats import ChatStore, generate_name
from .schemas import ChatRequest
from .worker import IngestionWorker


def create_app(cfg: Config) -> FastAPI:
    web_dist = Path(cfg.paths.web_dist)
    manifest = Manifest(cfg.paths.rag_db)
    chats = ChatStore(cfg.paths.chat_history)
    icfg = cfg.for_ingest()  # ingestion-only view for the worker + pending-paper checks
    worker = IngestionWorker(icfg, manifest)
    # Ensure the collection exists so the chat Searcher can open it even when the
    # DB is still empty (ingestion creates it too, but chat may be hit first).
    open_collection(cfg.paths.rag_db, cfg.collection)

    # Chat models (embedder + reranker + LLM) are heavy — build once on first use.
    # The lock keeps the build single-shot when the startup warmer (below) and an
    # early /api/chat race for it.
    lazy: dict = {"agent": None}
    lazy_lock = threading.Lock()

    def get_agent() -> ChatAgent:
        if lazy["agent"] is None:
            with lazy_lock:
                if lazy["agent"] is None:
                    searcher = Searcher(
                        db_dir=cfg.paths.rag_db,
                        collection=cfg.collection,
                        embedder_model=cfg.embedding.model,
                        reranker=build_reranker(cfg.reranker, llm=build_llm(cfg.llm.chat)),
                        sparse_enabled=cfg.sparse.enabled,
                        bm25_k1=cfg.sparse.k1 if isinstance(cfg.sparse, BM25Cfg) else 1.5,
                        bm25_b=cfg.sparse.b if isinstance(cfg.sparse, BM25Cfg) else 0.75,
                        rrf_k=cfg.sparse.rrf_k,
                        fetch_multiplier=cfg.sparse.fetch_multiplier,
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

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        agent = get_agent()
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
                text, citations = agent.run(
                    [m.model_dump() for m in req.messages],
                    req.tags,
                    req.papers,
                    on_text=lambda t: emit("token", t),
                    on_trace=on_trace,
                )
                emit("citations", json.dumps(citations))
                # Persist the turn (append user + assistant) and name new sessions.
                if req.chat_id and req.messages:
                    existing = chats.get(req.chat_id)
                    name = None
                    if not existing or not existing.get("name"):
                        name = generate_name(req.messages[0].content, cfg.llm.tagging)
                    saved = chats.append_turn(
                        req.chat_id,
                        req.messages[-1].content,
                        text,
                        citations,
                        trace_entries,
                        name=name,
                    )
                    emit("meta", json.dumps({"chat_id": saved["id"], "name": saved["name"]}))
            except Exception as e:  # surface errors to the client
                emit("error", f"{type(e).__name__}: {e}")
            finally:
                emit("done", "")

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
