"""IngestionWorker: per-paper error isolation and single-flight triggering.

Drives ``_run`` directly (no thread) so the isolation loop is exercised
synchronously — the module functions it calls are stubbed per test.
"""

from __future__ import annotations

from types import SimpleNamespace

from rag.manifest import Manifest
from server.worker import IngestionWorker


def _worker(make_config):
    cfg = make_config().for_ingest()
    return IngestionWorker(cfg, Manifest(cfg.paths.rag_db))


def test_run_isolates_per_paper_failures(make_config, monkeypatch):
    # One paper failing must not abort the run. The real catch-and-continue
    # per-paper logic now lives in pipeline.run_batch (see test_pipeline.py for
    # its own dedicated tests) — this test drives the REAL run_batch (only its
    # leaves faked) specifically to prove the worker wires stop_on_error=False
    # into it correctly. A fake run_batch here wouldn't catch a regression where
    # someone flips that flag, since the fake would just ignore it.
    worker = _worker(make_config)
    papers = [SimpleNamespace(name="paper1"), SimpleNamespace(name="paper2")]
    ingested: list[str] = []

    def fake_ingest(paper, *a, **k):
        if paper.name == "paper1":
            raise RuntimeError("boom")
        ingested.append(paper.name)
        return {"paper_id": paper.name, "tags": [], "n_chunks": 1}

    monkeypatch.setattr("server.worker.pending_papers", lambda *a, **k: papers)
    monkeypatch.setattr(
        "rag.pipeline.build_embedder_from_config",
        lambda *a, **k: SimpleNamespace(name=lambda: "fake"),  # worker reads embedder.name()
    )
    monkeypatch.setattr("rag.pipeline.open_collection", lambda *a, **k: object())
    monkeypatch.setattr("rag.pipeline.ingest_paper", fake_ingest)

    worker._run()

    snap = worker.snapshot()
    assert ingested == ["paper2"]  # paper2 still ingested after paper1 blew up
    assert snap["done"] == 2  # the run advanced past both papers
    assert snap["errors"] == [{"name": "paper1", "error": "boom"}]
    assert snap["state"] == "idle"  # completed cleanly, not stuck in error


def test_run_wires_hooks_from_a_fake_run_batch(make_config, monkeypatch):
    # Narrower complement to the test above: proves the worker's own hook
    # closures (_on_start/_on_stage/_on_done) update status correctly, without
    # depending on run_batch's real internals (covered separately).
    worker = _worker(make_config)
    papers = [SimpleNamespace(name="paper1"), SimpleNamespace(name="paper2")]

    def fake_run_batch(
        cfg,
        manifest,
        batch,
        embedder=None,
        collection=None,
        on_paper_start=None,
        on_stage=None,
        on_paper_done=None,
        retag=True,
        stop_on_error=True,
    ):
        for p in batch:
            on_paper_start(p)
            if p.name == "paper1":
                on_paper_done(p, None, RuntimeError("boom"))
            else:
                on_paper_done(p, {"paper_id": p.name, "tags": [], "n_chunks": 1}, None)
        return SimpleNamespace(
            records=[], embedder=embedder or object(), collection=collection or object()
        )

    monkeypatch.setattr("server.worker.pending_papers", lambda *a, **k: papers)
    monkeypatch.setattr("server.worker.run_batch", fake_run_batch)

    worker._run()

    snap = worker.snapshot()
    assert snap["done"] == 2  # the run advanced past both papers
    assert snap["errors"] == [{"name": "paper1", "error": "boom"}]
    assert snap["state"] == "idle"  # completed cleanly, not stuck in error


def test_run_with_no_pending_stays_idle(make_config, monkeypatch):
    worker = _worker(make_config)
    monkeypatch.setattr("server.worker.pending_papers", lambda *a, **k: [])
    worker._run()
    snap = worker.snapshot()
    assert snap["state"] == "idle"
    assert snap["total"] == 0


def test_trigger_is_single_flight(make_config):
    # A live run must block a second trigger, or two threads race the same papers.
    worker = _worker(make_config)
    worker._thread = SimpleNamespace(is_alive=lambda: True)  # type: ignore[assignment]
    assert worker.trigger() is False


def test_trigger_starts_a_run_when_idle(make_config, monkeypatch):
    worker = _worker(make_config)
    monkeypatch.setattr("server.worker.pending_papers", lambda *a, **k: [])
    assert worker.trigger() is True
    assert worker._thread is not None
    worker._thread.join(timeout=5)
    assert worker.snapshot()["state"] == "idle"


def test_run_picks_up_a_paper_added_mid_run(make_config, monkeypatch):
    # Reproduces the "add paper while a run is in flight" scenario: pending_papers
    # returns paper1 on the first check; a paper2 "arrives" while paper1 is being
    # ingested (simulated by the fake run_batch flipping a flag pending_papers
    # reads). trigger() is single-flight, so nothing else would re-scan for
    # paper2 — the worker's own outer re-poll loop (untouched by run_batch) must
    # notice before going idle.
    worker = _worker(make_config)
    paper2_arrived = {"value": False}
    ingested: list[str] = []
    seen_embedders: list[object] = []

    def fake_pending(cfg, manifest):
        papers = [SimpleNamespace(name="paper1")]
        if paper2_arrived["value"]:
            papers.append(SimpleNamespace(name="paper2"))
        return [p for p in papers if p.name not in ingested]

    def fake_run_batch(
        cfg,
        manifest,
        batch,
        embedder=None,
        collection=None,
        on_paper_start=None,
        on_stage=None,
        on_paper_done=None,
        retag=True,
        stop_on_error=True,
    ):
        embedder = embedder or SimpleNamespace(name=lambda: "f")
        collection = collection or object()
        seen_embedders.append(embedder)
        for p in batch:
            on_paper_start(p)
            ingested.append(p.name)
            if p.name == "paper1":
                paper2_arrived["value"] = True  # simulate the admin route firing mid-batch
            on_paper_done(p, {"paper_id": p.name, "tags": [], "n_chunks": 1}, None)
        return SimpleNamespace(records=[], embedder=embedder, collection=collection)

    monkeypatch.setattr("server.worker.pending_papers", fake_pending)
    monkeypatch.setattr("server.worker.run_batch", fake_run_batch)

    worker._run()

    assert ingested == ["paper1", "paper2"]  # paper2 wasn't stranded until a manual rescan
    assert worker.snapshot()["state"] == "idle"
    assert len(seen_embedders) == 2 and seen_embedders[0] is seen_embedders[1]  # built once, reused


def test_run_does_not_retry_a_permanently_failing_paper_forever(make_config, monkeypatch):
    # A paper that never reaches the manifest (e.g. a bad arxiv_id, or any ingest_paper
    # failure) keeps showing up in pending_papers() forever — the mid-run re-check must
    # not spin on it indefinitely.
    worker = _worker(make_config)
    attempts: list[str] = []

    def fake_run_batch(
        cfg,
        manifest,
        batch,
        embedder=None,
        collection=None,
        on_paper_start=None,
        on_stage=None,
        on_paper_done=None,
        retag=True,
        stop_on_error=True,
    ):
        for p in batch:
            attempts.append(p.name)
            on_paper_start(p)
            on_paper_done(p, None, RuntimeError("always fails"))
        return SimpleNamespace(
            records=[], embedder=embedder or object(), collection=collection or object()
        )

    monkeypatch.setattr(
        "server.worker.pending_papers", lambda *a, **k: [SimpleNamespace(name="bad-paper")]
    )
    monkeypatch.setattr("server.worker.run_batch", fake_run_batch)

    worker._run()  # must terminate, not loop forever

    assert attempts == ["bad-paper"]  # attempted exactly once this run, not retried in-loop
    snap = worker.snapshot()
    assert snap["state"] == "idle"
    assert snap["errors"] == [{"name": "bad-paper", "error": "always fails"}]
