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
    # One paper failing must not abort the run: the loop catches per-paper,
    # records the error, and keeps ingesting the rest.
    worker = _worker(make_config)
    papers = [SimpleNamespace(name="paper1"), SimpleNamespace(name="paper2")]
    ingested: list[str] = []

    def fake_ingest(paper, *a, **k):
        if paper.name == "paper1":
            raise RuntimeError("boom")
        ingested.append(paper.name)

    monkeypatch.setattr("server.worker.pending_papers", lambda *a, **k: papers)
    monkeypatch.setattr(
        "server.worker.build_embedder_from_config",
        lambda *a, **k: SimpleNamespace(name=lambda: "fake"),  # worker reads embedder.name()
    )
    monkeypatch.setattr("server.worker.open_collection", lambda *a, **k: object())
    monkeypatch.setattr("server.worker.ingest_paper", fake_ingest)

    worker._run()

    snap = worker.snapshot()
    assert ingested == ["paper2"]  # paper2 still ingested after paper1 blew up
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
