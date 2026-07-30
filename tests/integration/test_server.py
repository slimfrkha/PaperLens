"""HTTP layer: the SPA path-traversal guard and the /api/chat SSE contract.

Uses a TestClient over a real app. The chat test stubs the agent seam in
``server.main`` so the SSE machinery is exercised fully offline (no models, no
API). The client is built without entering the lifespan so the startup model
warmer never runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.llm import Usage
from server.main import create_app


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE body into ``[{"event", "data"}]`` events."""
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        ev: dict = {}
        for line in block.split("\n"):
            key, _, val = line.partition(":")
            if key == "event":
                ev["event"] = val.strip()
            elif key == "data":
                ev["data"] = val[1:] if val.startswith(" ") else val  # SSE drops one lead space
        if "event" in ev:
            events.append(ev)
    return events


@pytest.fixture
def spa_client(make_config):
    cfg = make_config()
    web_dist = Path(cfg.paths.web_dist)
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "index.html").write_text("<html>INDEX</html>")
    (web_dist / "app.js").write_text("ASSET")
    (web_dist.parent / "secret.txt").write_text("TOPSECRET")  # sits outside web_dist
    return TestClient(create_app(cfg))


# Single-level escape (secret.txt sits one dir above web_dist), percent-encoded so
# httpx can't collapse the `..` client-side before it reaches the server — each of
# these decodes to full_path="../secret.txt" and leaks against the unpatched code.
@pytest.mark.parametrize(
    "attack",
    [
        "/..%2fsecret.txt",
        "/%2e%2e/secret.txt",
        "/%2e%2e%2fsecret.txt",
    ],
)
def test_spa_blocks_path_traversal(spa_client, attack):
    # A path escaping web_dist must never serve the file — it falls back to
    # index.html. Before the resolve()/is_relative_to guard this leaked arbitrary
    # files off disk.
    resp = spa_client.get(attack)
    assert resp.status_code == 200
    assert "TOPSECRET" not in resp.text


def test_spa_serves_a_legit_asset(spa_client):
    # The guard must not over-block real files inside web_dist.
    resp = spa_client.get("/app.js")
    assert resp.status_code == 200
    assert resp.text == "ASSET"


def test_chat_streams_token_citations_done(make_config, monkeypatch):
    # Pin the SSE contract: a scripted turn emits token(s), then citations, then a
    # terminal done — the cross-thread emit -> queue -> event_stream machinery.
    import importlib

    # `import server.main` binds the re-exported main() function, not the module;
    # import_module returns the real module so the agent seam can be stubbed.
    main_mod = importlib.import_module("server.main")

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0):
            on_text("foo")
            on_text("bar")
            if on_trace:
                on_trace({"type": "action", "query": "q"})
            return "foobar", [{"ref": "r1", "paper_id": "p", "title": "P"}], Usage(10, 5)

    monkeypatch.setattr(main_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "build_reranker", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "Searcher", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "ChatAgent", FakeAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    kinds = [e["event"] for e in events]
    assert kinds[-1] == "done"  # stream terminates on done
    assert kinds.count("token") == 2
    assert kinds.index("token") < kinds.index("citations") < kinds.index("done")

    tokens = "".join(e["data"] for e in events if e["event"] == "token")
    assert tokens == "foobar"
    cits = json.loads(next(e["data"] for e in events if e["event"] == "citations"))
    assert cits[0]["ref"] == "r1"


def test_chat_streams_usage_event_and_persists_it(make_config, monkeypatch):
    import importlib

    main_mod = importlib.import_module("server.main")

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0):
            on_text("answer")
            return "answer", [], Usage(42, 7)

    monkeypatch.setattr(main_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "build_reranker", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "Searcher", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "ChatAgent", FakeAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    resp = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "chat_id": chat_id},
    )

    events = _parse_sse(resp.text)
    usage_event = json.loads(next(e["data"] for e in events if e["event"] == "usage"))
    assert usage_event["input_tokens"] == 42
    assert usage_event["output_tokens"] == 7
    assert usage_event["latency_ms"] >= 0

    saved = client.get(f"/api/chats/{chat_id}").json()
    assert saved["usage"][1]["input_tokens"] == 42


def test_chat_continues_citation_numbering_across_turns(make_config, monkeypatch):
    # Reproduces the reported bug: a second question in the same chat must not
    # restart citation numbering at r1 — main.py must offset ref numbering by
    # what's already stored for this chat_id.
    import importlib

    main_mod = importlib.import_module("server.main")

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0):
            ref = f"r{ref_start + 1}"
            text = f"See [{ref}]."
            on_text(text)
            return text, [{"ref": ref, "paper_id": "p", "title": "P"}], Usage(10, 5)

    monkeypatch.setattr(main_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "build_reranker", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "Searcher", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "ChatAgent", FakeAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]

    resp1 = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "q1"}], "chat_id": chat_id},
    )
    cits1 = json.loads(next(e["data"] for e in _parse_sse(resp1.text) if e["event"] == "citations"))
    assert cits1[0]["ref"] == "r1"

    resp2 = client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "See [r1]."},
                {"role": "user", "content": "q2"},
            ],
            "chat_id": chat_id,
        },
    )
    cits2 = json.loads(next(e["data"] for e in _parse_sse(resp2.text) if e["event"] == "citations"))
    assert cits2[0]["ref"] == "r2"


def test_feedback_route_sets_clears_and_rejects_invalid_index(make_config, monkeypatch):
    import importlib

    main_mod = importlib.import_module("server.main")

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(self, messages, tags, papers, on_text, on_trace=None, ref_start=0):
            on_text("answer")
            return "answer", [], Usage(10, 5)

    monkeypatch.setattr(main_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "build_reranker", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "Searcher", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "ChatAgent", FakeAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q"}], "chat_id": chat_id}
    )
    # messages[0] is the user turn, messages[1] is the assistant turn.

    resp = client.post(
        f"/api/chats/{chat_id}/feedback", json={"index": 1, "vote": "up", "note": "nice cite"}
    )
    assert resp.status_code == 200
    saved = resp.json()["feedback"][1]
    assert saved["vote"] == "up"
    assert saved["note"] == "nice cite"

    cleared = client.post(f"/api/chats/{chat_id}/feedback", json={"index": 1})
    assert cleared.json()["feedback"][1] is None

    # index 0 is the user turn — feedback only applies to assistant turns.
    rejected = client.post(f"/api/chats/{chat_id}/feedback", json={"index": 0, "vote": "up"})
    assert rejected.status_code == 400

    out_of_range = client.post(f"/api/chats/{chat_id}/feedback", json={"index": 99, "vote": "up"})
    assert out_of_range.status_code == 400


def test_feedback_route_404_for_unknown_chat(make_config):
    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.post("/api/chats/missing/feedback", json={"index": 0, "vote": "up"})
    assert resp.status_code == 404
