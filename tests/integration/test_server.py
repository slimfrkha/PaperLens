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

from rag.config import Paper
from rag.index import open_collection
from rag.llm import Usage
from rag.manifest import Manifest
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


class _RefStartAgent:
    """Shared fake ChatAgent: echoes ref_start into the returned citation's ref, so tests
    can assert ref numbering continues correctly across turns/edits."""

    def __init__(self, *a, **k):
        pass

    def run(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_trace=None,
        ref_start=0,
        per_paper=False,
        stop_check=None,
    ):
        ref = f"r{ref_start + 1}"
        text = f"See [{ref}]."
        on_text(text)
        return text, [{"ref": ref, "paper_id": "p", "title": "P"}], Usage(10, 5)


class _EchoAgent:
    """Shared fake ChatAgent: emits one token "answer", empty citations, Usage(10, 5)."""

    def __init__(self, *a, **k):
        pass

    def run(
        self,
        messages,
        tags,
        papers,
        on_text,
        on_trace=None,
        ref_start=0,
        per_paper=False,
        stop_check=None,
    ):
        on_text("answer")
        return "answer", [], Usage(10, 5)


@pytest.fixture
def patch_agent_seam(monkeypatch):
    """Factory: patch server.main's get_agent() seam (build_llm/build_reranker/Searcher/
    ChatAgent) so /api/chat never builds real models or hits a network. Pass `chat_agent`
    to substitute ChatAgent (omit to leave it unpatched — e.g. when `build_llm` itself is
    made to raise before ChatAgent would ever be reached); pass `build_llm` to override
    the trivial default stand-in. Returns the `server.main` module object."""

    def _patch(chat_agent=None, build_llm=None):
        import importlib

        # `import server.main` binds the re-exported main() function, not the module;
        # import_module returns the real module so the agent seam can be stubbed.
        main_mod = importlib.import_module("server.main")
        monkeypatch.setattr(main_mod, "build_llm", build_llm or (lambda *a, **k: object()))
        monkeypatch.setattr(main_mod, "build_reranker", lambda *a, **k: object())
        monkeypatch.setattr(main_mod, "Searcher", lambda *a, **k: object())
        if chat_agent is not None:
            monkeypatch.setattr(main_mod, "ChatAgent", chat_agent)
        return main_mod

    return _patch


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


def test_chat_streams_token_citations_done(make_config, patch_agent_seam):
    # Pin the SSE contract: a scripted turn emits token(s), then citations, then a
    # terminal done — the cross-thread emit -> queue -> event_stream machinery.
    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            on_text("foo")
            on_text("bar")
            if on_trace:
                on_trace({"type": "action", "query": "q"})
            return "foobar", [{"ref": "r1", "paper_id": "p", "title": "P"}], Usage(10, 5)

    patch_agent_seam(chat_agent=FakeAgent)

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


def test_chat_per_paper_true_reaches_the_agent(make_config, patch_agent_seam):
    received: dict = {}

    class RecordingAgent:
        def __init__(self, *a, **k):
            pass

        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            received["per_paper"] = per_paper
            on_text("ok")
            return "ok", [], Usage(10, 5)

    patch_agent_seam(chat_agent=RecordingAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "per_paper": True},
    )

    assert resp.status_code == 200
    assert received["per_paper"] is True


def test_chat_without_per_paper_field_defaults_false(make_config, patch_agent_seam):
    # A request body predating this field (no "per_paper" key at all) must still 200 and
    # default server-side to False — ChatRequest.per_paper's pydantic default covers it.
    received: dict = {}

    class RecordingAgent:
        def __init__(self, *a, **k):
            pass

        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            received["per_paper"] = per_paper
            on_text("ok")
            return "ok", [], Usage(10, 5)

    patch_agent_seam(chat_agent=RecordingAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200
    assert received["per_paper"] is False


def test_chat_streams_usage_event_and_persists_it(make_config, patch_agent_seam):
    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            on_text("answer")
            return "answer", [], Usage(42, 7)

    patch_agent_seam(chat_agent=FakeAgent)

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


def test_chat_continues_citation_numbering_across_turns(make_config, patch_agent_seam):
    # Reproduces the reported bug: a second question in the same chat must not
    # restart citation numbering at r1 — main.py must offset ref numbering by
    # what's already stored for this chat_id.
    patch_agent_seam(chat_agent=_RefStartAgent)

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


def test_chat_edit_index_truncates_and_restarts_ref_numbering(make_config, patch_agent_seam):
    # Edit the FIRST turn of a two-turn chat: the second exchange (and its r2 citation)
    # must be dropped, and the edited turn's citation must renumber from r1, not r3 —
    # ref_start has to be read off the truncated history, not the pre-edit one.
    patch_agent_seam(chat_agent=_RefStartAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q1"}], "chat_id": chat_id}
    )
    client.post(
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
    before = client.get(f"/api/chats/{chat_id}").json()
    assert len(before["messages"]) == 4

    edited = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "q1 edited"}],
            "chat_id": chat_id,
            "edit_index": 0,
        },
    )
    cits = json.loads(next(e["data"] for e in _parse_sse(edited.text) if e["event"] == "citations"))
    assert cits[0]["ref"] == "r1"  # renumbered from scratch, not r3

    after = client.get(f"/api/chats/{chat_id}").json()
    assert [m["content"] for m in after["messages"]] == ["q1 edited", "See [r1]."]


def test_chat_edit_index_mid_conversation_keeps_earlier_refs(make_config, patch_agent_seam):
    # Edit the SECOND turn of a two-turn chat: the first exchange (and its r1
    # citation) is retained, so the edited turn's ref must continue at r2.
    patch_agent_seam(chat_agent=_RefStartAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q1"}], "chat_id": chat_id}
    )
    client.post(
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

    edited = client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "See [r1]."},
                {"role": "user", "content": "q2 edited"},
            ],
            "chat_id": chat_id,
            "edit_index": 2,
        },
    )
    cits = json.loads(next(e["data"] for e in _parse_sse(edited.text) if e["event"] == "citations"))
    assert cits[0]["ref"] == "r2"  # continues past the retained r1, doesn't collide

    after = client.get(f"/api/chats/{chat_id}").json()
    assert [m["content"] for m in after["messages"]] == [
        "q1",
        "See [r1].",
        "q2 edited",
        "See [r2].",
    ]


def test_chat_edit_index_on_assistant_turn_surfaces_sse_error(make_config, patch_agent_seam):
    # truncate_at's ValueError (bad index) must come back as an SSE `error` event, not
    # an HTTP 4xx — the streaming response has already started by the time it's raised.
    patch_agent_seam(chat_agent=_EchoAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q"}], "chat_id": chat_id}
    )

    resp = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "q2"}], "chat_id": chat_id, "edit_index": 1},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(e["event"] == "error" for e in events)
    assert events[-1]["event"] == "done"


def test_chat_route_releases_guard_when_agent_build_fails(make_config, patch_agent_seam):
    # get_agent() is the first-touch lazy model build and can throw (bad key, cold
    # cloud client). That must not leak the single-flight guard: a failed turn has to
    # release its chat_id, or every later request against it gets stuck behind a
    # permanent 409 with nothing actually running.
    def boom(*a, **k):
        raise RuntimeError("boom: agent build failed")

    patch_agent_seam(build_llm=boom)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]

    first = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q"}], "chat_id": chat_id}
    )
    assert first.status_code == 200  # streams an SSE error rather than a bare 500
    events = _parse_sse(first.text)
    assert any(e["event"] == "error" for e in events)
    assert events[-1]["event"] == "done"

    # The guard must be released even though the turn failed — a second request on
    # the same chat_id must not be rejected with 409.
    second = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q2"}], "chat_id": chat_id}
    )
    assert second.status_code != 409


def test_chat_route_rejects_concurrent_turn_on_same_chat(make_config, patch_agent_seam):
    # The single-flight guard exists so an edit's truncate+append can't interleave
    # with another in-flight turn on the same chat. Force real overlap: block the
    # first request's agent mid-turn, confirm a second request on the same chat_id
    # is rejected with 409 while the first is still running, then let it finish.
    import threading

    started = threading.Event()
    release_agent = threading.Event()

    class SlowFakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            started.set()
            assert release_agent.wait(timeout=5), "test deadlocked waiting for release"
            on_text("answer")
            return "answer", [], Usage(10, 5)

    patch_agent_seam(chat_agent=SlowFakeAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    results: dict = {}

    def first():
        results["first"] = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "q"}], "chat_id": chat_id}
        )

    t = threading.Thread(target=first)
    t.start()
    assert started.wait(timeout=5), "first request never reached the agent"

    second = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q2"}], "chat_id": chat_id}
    )
    assert second.status_code == 409

    release_agent.set()
    t.join(timeout=5)
    assert results["first"].status_code == 200
    assert "done" in results["first"].text


def test_stop_route_signals_stop_check_and_releases_the_guard(make_config, patch_agent_seam):
    # The /stop route must reach the in-flight turn's stop_check (not just be a no-op),
    # and letting the agent return promptly on it must release the single-flight guard —
    # otherwise a follow-up request on the same chat would be stuck behind a stale 409
    # for as long as the (now-abandoned) turn takes to finish on its own.
    import threading
    import time

    started = threading.Event()

    class PollingFakeAgent:
        def __init__(self, *a, **k):
            pass

        def run(
            self,
            messages,
            tags,
            papers,
            on_text,
            on_trace=None,
            ref_start=0,
            per_paper=False,
            stop_check=None,
        ):
            started.set()
            on_text("partial")
            deadline = time.monotonic() + 5
            while not (stop_check and stop_check()):
                assert time.monotonic() < deadline, "test deadlocked waiting for stop_check"
                time.sleep(0.01)
            return "partial", [], Usage(10, 5)

    patch_agent_seam(chat_agent=PollingFakeAgent)

    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    chat_id = client.post("/api/chats").json()["id"]
    results: dict = {}

    def first():
        results["first"] = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "q"}], "chat_id": chat_id}
        )

    t = threading.Thread(target=first)
    t.start()
    assert started.wait(timeout=5), "first request never reached the agent"

    stop_resp = client.post(f"/api/chats/{chat_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json() == {"stopped": True}

    t.join(timeout=5)
    assert results["first"].status_code == 200
    saved = client.get(f"/api/chats/{chat_id}").json()
    assert saved["messages"][-1] == {"role": "assistant", "content": "partial"}

    # The turn actually finished (not force-unlocked out from under it) -> a new
    # request on the same chat isn't rejected.
    second = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "q2"}], "chat_id": chat_id}
    )
    assert second.status_code != 409


def test_stop_route_no_op_when_chat_not_in_flight(make_config, patch_agent_seam):
    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.post("/api/chats/nonexistent/stop")
    assert resp.status_code == 200
    assert resp.json() == {"stopped": False}


def test_feedback_route_sets_clears_and_rejects_invalid_index(make_config, patch_agent_seam):
    patch_agent_seam(chat_agent=_EchoAgent)

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


def test_annotation_routes_create_list_update_delete(make_config):
    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    created = client.post(
        "/api/papers/paper1/annotations",
        json={
            "snippet": "a passage worth remembering",
            "section_title": "2.1 Attention",
            "section_slug": "2-1-attention",
            "note": "check this against Table 3",
        },
    )
    assert created.status_code == 200
    annotation = created.json()
    assert annotation["note"] == "check this against Table 3"
    assert annotation["section_slug"] == "2-1-attention"

    listed = client.get("/api/papers/paper1/annotations")
    assert [a["id"] for a in listed.json()] == [annotation["id"]]
    assert client.get("/api/papers/other-paper/annotations").json() == []

    updated = client.patch(
        f"/api/papers/paper1/annotations/{annotation['id']}", json={"note": "updated note"}
    )
    assert updated.status_code == 200
    assert updated.json()["note"] == "updated note"

    deleted = client.delete(f"/api/papers/paper1/annotations/{annotation['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True}
    assert client.get("/api/papers/paper1/annotations").json() == []


def test_annotation_update_404_for_unknown_annotation(make_config):
    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.patch("/api/papers/paper1/annotations/missing-id", json={"note": "x"})
    assert resp.status_code == 404


def test_annotation_delete_returns_false_for_unknown_annotation(make_config):
    cfg = make_config()
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(cfg))

    resp = client.delete("/api/papers/paper1/annotations/missing-id")
    assert resp.status_code == 200
    assert resp.json() == {"ok": False}


def _admin_app(make_config, tmp_path, papers_yaml: str = "papers: []\n"):
    """A create_app() instance wired to a real config.yaml on disk, so the
    admin add/remove-paper routes (which rewrite that file via config_writer) have
    something real to read/write — unlike make_config()'s in-memory-only Config."""
    cfg = make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"collection: {cfg.collection}\n{papers_yaml}")
    cfg.source_path = config_path
    Path(cfg.paths.web_dist).mkdir(parents=True, exist_ok=True)
    return cfg, TestClient(create_app(cfg))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2412.19437", "2412.19437"),
        ("https://arxiv.org/abs/2412.19437", "2412.19437"),
        ("https://arxiv.org/pdf/2412.19437", "2412.19437"),
        ("https://arxiv.org/pdf/2412.19437.pdf", "2412.19437"),
        ("https://arxiv.org/abs/2412.19437v2", "2412.19437"),
        ("  2412.19437  ", "2412.19437"),
    ],
)
def test_add_paper_route_normalizes_id_or_url(make_config, tmp_path, raw, expected):
    cfg, client = _admin_app(make_config, tmp_path)
    resp = client.post("/api/admin/papers", json={"arxiv_id_or_url": raw})
    assert resp.status_code == 200
    assert resp.json() == {"queued": True, "name": expected}
    assert [p.name for p in cfg.papers] == [expected]


@pytest.mark.parametrize(
    "raw",
    [
        "hep-th/9901001",  # old-style slashed id — would corrupt name-derived file paths
        "not-an-arxiv-id",
        "",
    ],
)
def test_add_paper_route_rejects_unrecognizable_input(make_config, tmp_path, raw):
    cfg, client = _admin_app(make_config, tmp_path)
    resp = client.post("/api/admin/papers", json={"arxiv_id_or_url": raw})
    assert resp.status_code == 400
    assert cfg.papers == []


def test_add_paper_route_dedups_by_arxiv_id_against_curated_entry(make_config, tmp_path):
    # Pre-curated under a human-chosen name that won't textually match a
    # UI-generated name == arxiv_id.
    cfg, client = _admin_app(
        make_config,
        tmp_path,
        papers_yaml='papers:\n  - { name: deepseek-v3, arxiv_id: "2412.19437" }\n',
    )
    # A route-created Config doesn't re-parse the file it was handed source_path
    # for, so seed cfg.papers to match what a real load_config() would have done.
    cfg.papers.append(Paper(name="deepseek-v3", arxiv_id="2412.19437"))

    resp = client.post("/api/admin/papers", json={"arxiv_id_or_url": "2412.19437"})

    assert resp.status_code == 409
    assert resp.json() == {"error": "already curated as deepseek-v3"}
    assert [p.name for p in cfg.papers] == ["deepseek-v3"]  # no duplicate appended


def test_remove_paper_route_cleans_manifest_chunks_files_and_config(make_config, tmp_path):
    cfg, client = _admin_app(
        make_config,
        tmp_path,
        papers_yaml='papers:\n  - { name: paper-a, arxiv_id: "2412.19437" }\n',
    )
    cfg.papers.append(Paper(name="paper-a", arxiv_id="2412.19437"))

    manifest = Manifest(cfg.paths.rag_db)
    manifest.upsert({"paper_id": "paper-a", "title": "Paper A", "tags": [], "n_chunks": 1})

    Path(cfg.paths.pdf_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.markdown_dir).mkdir(parents=True, exist_ok=True)
    pdf_path = Path(cfg.paths.pdf_dir) / "paper-a.pdf"
    md_path = Path(cfg.paths.markdown_dir) / "paper-a.md"
    pdf_path.write_bytes(b"%PDF")
    md_path.write_text("## Paper A")

    collection = open_collection(cfg.paths.rag_db, cfg.collection)
    collection.upsert(
        ids=["chunk-1"],
        embeddings=[[0.0] * 8],
        documents=["a passage"],
        metadatas=[{"paper_id": "paper-a"}],
    )

    client.post(
        "/api/papers/paper-a/annotations",
        json={"snippet": "a passage worth remembering", "section_title": "", "note": ""},
    )
    assert len(client.get("/api/papers/paper-a/annotations").json()) == 1

    resp = client.delete("/api/admin/papers/paper-a")

    assert resp.status_code == 204
    assert manifest.get("paper-a") is None
    assert collection.get(where={"paper_id": "paper-a"}, include=[])["ids"] == []
    assert not pdf_path.exists()
    assert not md_path.exists()
    assert [p.name for p in cfg.papers] == []
    assert "paper-a" not in cfg.source_path.read_text()
    assert client.get("/api/papers/paper-a/annotations").json() == []


def test_remove_paper_route_404_for_unknown_paper(make_config, tmp_path):
    cfg, client = _admin_app(make_config, tmp_path)
    resp = client.delete("/api/admin/papers/does-not-exist")
    assert resp.status_code == 404
