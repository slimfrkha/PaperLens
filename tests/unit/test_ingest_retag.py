"""ingest.retag(): per-paper degrade-on-failure, doesn't abort the whole run."""

from __future__ import annotations

from pathlib import Path

from rag import ingest
from rag.manifest import Manifest


def _seed(cfg, tmp_path, paper_id: str) -> None:
    md_dir = Path(cfg.paths.markdown_dir)
    md_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / f"{paper_id}.md").write_text(f"## {paper_id}\n\n## Abstract\nx")


def test_retag_continues_past_a_failing_paper(monkeypatch, make_config, tmp_path):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    for paper_id in ("ok-paper", "bad-paper"):
        _seed(cfg, tmp_path, paper_id)
        manifest.upsert({"paper_id": paper_id, "tags": []})

    def _generate_tags(md, spec, existing_tags, max_tags, min_tags, max_excerpt_chars):
        if "bad-paper" in md:
            raise ValueError("instructor gave up after retries")
        return ["good-tag"]

    monkeypatch.setattr(ingest, "generate_tags", _generate_tags)
    monkeypatch.setattr(ingest, "normalize_manifest_tags", lambda cfg, manifest: {})

    ingest.retag(cfg, manifest)

    assert manifest.get("ok-paper")["tags"] == ["good-tag"]
    assert manifest.get("bad-paper")["tags"] == []


def test_retag_failure_is_logged(monkeypatch, capsys, make_config, tmp_path):
    cfg = make_config().for_ingest()
    manifest = Manifest(cfg.paths.rag_db)
    _seed(cfg, tmp_path, "bad-paper")
    manifest.upsert({"paper_id": "bad-paper", "tags": []})

    def _boom(md, spec, existing_tags, max_tags, min_tags, max_excerpt_chars):
        raise ValueError("instructor gave up after retries")

    monkeypatch.setattr(ingest, "generate_tags", _boom)
    monkeypatch.setattr(ingest, "normalize_manifest_tags", lambda cfg, manifest: {})

    ingest.retag(cfg, manifest)

    assert "[warn]" in capsys.readouterr().out
