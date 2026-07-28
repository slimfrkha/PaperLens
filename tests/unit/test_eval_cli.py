"""Phase 6 CLI: ``confirm`` + the config-block emitter.

Offline against a real temp Chroma (fake embedder, no models). ``load_config``/``load_pool``
are monkeypatched per test (no real ``config.yaml``/markdown dir needed) — the same style
``cmd_confirm`` uses for everything else. Every reranker is injected: ``cmd_confirm`` builds
one unconditionally (regardless of ``rerank``'s resolved value, mirroring optimizer.py's screen
arms), so a test that forgets to inject one would otherwise try to load a real cross-encoder.
"""

from __future__ import annotations

import argparse
import json
import tempfile

import pytest
import yaml

import eval.cli as cli_mod
from eval.queryset import QAItem, item_to_dict


def _item(query: str, paper_id: str, section_title: str, section_number: str = "1") -> QAItem:
    return QAItem(
        query=query,
        paper_id=paper_id,
        gold_span=(0, 1),
        source_unit=f"{section_number} {section_title}",
        section_number=section_number,
        section_title=section_title,
    )


class FakeReranker:
    """Scores a doc by how many of ``boost`` words it contains — deterministic, no model."""

    def __init__(self, boost: set[str]) -> None:
        self.boost = boost

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(sum(w in d.lower() for w in self.boost)) for d in docs]


def _confirm_args(
    config=None, limit=None, max_tokens=None, candidates=None, rerank=None
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config, limit=limit, max_tokens=max_tokens, candidates=candidates, rerank=rerank
    )


def _write_split(cfg, pool: dict[str, str], items: list[QAItem], split: str) -> str:
    """Write ``evals/<fp>.<split>.jsonl`` under ``cfg.root``, mirroring what ``gen`` writes."""
    from eval.fingerprint import corpus_fingerprint

    fp = corpus_fingerprint(pool)
    out_dir = cfg.root / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fp}.{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(item_to_dict(it)) + "\n")
    return fp


def _forbidden(name: str):
    def _raise(*args, **kwargs):
        raise AssertionError(f"{name} should not have been called")

    return _raise


def test_confirm_baseline_reproduces_run_and_skips_isolation(
    monkeypatch, make_config, make_searcher, seed_chunks, fake_embedder, tmp_path
):
    docs = [
        seed_chunks("p1", "Method", "latent attention compresses the cache", doc_id="p1-method")
    ]
    make_searcher(docs)  # seeds tmp_path/rag_db under the shared test collection
    cfg = make_config(root=tmp_path)
    pool = {"p1": "## Paper\n\nAuthors\n\n## 1. Method\n\nlatent attention compresses the cache\n"}
    items = [_item("latent attention compresses the cache", "p1", "Method")]
    fp = _write_split(cfg, pool, items, "test")

    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)
    monkeypatch.setattr(cli_mod, "build_isolated_searcher", _forbidden("build_isolated_searcher"))

    cli_mod.cmd_confirm(_confirm_args(), embedder=fake_embedder, reranker=FakeReranker(set()))

    marker = json.loads((tmp_path / "evals" / f"{fp}.confirm.json").read_text())
    assert marker["max_tokens"] == cfg.chunking.max_tokens
    assert marker["candidates"] == cfg.retrieval.candidates
    assert marker["rerank"] == cfg.reranker.enabled
    assert marker["success_at_candidates"] == 1.0


def test_confirm_override_takes_isolated_branch(monkeypatch, make_config, fake_embedder, tmp_path):
    cfg = make_config(root=tmp_path)  # default chunking max_tokens=512
    pool = {"p1": "## Paper\n\nAuthors\n\n## 1. Method\n\nlatent attention compresses the cache\n"}
    items = [_item("latent attention compresses the cache", "p1", "Method")]
    fp = _write_split(cfg, pool, items, "test")

    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)

    real_build = cli_mod.build_isolated_searcher
    calls: list[object] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "build_isolated_searcher", spy)

    args = _confirm_args(max_tokens=256, candidates=10, rerank=False)
    cli_mod.cmd_confirm(args, embedder=fake_embedder, reranker=FakeReranker(set()))

    assert len(calls) == 1  # the isolated branch — and only it — was taken
    marker = json.loads((tmp_path / "evals" / f"{fp}.confirm.json").read_text())
    assert marker["max_tokens"] == 256
    assert marker["candidates"] == 10
    assert marker["rerank"] is False


def test_confirm_rejects_candidates_below_k(monkeypatch, make_config, tmp_path):
    cfg = make_config(root=tmp_path)  # retrieval.k defaults to 5
    pool = {"p1": "## Paper\n\nAuthors\n\n## 1. Method\n\nalpha beta\n"}
    items = [_item("alpha beta", "p1", "Method")]
    _write_split(cfg, pool, items, "test")
    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)

    with pytest.raises(SystemExit, match="below retrieval.k"):
        cli_mod.cmd_confirm(_confirm_args(candidates=1))


def test_confirm_builds_reranker_via_cfg_helper_not_lazy_default(
    monkeypatch, make_config, make_searcher, seed_chunks, fake_embedder, tmp_path
):
    # Regression guard: `cmd_confirm` must resolve the reranker via `build_reranker_for_cfg`
    # (correct for an LLM-backend reranker config) rather than `Searcher.reranker`'s lazy
    # default (always the hf cross-encoder — silently wrong for a `type: llm` config).
    docs = [
        seed_chunks("p1", "Method", "alpha delta", doc_id="p1-m0"),
        seed_chunks("p1", "Method", "alpha beta", doc_id="p1-m1"),
    ]
    make_searcher(docs)
    cfg = make_config(root=tmp_path)
    pool = {"p1": "## Paper\n\nAuthors\n\n## 1. Method\n\nalpha delta\n"}
    items = [_item("alpha", "p1", "Method")]
    _write_split(cfg, pool, items, "test")

    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)

    fake = FakeReranker({"delta"})
    calls: list[object] = []
    monkeypatch.setattr(
        cli_mod, "build_reranker_for_cfg", lambda cfg_arg: (calls.append(cfg_arg), fake)[1]
    )

    # No `reranker=` injected here on purpose — cmd_confirm must build one itself.
    cli_mod.cmd_confirm(_confirm_args(rerank=True), embedder=fake_embedder)

    assert calls == [cfg]


def test_confirm_touched_once_marker_warns_but_does_not_block(
    monkeypatch, make_config, make_searcher, seed_chunks, fake_embedder, tmp_path, capsys
):
    docs = [seed_chunks("p1", "Method", "alpha beta", doc_id="p1-m")]
    make_searcher(docs)
    cfg = make_config(root=tmp_path)
    pool = {"p1": "## Paper\n\nAuthors\n\n## 1. Method\n\nalpha beta\n"}
    items = [_item("alpha beta", "p1", "Method")]
    _write_split(cfg, pool, items, "test")
    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)

    args = _confirm_args()
    cli_mod.cmd_confirm(args, embedder=fake_embedder, reranker=FakeReranker(set()))
    capsys.readouterr()  # discard first run's output

    cli_mod.cmd_confirm(args, embedder=fake_embedder, reranker=FakeReranker(set()))  # second call
    out = capsys.readouterr().out
    assert "already confirmed" in out
    assert "weakens the held-out guarantee" in out


def test_format_config_block_round_trips_and_scopes_retrieval_to_candidates():
    from rag.config import ChunkingCfg, Config, HFEmbeddingCfg, HFRerankerCfg

    cfg = Config(embedding=HFEmbeddingCfg(), reranker=HFRerankerCfg(enabled=False))
    text = cli_mod.format_config_block(
        cfg, chunking=ChunkingCfg(max_tokens=256), candidates=50, rerank=True
    )
    data = yaml.safe_load(text)
    assert data["chunking"]["max_tokens"] == 256
    assert data["embedding"]["type"] == "hf"
    assert data["reranker"]["type"] == "hf"
    assert data["reranker"]["enabled"] is True  # rerank override, not cfg's own (disabled)
    assert data["retrieval"] == {"candidates": 50}  # k/max_rounds never leaked in


def test_load_test_set_raises_when_test_split_missing(monkeypatch, make_config, tmp_path):
    cfg = make_config(root=tmp_path)
    pool = {"p1": "## Paper\n\nAuthors\n\n## 1. Method\n\nalpha beta\n"}
    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)

    with pytest.raises(SystemExit, match="paperlens-eval gen"):
        cli_mod._load_test_set(_confirm_args())


def test_end_to_end_gen_screen_sweep_confirm_produces_a_parseable_block(
    monkeypatch, make_config, fake_llm, fake_embedder, tmp_path, capsys
):
    from eval.optimizer import screen_chunking, screen_retrieval, sweep
    from eval.queryset import GenConfig, build_queryset, split_by_paper

    pool = {
        "p1": "## Paper One\n\nAuthors\n\n"
        "## 1. Method\n\nlatent attention compresses the key value cache substantially\n\n"
        "## 2. Training\n\nfp8 mixed precision schedule with a long warmup for stability\n",
        "p2": "## Paper Two\n\nAuthors\n\n"
        "## 1. Method\n\nrotary embeddings extend the context window efficiently\n\n"
        "## 2. Training\n\nadamw optimizer with cosine decay and gradient clipping\n",
    }
    llm = fake_llm(answer='{"question": "What method is used?", "answer": "see section"}')
    # min_section_tokens=1: the test's sections are short, don't filter them out.
    # test_frac=0.5: with only 2 papers, the default 0.25 rounds to 0 held-out papers.
    gen = GenConfig(min_section_tokens=1, test_frac=0.5)
    items = build_queryset(pool, llm, gen)
    dev_items, test_items = split_by_paper(items, pool, gen)
    assert dev_items and test_items  # both splits non-empty, else the e2e chain is vacuous

    cfg = make_config(root=tmp_path)
    _write_split(cfg, pool, test_items, "test")

    reranker = FakeReranker(set())
    with tempfile.TemporaryDirectory() as tmp_a:
        dev_searcher = cli_mod.build_isolated_searcher(
            pool,
            cfg.chunking,
            cfg.embedding,
            db_dir=tmp_a,
            embedder=fake_embedder,
            reranker=reranker,
        )
        screen_retrieval(cfg, dev_items, searcher=dev_searcher, candidate_grid=[5, 10])

    with tempfile.TemporaryDirectory() as tmp_b:
        screen_chunking(
            cfg,
            pool,
            dev_items,
            db_dir=tmp_b,
            grids={"max_tokens": [256]},
            embedder=fake_embedder,
            reranker=reranker,
        )

    with tempfile.TemporaryDirectory() as tmp_s:
        sweep(
            cfg,
            pool,
            dev_items,
            db_dir=tmp_s,
            max_tokens_grid=[256],
            candidate_grid=[5, 10],
            embedder=fake_embedder,
            reranker=reranker,
        )

    monkeypatch.setattr(cli_mod, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli_mod, "load_pool", lambda md_dir: pool)

    args = _confirm_args(max_tokens=256, candidates=5, rerank=False)
    cli_mod.cmd_confirm(args, embedder=fake_embedder, reranker=reranker)

    out = capsys.readouterr().out
    assert "config.yaml block" in out
    block_text = out.split("config.yaml block ---")[1].strip()
    parsed = yaml.safe_load(block_text)
    assert parsed["chunking"]["max_tokens"] == 256
    assert parsed["retrieval"]["candidates"] == 5
