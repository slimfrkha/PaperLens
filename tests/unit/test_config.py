"""Config loading and project-root path anchoring."""

from __future__ import annotations

import dataclasses
import textwrap

import pytest

from rag.config import (
    CONFIG_ENV_VAR,
    ChunkingCfg,
    IngestConfig,
    RetrievalCfg,
    TaggerCfg,
    load_config,
    parse_config,
)

_YAML = textwrap.dedent(
    """
    collection: my_papers
    paths:
      rag_db: data/db
      pdf_dir: /tmp/abs_pdfs
    papers:
      - { name: foo, arxiv_id: "1234.5678" }
    """
)


def _write_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(_YAML)
    return p


def test_relative_paths_anchored_to_config_dir(tmp_path):
    cfg_path = _write_config(tmp_path)
    cfg = load_config(str(cfg_path))
    # Relative rag_db is resolved under the config file's directory.
    assert cfg.paths.rag_db == str((tmp_path / "data/db").resolve())
    assert cfg.root == tmp_path.resolve()


def test_relative_paths_anchored_to_pyproject_root_not_config_dir(tmp_path):
    # The exact bug: config lives in configs/, a pyproject.toml marks the real
    # root above it. Relative paths must anchor to the root, not to configs/.
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    p = cfgdir / "config.yaml"
    p.write_text(_YAML)

    cfg = load_config(str(p))
    assert cfg.root == tmp_path.resolve()  # NOT tmp_path/configs
    assert cfg.paths.rag_db == str((tmp_path / "data/db").resolve())


def test_absolute_paths_left_untouched(tmp_path):
    cfg = load_config(str(_write_config(tmp_path)))
    assert cfg.paths.pdf_dir == "/tmp/abs_pdfs"


def test_parses_scalars_and_papers(tmp_path):
    cfg = load_config(str(_write_config(tmp_path)))
    assert cfg.collection == "my_papers"
    assert [p.name for p in cfg.papers] == ["foo"]
    assert cfg.papers[0].arxiv_id == "1234.5678"


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(str(tmp_path / "nope.yaml"))
    assert cfg.collection == "arxiv_papers"  # default
    assert cfg.papers == []


def test_env_var_override(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg_path))
    cfg = load_config()  # no explicit path -> reads the env var
    assert cfg.collection == "my_papers"


def test_choice_registry_selects_variant(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "embedding:\n  type: openai\n  api_base: http://x/v1\n"
        "llm:\n  chat:\n    type: openai\n    api_base: http://x/v1\n"
    )
    cfg = load_config(str(p))
    assert type(cfg.embedding).__name__ == "OpenAIEmbeddingCfg"
    assert type(cfg.llm.chat).__name__ == "OpenAISpec"
    assert cfg.llm.chat.api_base == "http://x/v1"


def test_interpolation_resolves_from_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("collection: papers_${server.port}\nserver:\n  port: 9000\n")
    assert load_config(str(p)).collection == "papers_9000"


def test_cli_override_feeds_interpolation(tmp_path):
    # Regression guard for the parse_config _postprocessing seam: a CLI override
    # must flow into a ${...} that references it. Pinned to draccus internals.
    p = tmp_path / "config.yaml"
    p.write_text("collection: papers_${server.port}\nserver:\n  port: 9000\n")
    cfg = parse_config(["--config_path", str(p), "--server.port=1234"])
    assert cfg.server.port == 1234
    assert cfg.collection == "papers_1234"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("collectionn: typo\n")  # misspelled key -> loud failure
    with pytest.raises(Exception, match="not valid"):
        load_config(str(p))


def test_legacy_provider_key_rejected(tmp_path):
    # `provider` was renamed to `type`; the old key must fail loudly, not default.
    p = tmp_path / "config.yaml"
    p.write_text("llm:\n  chat:\n    provider: openai\n")
    with pytest.raises(Exception, match="not valid"):
        load_config(str(p))


def test_incoherent_knobs_fail_at_construction():
    # Guardrails, not gold-plating: overlap >= max makes chunk packing carry every
    # prior block forward, silently poisoning the index instead of erroring.
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingCfg(max_tokens=512, overlap_tokens=512)
    with pytest.raises(ValueError, match="noise_ratio"):
        ChunkingCfg(noise_ratio=1.5)
    with pytest.raises(ValueError, match="retrieval.k"):
        RetrievalCfg(k=30, candidates=20)
    with pytest.raises(ValueError, match="min_tags"):
        TaggerCfg(min_tags=8, max_tags=3)


def test_incoherent_knobs_fail_at_config_load(tmp_path):
    # The failure must land at startup, not on the tenth paper. draccus wraps the
    # __post_init__ ValueError in a ParsingError, so the reason rides on __cause__.
    p = tmp_path / "config.yaml"
    p.write_text("chunking:\n  max_tokens: 512\n  overlap_tokens: 512\n")
    with pytest.raises(Exception, match="ChunkingCfg") as exc:
        load_config(str(p))
    assert "overlap_tokens" in str(exc.value.__cause__)


def test_for_ingest_exposes_only_ingestion_fields(tmp_path):
    # The projection's surface locks out serve-only config (server/reranker/retrieval/chat).
    names = {f.name for f in dataclasses.fields(IngestConfig)}
    assert names == {
        "paths",
        "collection",
        "embedding",
        "tagging",
        "chunking",
        "extraction",
        "tagger",
        "papers",
    }


def test_for_ingest_is_a_shallow_view_of_config(tmp_path):
    cfg = load_config(str(_write_config(tmp_path)))
    icfg = cfg.for_ingest()
    # Same objects, not copies: a genuine view over the parent Config.
    assert icfg.paths is cfg.paths
    assert icfg.embedding is cfg.embedding
    assert icfg.papers is cfg.papers
    assert icfg.tagging is cfg.llm.tagging  # flattened from llm.tagging
    assert icfg.collection == cfg.collection


def test_for_ingest_result_is_frozen(tmp_path):
    icfg = load_config(str(_write_config(tmp_path))).for_ingest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        icfg.collection = "mutated"  # ty: ignore[invalid-assignment]  # asserting frozen at runtime
