"""Config loading and project-root path anchoring."""

from __future__ import annotations

import textwrap

from rag.config import CONFIG_ENV_VAR, load_config

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
