"""config_writer: comment-preserving add/remove over config.yaml's papers: list."""

from __future__ import annotations

import textwrap

from rag.config import load_config
from rag.config_writer import add_paper, remove_paper

_YAML = textwrap.dedent(
    """\
    # Top-level comment describing this pool.
    collection: test_papers

    embedding:
      type: hf
      model: BAAI/bge-m3  # default local embedder

    papers:
      - { name: deepseek-v3, arxiv_id: "2412.19437" }  # foundational MoE paper
      - { name: kimi-k2, arxiv_id: "2507.20534" }
      - { name: glm-4.5, arxiv_id: "2508.06471" }  # trailing comment
    """
)


def _write_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(_YAML)
    return p


def test_add_paper_appends_new_entry(tmp_path):
    p = _write_config(tmp_path)
    result = add_paper(p, "new-paper", "2601.00001")
    assert result is None

    text = p.read_text()
    assert "name: new-paper" in text
    assert 'arxiv_id: "2601.00001"' in text
    # Pre-existing entries and their comments are untouched.
    assert "# foundational MoE paper" in text
    assert "# trailing comment" in text
    assert "# default local embedder" in text
    assert "# Top-level comment describing this pool." in text


def test_add_paper_dedups_by_arxiv_id_against_differently_named_entry(tmp_path):
    p = _write_config(tmp_path)
    before = p.read_text()

    # "deepseek-v3" is already curated under this arxiv_id; a UI-generated
    # name == arxiv_id must not slip past a name-only dedup check.
    result = add_paper(p, "2412.19437", "2412.19437")

    assert result == "deepseek-v3"
    assert p.read_text() == before  # no write on a duplicate


def test_remove_paper_removes_middle_entry_preserves_comments(tmp_path):
    p = _write_config(tmp_path)
    result = remove_paper(p, "kimi-k2")
    assert result is True

    text = p.read_text()
    assert "kimi-k2" not in text
    # Surviving entries' own comments are still attached to the right lines, and
    # unrelated sections are untouched — proves the removal isn't a blind
    # whole-file reserialization that could scramble comment attachment.
    assert "# foundational MoE paper" in text
    assert "deepseek-v3" in text
    assert "# trailing comment" in text
    assert "glm-4.5" in text
    assert "# default local embedder" in text
    assert "# Top-level comment describing this pool." in text


def test_remove_paper_unknown_name_is_a_noop(tmp_path):
    p = _write_config(tmp_path)
    before = p.read_text()
    assert remove_paper(p, "does-not-exist") is False
    assert p.read_text() == before


def test_remove_paper_when_no_papers_key_is_a_noop(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("collection: test_papers\n")
    assert remove_paper(p, "anything") is False


def test_add_paper_creates_papers_key_when_absent(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("collection: test_papers\n")

    result = add_paper(p, "new-paper", "2601.00001")

    assert result is None
    cfg = load_config(str(p))
    assert [paper.name for paper in cfg.papers] == ["new-paper"]
    assert cfg.papers[0].arxiv_id == "2601.00001"


def test_round_trip_is_still_loadable_by_load_config(tmp_path):
    p = _write_config(tmp_path)
    add_paper(p, "new-paper", "2601.00001")
    remove_paper(p, "kimi-k2")

    cfg = load_config(str(p))
    names = {paper.name for paper in cfg.papers}
    assert names == {"deepseek-v3", "glm-4.5", "new-paper"}
    new_entry = next(paper for paper in cfg.papers if paper.name == "new-paper")
    assert new_entry.arxiv_id == "2601.00001"
