"""Regression tests for the documentation checker's GitHub-style anchors."""

from scripts.check_docs import github_slug, headings


def test_github_slug_preserves_characters_that_distinguish_literal_fragments():
    assert github_slug("foo_bar") == "foo_bar"
    assert github_slug("foo  bar") == "foo--bar"
    assert github_slug("✂️ chunking") == "️-chunking"


def test_headings_suffix_duplicates_without_accepting_normalized_near_matches(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("# foo_bar\n# foo_bar\n", encoding="utf-8")

    assert headings(page) == {"foo_bar", "foo_bar-1"}
    assert "foobar" not in headings(page)
