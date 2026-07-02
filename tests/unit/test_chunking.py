"""Section-aware chunking + breadcrumb hierarchy."""

from __future__ import annotations

import textwrap

from rag.chunking import approx_tokens, chunk_markdown

_LONG = " ".join(["word"] * 60)  # ~80 approx-tokens, above min, below max


def _md(body_sections: str) -> str:
    return textwrap.dedent(body_sections).strip()


def test_empty_input_yields_no_chunks():
    assert chunk_markdown("", paper_id="p") == []


def test_first_heading_is_paper_title_and_breadcrumb_root():
    md = _md(
        f"""
        ## My Paper

        ## 1 Intro
        {_LONG}
        """
    )
    chunks = chunk_markdown(md, paper_id="p")
    intro = [c for c in chunks if c.metadata["section_title"] == "Intro"][0]
    assert intro.metadata["paper_title"] == "My Paper"
    assert intro.metadata["breadcrumb"] == "My Paper > 1 Intro"


def test_nested_numbering_builds_ancestor_breadcrumb():
    md = _md(
        f"""
        ## Title

        ## 2 Methods
        {_LONG}

        ## 2.1 Attention
        {_LONG}
        """
    )
    chunks = chunk_markdown(md, paper_id="p")
    sub = [c for c in chunks if c.metadata["section_number"] == "2.1"][0]
    assert sub.metadata["breadcrumb"] == "Title > 2 Methods > 2.1 Attention"


def test_skip_titles_dropped():
    md = _md(
        f"""
        ## Title

        ## References
        {_LONG}
        """
    )
    assert chunk_markdown(md, paper_id="p") == []


def test_tiny_sections_dropped_by_min_tokens():
    md = _md(
        """
        ## Title

        ## 1 Intro
        too short
        """
    )
    assert chunk_markdown(md, paper_id="p") == []


def test_unnumbered_caption_noise_dropped():
    md = _md(
        f"""
        ## Title

        ## Figure 3 Some caption {_LONG}
        """
    )
    assert chunk_markdown(md, paper_id="p") == []


def test_markdown_table_is_kept_as_one_block():
    rows = "\n".join(f"| a{i} | b{i} |" for i in range(6))
    md = _md(
        f"""
        ## Title

        ## 3 Results
        | col a | col b |
        {rows}
        """
    )
    chunks = chunk_markdown(md, paper_id="p")
    assert any("| col a | col b |" in c.body for c in chunks)


def test_large_section_splits_with_overlap():
    huge = "\n\n".join(" ".join(["tok"] * 40) for _ in range(20))
    md = f"## Title\n\n## 4 Big\n{huge}"
    chunks = chunk_markdown(md, paper_id="p", max_tokens=120, overlap_tokens=40)
    parts = [c for c in chunks if c.metadata["section_title"] == "Big"]
    assert len(parts) > 1
    assert parts[0].metadata["n_parts"] == len(parts)


def test_approx_tokens_monotonic():
    assert approx_tokens("one two three") > approx_tokens("one")
