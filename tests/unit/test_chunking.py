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
    # Distinguishable blocks so the overlap carry-forward is observable: the tail
    # block of each sub-chunk must reappear at the head of the next one. With
    # identical blocks the test can't tell overlap from coincidence.
    blocks = [f"BLOCK{j} " + " ".join(["tok"] * 14) for j in range(12)]  # ~20 tokens each
    md = "## Title\n\n## 4 Big\n" + "\n\n".join(blocks)
    parts = [
        c
        for c in chunk_markdown(md, paper_id="p", max_tokens=60, overlap_tokens=25)
        if c.metadata["section_title"] == "Big"
    ]
    assert len(parts) > 1
    assert parts[0].metadata["n_parts"] == len(parts)
    # The last BLOCK marker in each part reappears in the next — proof the overlap
    # carry-forward ran. Delete it in _pack_blocks and this assertion goes red.
    for a, b in zip(parts, parts[1:], strict=False):  # parts[1:] is intentionally shorter
        last_marker = [w for w in a.body.split() if w.startswith("BLOCK")][-1]
        assert last_marker in b.body


def test_approx_tokens_monotonic():
    assert approx_tokens("one two three") > approx_tokens("one")
