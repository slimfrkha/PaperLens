"""Unit tests for arXiv-id extraction from reference markdown (offline, pure)."""

from __future__ import annotations

from rag import extract_cited_arxiv_ids


def test_extracts_the_three_arxiv_forms_in_order():
    md = (
        "## References\n"
        "- A. Author. Title one. arXiv:2402.17762, 2024.\n"
        "- B. Author. Title two. URL https://arxiv.org/abs/2307.09288.\n"
        "- C. Author. Title three. 10.48550/arXiv.2406.01574.\n"
    )
    out = extract_cited_arxiv_ids(md)
    assert [c.arxiv_id for c in out] == ["2402.17762", "2307.09288", "2406.01574"]


def test_strips_version_suffix_from_the_id():
    out = extract_cited_arxiv_ids("- D. Author. arXiv:2311.07911v3.\n")
    assert [c.arxiv_id for c in out] == ["2311.07911"]


def test_dedupes_same_id_across_forms_keeping_first_seen_label():
    md = "- first: arXiv:2307.09288 here.\n- again: 10.48550/arXiv.2307.09288 there.\n"
    out = extract_cited_arxiv_ids(md)
    assert len(out) == 1
    assert out[0].arxiv_id == "2307.09288"
    assert out[0].label == "first: arXiv:2307.09288 here."


def test_ignores_bare_numeric_tokens_without_arxiv_context():
    # A `\d{4}.\d{4,5}`-shaped token that is not an arXiv citation must not match.
    md = "- A result of 2402.17762 on the benchmark, ratio 1234.5678 improved.\n"
    assert extract_cited_arxiv_ids(md) == []


def test_label_drops_list_marker_and_collapses_whitespace():
    out = extract_cited_arxiv_ids("-   A. Author.   A   Title. arXiv:2402.17762, 2024.\n")
    assert out[0].label == "A. Author. A Title. arXiv:2402.17762, 2024."


def test_empty_and_reference_free_markdown_return_nothing():
    assert extract_cited_arxiv_ids("") == []
    assert extract_cited_arxiv_ids("## Intro\n\nNo citations here.\n") == []
