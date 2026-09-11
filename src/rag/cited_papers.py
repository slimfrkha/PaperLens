"""Extract cited arXiv papers from a paper's already-extracted markdown.

References survive into the canonical ``{paper_id}.md`` — chunking drops them from
the *embeddings* (``chunking._SKIP_TITLES``), never from the file — so identifying
the arXiv papers a paper cites is a regex over that text: no re-extraction, no
network.

Only references carrying an explicit arXiv id are recovered — an id in an ``arXiv:``
tag, an ``arxiv.org/abs/`` URL, or a ``10.48550/arXiv.`` DOI. References without one
(blogs, model cards, conference/journal-only papers) are intentionally not surfaced.
A bare ``\\d{4}.\\d{4,5}`` is deliberately NOT matched: without the arXiv context it
picks up stray numeric tokens (table values, decimals).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL_MAX = 300

# `\d{4}\.\d{4,5}` in an arXiv context: an `arXiv:` / `arXiv.` / `arXiv ` tag (the
# `.` form also covers the `10.48550/arXiv.<id>` DOI) or an `arxiv.org/abs/<id>` URL.
# A version suffix (`v2`) falls outside the capture group. Case-insensitive.
_ARXIV_IN_TEXT = re.compile(
    r"(?:arxiv[:.\s]\s*|arxiv\.org/abs/)(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
# A leading markdown list marker on a reference line: `- `, `* `, or `12. `.
_LIST_MARKER = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


@dataclass(frozen=True)
class CitedPaper:
    arxiv_id: str
    label: str  # the reference line the id sits on — human-readable on the HTML path


def _clean_label(line: str) -> str:
    label = _LIST_MARKER.sub("", line.strip())
    return re.sub(r"\s+", " ", label).strip()[:_LABEL_MAX]


def extract_cited_arxiv_ids(markdown: str) -> list[CitedPaper]:
    """Cited arXiv papers found in ``markdown``, deduped by id, first-seen order.

    The label is the cleaned, length-capped line the id appears on — one bibliography
    entry per line on the HTML extraction path; on the Docling/PDF path references can
    arrive as unstructured prose, so the id is clean but the label may be a fragment.
    """
    seen: set[str] = set()
    out: list[CitedPaper] = []
    for line in markdown.splitlines():
        ids = _ARXIV_IN_TEXT.findall(line)
        if not ids:
            continue
        label = _clean_label(line)
        for arxiv_id in ids:
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            out.append(CitedPaper(arxiv_id=arxiv_id, label=label))
    return out
