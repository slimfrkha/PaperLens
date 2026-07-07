"""Section-aware chunking for Docling-extracted arXiv papers.

Docling flattens every heading to `##`, but the section *numbering*
(`2`, `2.1`, `2.1.1`) preserves the true hierarchy in the heading text.
We split on `##` boundaries, rebuild that hierarchy into a breadcrumb,
prepend it to each chunk (so the embedding carries context), and
normalize the size tails: big sections are split on paragraph/table
boundaries with overlap, tiny/noise sections are dropped or merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sections we never want in the index (citation lists, TOCs, author blocks).
_SKIP_TITLES = re.compile(
    r"^(references|bibliography|contents|table of contents|"
    r"acknowledge?ments?|contributors?|core contributors?|tech leads?|advisors?|"
    r"appendix\s*$|nvidia\s*$|.*\bteam\s*$)",
    re.IGNORECASE,
)

# A body that opens with a figure/table caption ("Figure 9 | Expert load ...").
_CAPTION = re.compile(r"^(figure|table)\s+\d", re.IGNORECASE)

# A caption line anywhere in the body ("... Figure 10 | Expert load ..."), which
# marks a Docling figure fragment even when plot numbers precede it.
_CAPTION_INLINE = re.compile(r"(figure|table)\s+\d+\s*\|", re.IGNORECASE)

# A token that is purely numeric/punctuation (plot axis dumps, e.g. "0 2 4 6").
_NUMERIC_TOK = re.compile(r"^[\d.,%()\-]+$")

# A heading like "2.1.1. Multi-Head Latent Attention" -> ("2.1.1", "Multi-Head ...")
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*\S)\s*$")

# A markdown table row: leading optional whitespace then a pipe.
_TABLE_ROW = re.compile(r"^\s*\|")


@dataclass
class Chunk:
    text: str  # breadcrumb + body, i.e. what actually gets embedded
    body: str  # the raw section/sub-chunk body without the breadcrumb prefix
    metadata: dict = field(default_factory=dict)


def approx_tokens(text: str) -> int:
    """Cheap, embedder-agnostic token estimate (~1.33 tokens/word).

    Used only for size thresholds, so it needn't match any specific
    tokenizer — it just has to be monotonic and roughly right.
    """
    return int(len(text.split()) * 1.33) + 1


def _split_sections(md: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on `##` boundaries."""
    parts = re.split(r"(?m)^##[ \t]+", md)
    sections: list[tuple[str, str]] = []
    for part in parts:
        if not part.strip():
            continue
        line, _, rest = part.partition("\n")
        sections.append((line.strip(), rest.strip()))
    return sections


def _blocks(body: str) -> list[str]:
    """Split a section body into atomic blocks.

    Paragraphs split on blank lines, but a run of consecutive table rows
    is kept as a single atomic block so we never break a table mid-way.
    """
    lines = body.split("\n")
    blocks: list[str] = []
    buf: list[str] = []
    in_table = False

    def flush():
        if buf:
            blocks.append("\n".join(buf).strip())
            buf.clear()

    for line in lines:
        is_table = bool(_TABLE_ROW.match(line))
        if is_table and not in_table:
            flush()  # close the previous paragraph, start the table block
            in_table = True
        elif not is_table and in_table and line.strip():
            flush()  # table ended, non-table content begins
            in_table = False
        if not line.strip() and not in_table:
            flush()
            continue
        buf.append(line)
    flush()
    return [b for b in blocks if b]


def _is_caption_noise(body: str) -> bool:
    """Detect Docling figure fragments: caption bodies or plot-number dumps.

    Only applied to *unnumbered* sections — numbered sections and markdown
    tables (which carry `|`) are always kept, so eval tables are never lost.
    """
    body = body.strip()
    # A genuine markdown table has several pipe-delimited rows -> always keep.
    if sum(bool(_TABLE_ROW.match(ln)) for ln in body.splitlines()) >= 2:
        return False
    if _CAPTION.match(body) or _CAPTION_INLINE.search(body):
        return True
    toks = body.split()
    return bool(toks) and sum(bool(_NUMERIC_TOK.match(t)) for t in toks) / len(toks) > 0.4


def _pack_blocks(blocks: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack blocks into sub-chunks up to max_tokens, with overlap.

    A single block larger than max_tokens (e.g. a huge table) is emitted
    whole rather than broken — losing table integrity is worse than an
    oversized chunk, and modern embedders tolerate long context.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for block in blocks:
        bt = approx_tokens(block)
        if cur and cur_tok + bt > max_tokens:
            chunks.append("\n\n".join(cur))
            # seed the next chunk with trailing blocks for overlap continuity
            carry, carry_tok = [], 0
            for prev in reversed(cur):
                pt = approx_tokens(prev)
                if carry_tok + pt > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tok += pt
            cur, cur_tok = carry[:], carry_tok
        cur.append(block)
        cur_tok += bt
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


class _Hierarchy:
    """Tracks the latest title seen at each section-number depth."""

    def __init__(self, paper_title: str):
        self.paper_title = paper_title
        self._by_number: dict[str, str] = {}

    def breadcrumb(self, number: str | None, title: str) -> str:
        trail = [self.paper_title]
        if number:
            self._by_number[number] = title
            parts = number.split(".")
            for i in range(1, len(parts)):  # ancestor prefixes: 2, 2.1, ...
                anc = ".".join(parts[:i])
                if anc in self._by_number:
                    trail.append(f"{anc} {self._by_number[anc]}")
            trail.append(f"{number} {title}")
        else:
            trail.append(title)
        return " > ".join(trail)


def chunk_markdown(
    md: str,
    *,
    paper_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 24,
) -> list[Chunk]:
    """Turn one paper's markdown into breadcrumb-prefixed chunks."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)  # drop Docling image placeholders
    sections = _split_sections(md)
    if not sections:
        return []

    paper_title = sections[0][0]  # first `##` heading is the paper title
    hier = _Hierarchy(paper_title)
    chunks: list[Chunk] = []

    for heading, body in sections:
        m = _NUMBERED.match(heading)
        number, title = (m.group(1), m.group(2)) if m else (None, heading)

        # Register in the hierarchy even if we skip/merge, so breadcrumbs of
        # later subsections can still resolve their ancestors.
        breadcrumb = hier.breadcrumb(number, title)

        if _SKIP_TITLES.match(title.strip()):
            continue
        if not body or approx_tokens(body) < min_tokens:
            continue  # bare title lines, author lists, stray captions
        if number is None and _is_caption_noise(body):
            continue  # unnumbered figure-caption / plot-number fragments

        blocks = _blocks(body)
        sub_bodies = _pack_blocks(blocks, max_tokens, overlap_tokens)

        for i, sub in enumerate(sub_bodies):
            text = f"{breadcrumb}\n\n{sub}"
            chunks.append(
                Chunk(
                    text=text,
                    body=sub,
                    metadata={
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "section_number": number or "",
                        "section_title": title,
                        "breadcrumb": breadcrumb,
                        "body": sub,  # the reader reads this back; don't re-derive it downstream
                        "part": i,
                        "n_parts": len(sub_bodies),
                        "approx_tokens": approx_tokens(text),
                    },
                )
            )
    return chunks
