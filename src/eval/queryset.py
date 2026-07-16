"""Span-anchored QA generation from the loaded pool.

Generation strategy: one paper at a time, one question per section. Sections are
split on ``##`` boundaries — a *config-independent* unit, so no chunking arm gets to
recognize its own handwriting (the circular-query-gen trap). The section's character
span in the source markdown is the gold; a retrieved chunk counts relevant later iff
it overlaps that span.

Gold is a ``[start, end]`` char range in the raw markdown, never a chunk id, so the
set survives a re-chunk and keeps chunking configs comparable.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass

# Reuse chunking's skip/numbering rules so generation samples the *same* substantive
# sections chunking keeps in the index (references/TOC/etc. are dropped there too).
from rag.chunking import _NUMBERED, _SKIP_TITLES, approx_tokens
from rag.llm import LLMBackend

# A `##` heading line; group 1 is the heading text (trailing spaces trimmed in code).
_HEADING = re.compile(r"(?m)^##[ \t]+(.*)$")


@dataclass(frozen=True)
class GenConfig:
    """Frozen question-generation config.

    Kept separate from the retrieval ``Config`` on purpose: swapping retrieval knobs
    (chunking/embedder/reranker) must never change the eval set. Freeze these and the
    fingerprint governs regeneration.
    """

    min_section_tokens: int = 24  # skip sections too short to ask a real question about
    max_section_chars: int = 6000  # cap the section text sent to the generator
    test_frac: float = 0.25  # fraction of *papers* held out for the test split
    seed: int = 0  # deterministic paper shuffle for the split


@dataclass
class Section:
    """One `##`-delimited section with its char span in the source markdown."""

    paper_id: str
    number: str | None
    title: str
    body: str
    start: int  # char offset of `body` in the source markdown (md[start:end] == body)
    end: int


@dataclass
class QAItem:
    query: str
    paper_id: str
    gold_span: tuple[int, int]
    source_unit: str  # section number + title the question was generated from
    # Section identity of the gold: scoring matches a retrieved chunk as relevant iff it
    # carries this same (paper_id, section_number, section_title). Config-independent (the
    # `##` split ignores chunking knobs), so it keeps arms comparable. `section_number` is
    # "" (not None) to mirror how `rag.index` stores it in chunk metadata.
    section_number: str = ""
    section_title: str = ""
    stratum: str = "section"
    answer: str = ""  # kept for reference / optional closed-book genfilter (Phase 7)


def iter_sections(md: str, paper_id: str, *, min_tokens: int):
    """Yield `Section`s with exact spans, skipping the same sections chunking drops.

    The first `##` heading is the paper title; its body is authorship/affiliation
    boilerplate, so it is never a generation source.
    """
    matches = list(_HEADING.finditer(md))
    for i, m in enumerate(matches):
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        raw_body = md[m.end() : sec_end]
        lead = len(raw_body) - len(raw_body.lstrip())
        body = raw_body.strip()
        if not body:
            continue
        start = m.end() + lead
        end = start + len(body)  # md[start:end] == body by construction

        heading = m.group(1).strip()
        nm = _NUMBERED.match(heading)
        number, title = (nm.group(1), nm.group(2)) if nm else (None, heading)

        if i == 0:  # paper-title section
            continue
        if _SKIP_TITLES.match(title.strip()):
            continue
        if approx_tokens(body) < min_tokens:
            continue
        yield Section(paper_id, number, title, body, start, end)


_GEN_SYSTEM = (
    "You write evaluation questions for a retrieval system over machine-learning "
    "research papers. Given ONE section of a paper, write a single specific, factual "
    "question that can be answered ONLY from that section's text — not from outside "
    "knowledge of the paper. Respond ONLY with a JSON object: "
    '{"question": "...", "answer": "..."}.'
)


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def generate_query(section: Section, llm: LLMBackend, max_chars: int) -> dict | None:
    """One short LLM call → a question+answer grounded in the section, or None.

    Long sections are truncated to ``max_chars`` for the prompt, but the gold span
    still covers the whole section — so questions from sections longer than
    ``max_chars`` are front-biased. Acceptable at this scale; revisit if sections
    routinely exceed the cap.
    """
    user = (
        f"Section: {section.title}\n\n"
        f"{section.body[:max_chars]}\n\n"
        "Write one question answerable only from this section, plus its short answer. "
        "JSON only."
    )
    obj = _parse_json(llm.complete(system=_GEN_SYSTEM, user=user))
    if not obj or not str(obj.get("question", "")).strip():
        return None
    return obj


def _section_unit(sec: Section) -> str:
    return f"{sec.number} {sec.title}".strip() if sec.number else sec.title


def iter_queryset(pool: dict[str, str], llm: LLMBackend, gen: GenConfig) -> Iterator[QAItem]:
    """Yield one QA item per substantive section across the pool, streaming.

    Streaming (vs. buffering a list) lets the caller write to disk as questions are
    produced, so a crash keeps partial progress. A failure on one section — e.g. a
    server hiccup on a long serial run — is logged and skipped rather than losing
    the whole batch. ``KeyboardInterrupt`` is not caught, so Ctrl-C stops cleanly
    with everything generated so far already on disk.
    """
    for i, (paper_id, md) in enumerate(pool.items(), 1):
        n = 0
        for sec in iter_sections(md, paper_id, min_tokens=gen.min_section_tokens):
            try:
                obj = generate_query(sec, llm, gen.max_section_chars)
            except Exception as e:  # one bad section must not sink the run
                print(f"  ! {paper_id} :: {sec.title}: {e}")
                continue
            if obj is None:
                continue
            n += 1
            yield QAItem(
                query=str(obj["question"]).strip(),
                paper_id=paper_id,
                gold_span=(sec.start, sec.end),
                source_unit=_section_unit(sec),
                section_number=sec.number or "",
                section_title=sec.title,
                answer=str(obj.get("answer", "")).strip(),
            )
        print(f"  [{i}/{len(pool)}] {paper_id}: {n} questions")


def build_queryset(pool: dict[str, str], llm: LLMBackend, gen: GenConfig) -> list[QAItem]:
    """Eager wrapper over :func:`iter_queryset` (convenient in tests)."""
    return list(iter_queryset(pool, llm, gen))


def held_out_paper_ids(pool: dict[str, str], gen: GenConfig) -> set[str]:
    """Paper ids held out for the test split — a pure function of the pool and seed,
    independent of which sections happened to yield questions. Lets the caller route
    each streamed item to dev/test without buffering the whole set first."""
    papers = sorted(pool)
    n_test = round(len(papers) * gen.test_frac) if len(papers) > 1 else 0
    shuffled = papers[:]
    random.Random(gen.seed).shuffle(shuffled)
    return set(shuffled[:n_test])


def split_by_paper(
    items: list[QAItem], pool: dict[str, str], gen: GenConfig
) -> tuple[list[QAItem], list[QAItem]]:
    """Partition into (dev, test) by *paper* — never by query, so no paper leaks
    across both splits. Deterministic given ``gen.seed``."""
    test_ids = held_out_paper_ids(pool, gen)
    dev = [it for it in items if it.paper_id not in test_ids]
    test = [it for it in items if it.paper_id in test_ids]
    return dev, test


def item_to_dict(it: QAItem) -> dict:
    d = asdict(it)
    d["gold_span"] = list(it.gold_span)  # tuple -> list for JSON round-trip stability
    return d


def load_queryset(path: str) -> list[QAItem]:
    """Read a ``.jsonl`` eval split back into ``QAItem``s.

    Raises with a regenerate hint if the file predates the section-identity fields
    (``gen`` before Phase 2), since scoring needs them to match retrieved chunks.
    """
    items: list[QAItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "section_title" not in d:
                raise SystemExit(
                    f"{path} predates the section-identity fields — regenerate the eval "
                    f"set with `paperlens-eval gen`."
                )
            items.append(
                QAItem(
                    query=d["query"],
                    paper_id=d["paper_id"],
                    gold_span=(d["gold_span"][0], d["gold_span"][1]),
                    source_unit=d["source_unit"],
                    section_number=d.get("section_number", ""),
                    section_title=d["section_title"],
                    stratum=d.get("stratum", "section"),
                    answer=d.get("answer", ""),
                )
            )
    return items
