"""Trial-based generation of cross-paper comparative questions.

Unlike :mod:`queryset` (deterministic sweep over every substantive section, one question
per section), finding genuinely comparable content across papers isn't enumerable up
front — whether two papers even discuss the same thing is exactly what needs discovering.
So generation here is a **trial loop**: draw a handful of random papers, show an LLM their
outlines (titles + full section list, never body text at this stage) and ask it to spot
sections in different papers that discuss the same underlying thing, then — for every
match it finds — ask a second LLM call to write one question whose complete answer needs
every matched section. Repeat until a target pool size is reached or a trial budget is
exhausted.

Reuses :class:`queryset.Section` directly as the per-paper gold unit (see
:class:`ComparativeQAItem`) — no parallel data model needed, it already carries everything
a comparative item's gold requires, including the character span each single-paper
``QAItem`` gold span already relies on.

See ``comparative-eval-spec.md`` for the full design.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from tqdm import tqdm

from rag.llm import LLMBackend

from .queryset import _HEADING, Section, _parse_json, _section_unit, iter_sections


@dataclass
class ComparativeQAItem:
    """One cross-paper question: ``sections`` is >= 2 gold units, one per paper it spans.

    ``sections[0]`` is not necessarily the paper "in charge" of anything — it's simply the
    paper that appeared earliest in the trial's own random paper-draw order among the
    papers that made it into the match (see :func:`build_comparative_queryset`). Scoring
    (``eval.comparative_metrics``) uses that ordering as the bootstrap cluster key.
    """

    query: str
    sections: list[Section]  # >=2, one per gold paper
    answer: str = ""  # reference answer, same as QAItem.answer


# Named so cli.py's argparse defaults can reference the same numbers instead of
# duplicating them (mirrors harness.DEFAULT_PER_PAPER_N's reasoning). Every one of these
# is a placeholder, not a calibrated number — see comparative-eval-spec.md's "Open items".
DEFAULT_TARGET_P = 40
DEFAULT_MAX_TRIALS = 200
DEFAULT_N_PAPERS_MIN = 2
DEFAULT_N_PAPERS_MAX = 6


@dataclass(frozen=True)
class ComparativeGenConfig:
    """Frozen trial-loop config. Every default below is a placeholder, not a calibrated
    number — see comparative-eval-spec.md's "Open items"."""

    target_p: int = DEFAULT_TARGET_P  # soft floor: an overshooting trial keeps every match
    max_trials: int = DEFAULT_MAX_TRIALS  # hard cap regardless of hit rate
    n_papers_min: int = DEFAULT_N_PAPERS_MIN
    n_papers_max: int = DEFAULT_N_PAPERS_MAX
    min_section_tokens: int = 24  # mirrors GenConfig
    max_section_chars: int = 6000  # mirrors GenConfig
    test_frac: float = 0.25  # item-level split, decided per item as it streams
    seed: int = 0  # governs trial paper/size draws and the dev/test split coin flip


def _paper_title(md: str) -> str:
    """The paper's own title — the first `##` heading's text.

    ``iter_sections`` deliberately skips this section as a generation source (its body is
    authorship/affiliation boilerplate), but the outline shown to the spotting call needs
    it for cross-paper context (see :func:`_paper_outline`).
    """
    m = _HEADING.search(md)
    return m.group(1).strip() if m else ""


def _paper_outline(paper_id: str, title: str, sections: list[Section]) -> str:
    """Title + the whole list of substantive `##`-level sections for one paper — headings
    alone invite false matches (two papers can each have a section literally titled
    "Training" about unrelated things); the *whole* hierarchy (e.g. "3. Training" followed
    by "3.1 Data Curation", "3.2 Learning Rate Schedule") lets the model reason about what
    each section actually covers, without paying for any body text at this stage."""
    lines = [f"Paper {paper_id}: {title}"] + [f"  {_section_unit(s)}" for s in sections]
    return "\n".join(lines)


_SPOT_SYSTEM = (
    "You review outlines of multiple machine-learning research papers to find sections "
    "in DIFFERENT papers that discuss the same underlying concept, method, or result -- "
    "genuinely comparable content, not just a shared generic heading. Only report a match "
    "if you are confident the sections could be meaningfully compared or contrasted. "
    "Respond ONLY with a JSON array; each element is a list of "
    '{"paper_id": "...", "section_number": "...", "section_title": "..."} objects (2 or '
    "more, from different papers) naming one matching group. Return an empty array if "
    "nothing genuinely comparable is found -- do not force a match."
)


def _parse_match_groups(text: str) -> list[list[dict]] | None:
    """Syntactic parse only — a raw JSON array of raw group lists, or ``None`` if the
    reply isn't parseable at all. Per-entry validation against the real candidate section
    lists happens separately in :func:`_valid_group`, mirroring ``generate_query``'s own
    two-step "parse, then validate" tolerance for malformed LLM output."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        val = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(val, list):
        return None
    return [g for g in val if isinstance(g, list)]


def _spot_matches(
    outlines: dict[str, tuple[str, list[Section]]], llm: LLMBackend
) -> list[list[dict]] | None:
    """One LLM call over every offered paper's outline -> zero or more raw match groups.
    ``outlines`` maps paper_id -> (title, candidate sections)."""
    user = "\n\n".join(_paper_outline(pid, title, secs) for pid, (title, secs) in outlines.items())
    return _parse_match_groups(llm.complete(system=_SPOT_SYSTEM, user=user))


def _valid_group(
    group: list[dict], outlines: dict[str, tuple[str, list[Section]]]
) -> list[Section] | None:
    """Resolve one raw match group to real ``Section``s, or ``None`` if it's malformed:
    fewer than 2 distinct papers, the same paper referenced twice (violates the "one
    section per paper" gold shape), or a ``(paper_id, section_number, section_title)``
    that doesn't exist in that paper's own candidate list — a hallucinated reference.
    Discarded, never crashed on, same tolerance ``generate_query`` already has for
    malformed LLM output.
    """
    resolved: list[Section] = []
    seen: set[str] = set()
    for entry in group:
        if not isinstance(entry, dict):
            return None
        pid = str(entry.get("paper_id", ""))
        if pid not in outlines or pid in seen:
            return None
        _, secs = outlines[pid]
        sec = next(
            (
                s
                for s in secs
                if (s.number or "") == str(entry.get("section_number", ""))
                and s.title == entry.get("section_title")
            ),
            None,
        )
        if sec is None:
            return None
        resolved.append(sec)
        seen.add(pid)
    return resolved if len(seen) >= 2 else None


def _order_by_draw(resolved: list[Section], paper_ids: list[str]) -> list[Section]:
    """Reorder a resolved match group to the trial's own paper-draw order, discarding the
    LLM's own (uncontrolled) ordering — keeps ``sections[0]`` reproducible from the trial's
    seed alone, which is what the bootstrap cluster key (``primary_paper_id``,
    ``eval.comparative_metrics``) depends on."""
    by_paper = {s.paper_id: s for s in resolved}
    return [by_paper[pid] for pid in paper_ids if pid in by_paper]


_WRITE_SYSTEM = (
    "You write evaluation questions for a retrieval system over machine-learning "
    "research papers. Given matching sections from DIFFERENT papers, write ONE specific "
    "question whose complete, correct answer genuinely requires information from EVERY "
    "section shown -- a question answerable from only one of them does not count. "
    'Respond ONLY with a JSON object: {"question": "...", "answer": "..."}.'
)


def _write_comparative_query(
    sections: list[Section], llm: LLMBackend, max_chars: int
) -> dict | None:
    """One LLM call -> a question+answer needing every section in ``sections``, or
    ``None``. Mirrors ``generate_query``'s truncation/parsing exactly, just over N
    sections' text instead of one."""
    parts = [
        f"Paper {s.paper_id}, section {_section_unit(s)}:\n\n{s.body[:max_chars]}" for s in sections
    ]
    user = (
        "\n\n---\n\n".join(parts)
        + "\n\nWrite one question that needs every section above, plus its short answer. "
        "JSON only."
    )
    obj = _parse_json(llm.complete(system=_WRITE_SYSTEM, user=user))
    if not obj or not str(obj.get("question", "")).strip():
        return None
    return obj


@dataclass
class _GenResult:
    items: list[ComparativeQAItem] = field(default_factory=list)
    trials: int = 0
    paper_pair_frequency: Counter[tuple[str, str]] = field(default_factory=Counter)


def build_comparative_queryset(
    pool: dict[str, str],
    llm: LLMBackend,
    gen: ComparativeGenConfig,
    *,
    show_progress: bool = False,
    on_item: Callable[[ComparativeQAItem], None] | None = None,
) -> _GenResult:
    """Run the trial loop until ``target_p`` items are collected or ``max_trials`` trials
    have run, whichever comes first.

    Each trial: draw ``n_papers ~ Uniform[n_papers_min, n_papers_max]`` (clamped to the
    pool size) papers uniformly at random, spot comparable sections across their outlines,
    then write one question per match group found — a trial yielding multiple match
    groups yields multiple items, not capped at one. ``target_p`` is a soft floor: a trial
    that pushes the pool past it still contributes every match group it found, never
    split or truncated mid-trial.

    ``on_item``, when given, is called immediately as each item is produced — before it's
    appended to the returned result — so a caller can stream it to disk and keep partial
    progress on a crash or Ctrl-C, the same property ``iter_queryset``'s own streaming
    (generator) interface gives the single-paper path. This isn't a generator itself
    because trial count and paper-pair frequency are intrinsic to the loop rather than the
    yielded items, and the caller needs both (for the ``gen`` report) alongside the items
    — returning everything once at the end is simpler than threading two more values
    through a side channel a generator would need.

    Raises ``SystemExit`` if the pool doesn't have enough papers to form even one valid
    trial — better than burning the whole ``max_trials`` budget on trials that can never
    succeed. Raises ``ValueError`` if ``n_papers_min > n_papers_max`` — ``cmd_comparative_gen``
    already rejects this at the CLI boundary, but this function is a public, directly
    callable entry point too (tests, scripts), and without this check the same inverted
    range would instead crash with a bare ``ValueError`` from ``random.randint`` deep
    inside the trial loop.
    """
    if gen.n_papers_min > gen.n_papers_max:
        raise ValueError(
            f"n_papers_min ({gen.n_papers_min}) must be <= n_papers_max ({gen.n_papers_max})"
        )
    if len(pool) < gen.n_papers_min:
        raise SystemExit(
            f"Pool has {len(pool)} papers, fewer than n_papers_min={gen.n_papers_min} — "
            f"cannot form even one valid trial."
        )

    all_papers = sorted(pool)
    rng = random.Random(gen.seed)
    paper_titles = {pid: _paper_title(md) for pid, md in pool.items()}
    paper_sections = {
        pid: list(iter_sections(md, pid, min_tokens=gen.min_section_tokens))
        for pid, md in pool.items()
    }

    result = _GenResult()
    bar = tqdm(total=gen.max_trials, desc="comparative gen (trials)", disable=not show_progress)
    try:
        while len(result.items) < gen.target_p and result.trials < gen.max_trials:
            result.trials += 1
            bar.update(1)
            n_papers = rng.randint(gen.n_papers_min, min(gen.n_papers_max, len(pool)))
            paper_ids = rng.sample(all_papers, n_papers)
            outlines = {pid: (paper_titles[pid], paper_sections[pid]) for pid in paper_ids}

            try:
                raw_groups = _spot_matches(outlines, llm)
            except Exception as e:  # one bad trial must not sink the run
                tqdm.write(f"  ! spotting call failed: {e}", file=sys.stderr)
                continue
            if not raw_groups:
                continue

            for raw_group in raw_groups:
                resolved = _valid_group(raw_group, outlines)
                if resolved is None:
                    continue
                ordered = _order_by_draw(resolved, paper_ids)
                try:
                    obj = _write_comparative_query(ordered, llm, gen.max_section_chars)
                except Exception as e:
                    tqdm.write(f"  ! writing call failed: {e}", file=sys.stderr)
                    continue
                if obj is None:
                    continue
                item = ComparativeQAItem(
                    query=str(obj["question"]).strip(),
                    sections=ordered,
                    answer=str(obj.get("answer", "")).strip(),
                )
                if on_item is not None:
                    on_item(item)
                result.items.append(item)
                for a, b in _paper_pairs(ordered):
                    result.paper_pair_frequency[(a, b)] += 1
    finally:
        bar.close()

    return result


def _paper_pairs(sections: list[Section]) -> list[tuple[str, str]]:
    """Every sorted `(paper_a, paper_b)` pair among an item's gold papers — order-
    independent, so (A, B) and (B, A) across different items count as the same pair in
    the frequency log."""
    papers = sorted({s.paper_id for s in sections})
    return [(papers[i], papers[j]) for i in range(len(papers)) for j in range(i + 1, len(papers))]


def comparative_item_to_dict(it: ComparativeQAItem) -> dict:
    """JSON-serializable form — ``body`` is kept (unlike ``QAItem``'s span-only gold)
    so a human can read the actual matched text back from the eval-set file without
    re-parsing the pool, for a manual audit pass."""
    return {
        "query": it.query,
        "answer": it.answer,
        "sections": [
            {
                "paper_id": s.paper_id,
                "number": s.number,
                "title": s.title,
                "body": s.body,
                "start": s.start,
                "end": s.end,
            }
            for s in it.sections
        ],
    }


def comparative_item_from_dict(d: dict) -> ComparativeQAItem:
    return ComparativeQAItem(
        query=d["query"],
        answer=d.get("answer", ""),
        sections=[
            Section(
                paper_id=sd["paper_id"],
                number=sd.get("number"),
                title=sd["title"],
                body=sd.get("body", ""),
                start=sd["start"],
                end=sd["end"],
            )
            for sd in d["sections"]
        ],
    )


def load_comparative_queryset(path: str) -> list[ComparativeQAItem]:
    """Read a ``.comparative.{dev,test}.jsonl`` split back into ``ComparativeQAItem``s."""
    items: list[ComparativeQAItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(comparative_item_from_dict(json.loads(line)))
    return items
