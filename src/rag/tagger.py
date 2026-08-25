"""LLM-generated, normalized tags for a paper (used at ingestion time)."""

from __future__ import annotations

import re

from pydantic import BaseModel

from .config import LLMSpec
from .llm import build_llm

_SYSTEM = (
    "You tag research papers for a search index. Emit concise, reusable topical tags "
    "naming the paper's field, methods, and key techniques or concepts. Prefer "
    "widely-used terms."
)


def _excerpt(md: str, max_chars: int = 6000) -> str:
    """Title + abstract + section headings — enough to tag without the full paper."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    lines = md.splitlines()
    headings = [ln for ln in lines if ln.startswith("## ")]
    # Grab text under the first "Abstract" heading, if present.
    abstract = ""
    for i, ln in enumerate(lines):
        if re.match(r"##\s*abstract", ln, re.IGNORECASE):
            abstract = "\n".join(lines[i + 1 : i + 25])
            break
    title = headings[0][3:].strip() if headings else ""
    body = f"Title: {title}\n\nAbstract:\n{abstract}\n\nSection headings:\n" + "\n".join(headings)
    return body[:max_chars]


def _canon(t: str) -> str:
    """Lowercase, kebab-case a single tag."""
    return re.sub(r"[^a-z0-9]+", "-", str(t).lower()).strip("-")


def _normalize(tags: list[str], max_tags: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = _canon(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:max_tags]


class _TagsOut(BaseModel):
    tags: list[str]


def generate_tags(
    md: str,
    spec: LLMSpec,
    existing_tags: list[str] | None = None,
    max_tags: int = 12,
    min_tags: int = 5,
    max_excerpt_chars: int = 6000,
) -> list[str]:
    existing = ", ".join(existing_tags or []) or "(none yet)"
    prompt = (
        f"Existing tags in the library (reuse these when they fit, to avoid "
        f"near-duplicates): {existing}\n\n"
        f"Paper:\n{_excerpt(md, max_chars=max_excerpt_chars)}\n\n"
        f"Return {min_tags}-{max_tags} lowercase kebab-case tags."
    )
    # No local try/except: a malformed reply that exhausts instructor's retries
    # propagates to the caller — pipeline.py's `except Exception` around both
    # call sites already logs a `[warn]` and degrades to `[]`.
    out = build_llm(spec).complete_structured(system=_SYSTEM, user=prompt, response_model=_TagsOut)
    return _normalize(out.tags, max_tags)


_NORMALIZE_SYSTEM = (
    "You curate a controlled vocabulary of topical tags for a research paper search "
    "index. You merge only tags that name the SAME concept — spelling variants, "
    "acronym vs expansion, near-synonyms. You never merge tags that name distinct "
    "concepts."
)


class _RemapOut(BaseModel):
    remap: dict[str, str]


def _filter_remap(remap: dict[str, str], valid: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in remap.items():
        src, dst = _canon(k), _canon(v)
        # Keep only real remaps: a known tag mapped to a different canonical form.
        if src in valid and dst and src != dst:
            out[src] = dst
    return out


def normalize_tags(tags: list[str], spec: LLMSpec) -> dict[str, str]:
    """Ask the LLM for a ``{tag -> canonical}`` map that merges only near-duplicates.

    Returns only the tags that should change; any tag absent from the map keeps its
    current form. Empty on empty input, so callers leave the vocabulary untouched
    rather than corrupt it. A malformed reply that exhausts retries raises — see
    `generate_tags` above for why that's left to the caller.
    """
    if not tags:
        return {}
    listing = "\n".join(f"- {t}" for t in tags)
    prompt = (
        f"Full tag vocabulary of the library:\n{listing}\n\n"
        "Some of these name the same concept in different words. Pick one canonical "
        "form per concept — prefer the clearest, most widely-used term already in the "
        "list — and map the near-duplicates onto it. Keep distinct concepts separate; "
        "never merge tags that mean different things.\n\n"
        "Map each tag that should change to its canonical form (omit tags that stay "
        "as they are)."
    )
    out = build_llm(spec).complete_structured(
        system=_NORMALIZE_SYSTEM, user=prompt, response_model=_RemapOut
    )
    return _filter_remap(out.remap, valid={_canon(t) for t in tags})
