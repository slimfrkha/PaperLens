"""LLM-generated, normalized tags for a paper (used at ingestion time)."""

from __future__ import annotations

import json
import re

from .config import LLMSpec
from .llm import build_llm

_SYSTEM = (
    "You tag machine-learning research papers for a search index. Emit concise, "
    "reusable topical tags (techniques, architectures, training/inference methods, "
    "model family). Prefer widely-used terms."
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


def _parse(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, list):
                return [str(x) for x in val]
        except json.JSONDecodeError:
            pass
    # Fallback: comma/newline separated.
    return [p.strip("-*• \t") for p in re.split(r"[,\n]", text) if p.strip()]


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
        f"Return ONLY a JSON array of {min_tags}-{max_tags} lowercase kebab-case tags."
    )
    raw = build_llm(spec).complete(system=_SYSTEM, user=prompt)
    return _normalize(_parse(raw), max_tags)


_NORMALIZE_SYSTEM = (
    "You curate a controlled vocabulary of topical tags for a machine-learning "
    "paper search index. You merge only tags that name the SAME concept — spelling "
    "variants, acronym vs expansion, near-synonyms. You never merge tags that name "
    "distinct techniques."
)


def _parse_map(text: str, valid: set[str]) -> dict[str, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in obj.items():
        src, dst = _canon(str(k)), _canon(str(v))
        # Keep only real remaps: a known tag mapped to a different canonical form.
        if src in valid and dst and src != dst:
            out[src] = dst
    return out


def normalize_tags(tags: list[str], spec: LLMSpec) -> dict[str, str]:
    """Ask the LLM for a ``{tag -> canonical}`` map that merges only near-duplicates.

    Returns only the tags that should change; any tag absent from the map keeps its
    current form. Empty on empty input or an unparsable reply, so callers leave the
    vocabulary untouched rather than corrupt it.
    """
    if not tags:
        return {}
    listing = "\n".join(f"- {t}" for t in tags)
    prompt = (
        f"Full tag vocabulary of the library:\n{listing}\n\n"
        "Some of these name the same concept in different words. Pick one canonical "
        "form per concept — prefer the clearest, most widely-used term already in the "
        "list — and map the near-duplicates onto it. Keep distinct techniques separate; "
        "never merge tags that mean different things.\n\n"
        "Return ONLY a JSON object mapping each tag that should change to its canonical "
        "form (omit tags that stay as they are)."
    )
    raw = build_llm(spec).complete(system=_NORMALIZE_SYSTEM, user=prompt)
    return _parse_map(raw, valid={_canon(t) for t in tags})
