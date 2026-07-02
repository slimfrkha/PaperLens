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
    body = f"Title: {title}\n\nAbstract:\n{abstract}\n\nSection headings:\n" + "\n".join(
        headings
    )
    return body[:max_chars]


def _normalize(tags: list[str], max_tags: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = re.sub(r"[^a-z0-9]+", "-", str(t).lower()).strip("-")
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
    md: str, spec: LLMSpec, existing_tags: list[str] | None = None, max_tags: int = 12
) -> list[str]:
    existing = ", ".join(existing_tags or []) or "(none yet)"
    prompt = (
        f"Existing tags in the library (reuse these when they fit, to avoid "
        f"near-duplicates): {existing}\n\n"
        f"Paper:\n{_excerpt(md)}\n\n"
        f"Return ONLY a JSON array of 5-{max_tags} lowercase kebab-case tags."
    )
    raw = build_llm(spec).complete(system=_SYSTEM, user=prompt)
    return _normalize(_parse(raw), max_tags)
