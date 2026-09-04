"""Check PaperLens Markdown links, anchors, code fences, and navigation depth."""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import deque
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


def documentation_files() -> list[Path]:
    """Return PaperLens docs without vendored repositories or research notes."""
    return sorted(
        [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md"), ROOT / "configs/examples/README.md"]
    )


def github_slug(value: str) -> str:
    """Generate the GitHub-style slug for the Markdown heading characters we support.

    GitHub lowercases heading text, drops punctuation and symbols, preserves letters,
    numbers, combining marks, hyphens, and underscores, then replaces each ASCII space
    with a hyphen. Keeping combining marks matters for emoji variation selectors.
    """
    value = value.lower()
    value = re.sub(r"<[^>]+>", "", value)
    allowed = {"L", "M", "N"}
    slug = "".join(
        char
        for char in value
        if unicodedata.category(char)[0] in allowed or char in {" ", "-", "_"}
    )
    return slug.replace(" ", "-")


def headings(path: Path) -> set[str]:
    """Return unique anchors, including GitHub's numeric suffix for duplicates."""
    anchors: set[str] = set()
    occurrences: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        original = github_slug(match.group(1))
        anchor = original
        while anchor in occurrences:
            occurrences[original] = occurrences.get(original, 0) + 1
            anchor = f"{original}-{occurrences[original]}"
        occurrences[anchor] = 0
        anchors.add(anchor)
    return anchors


def link_target(source: Path, raw: str) -> tuple[Path, str] | None:
    raw = raw.strip()
    if raw.startswith(("http://", "https://", "mailto:")):
        return None
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    target, _, fragment = raw.partition("#")
    path = source if not target else (source.parent / unquote(target)).resolve()
    return path, fragment


def check() -> list[str]:
    files = documentation_files()
    known = {path.resolve() for path in files}
    heading_cache: dict[Path, set[str]] = {}
    graph: dict[Path, set[Path]] = {path.resolve(): set() for path in files}
    errors: list[str] = []

    for source in files:
        relative = source.relative_to(ROOT)
        fence: str | None = None
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if match := FENCE_RE.match(line):
                marker, info = match.groups()
                if fence is None:
                    fence = marker[0]
                    if not info.strip():
                        errors.append(
                            f"{relative}:{line_number}: opening code fence has no language"
                        )
                elif marker[0] == fence:
                    fence = None
                continue

            if fence is not None:
                continue

            for match in LINK_RE.finditer(line):
                resolved = link_target(source.resolve(), match.group(1))
                if resolved is None:
                    continue
                target, fragment = resolved
                if not target.exists():
                    errors.append(
                        f"{relative}:{line_number}: missing target {target.relative_to(ROOT)}"
                    )
                    continue
                if target in known:
                    graph[source.resolve()].add(target)
                if fragment and target.suffix.lower() == ".md":
                    anchors = heading_cache.setdefault(target, headings(target))
                    if unquote(fragment) not in anchors:
                        errors.append(
                            f"{relative}:{line_number}: missing heading #{fragment} in "
                            f"{target.relative_to(ROOT)}"
                        )

        if fence is not None:
            errors.append(f"{relative}: unclosed code fence")

    # Reader-facing Markdown should be reachable from the README in at most two clicks.
    reader_docs = {
        path.resolve()
        for path in files
        if path.name != "CLAUDE.md" and path.relative_to(ROOT).parts[0] != ".claude"
    }
    start = (ROOT / "README.md").resolve()
    distance = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for target in graph.get(current, set()):
            if target not in distance:
                distance[target] = distance[current] + 1
                queue.append(target)
    for path in sorted(reader_docs):
        if distance.get(path, 3) > 2:
            errors.append(f"{path.relative_to(ROOT)}: not reachable from README.md in two clicks")

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("\n".join(errors))
        return 1
    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
