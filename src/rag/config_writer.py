"""Comment-preserving read-modify-write helpers over config.yaml's `papers:` list.

Used by the admin add/remove-paper HTTP routes (``server.main``) so a UI-driven edit
survives alongside the human-authored comments and formatting elsewhere in
config.yaml, instead of the whole file being reserialized by a comment-blind dumper
(plain ``pyyaml``, used everywhere else in ``rag.config``, would do exactly that).
"""

from __future__ import annotations

import threading
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

_yaml = YAML()
_yaml.preserve_quotes = True
# Matches this repo's existing config.yaml convention (see configs/*.yaml): a block
# sequence's dash indented 2 past its key, item content indented 4 total.
_yaml.indent(mapping=2, sequence=4, offset=2)

# Serializes concurrent admin requests against the same file: two near-simultaneous
# add/remove calls must not interleave a read-modify-write cycle, or one write can
# clobber the other.
_lock = threading.Lock()


def _load(config_path: Path) -> CommentedMap:
    with open(config_path) as f:
        return _yaml.load(f) or CommentedMap()


def _write(config_path: Path, data: CommentedMap) -> None:
    # Write-temp-then-rename so a crash or full disk mid-write can't leave a
    # truncated config.yaml (same pattern as server.annotations.AnnotationStore).
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        _yaml.dump(data, f)
    tmp.replace(config_path)


def add_paper(config_path: Path, name: str, arxiv_id: str) -> str | None:
    """Append ``{name, arxiv_id}`` to ``papers:`` in config.yaml.

    Dedups by ``arxiv_id`` (not ``name``) against every existing entry, since a
    hand-curated entry may already reference this paper under a human-chosen name
    (e.g. ``deepseek-v3`` for arxiv_id ``2412.19437``) that a UI-generated
    ``name == arxiv_id`` won't match textually. Returns ``None`` on success, or the
    existing entry's name if this arxiv_id is already present (the caller turns that
    into a 409 "already curated as <name>").
    """
    with _lock:
        data = _load(config_path)
        papers = data.get("papers")
        if papers is None:
            papers = CommentedSeq()
            data["papers"] = papers
        for entry in papers:
            if str(entry.get("arxiv_id")) == arxiv_id:
                return str(entry.get("name"))
        # DoubleQuotedScalarString so arxiv_id renders quoted, matching the existing
        # convention (unquoted would parse back as a float and lose precision/format).
        entry = CommentedMap([("name", name), ("arxiv_id", DoubleQuotedScalarString(arxiv_id))])
        entry.fa.set_flow_style()  # matches the `{ name: ..., arxiv_id: "..." }` convention
        papers.append(entry)
        _write(config_path, data)
        return None


def remove_paper(config_path: Path, name: str) -> bool:
    """Remove the ``papers:`` entry matching ``name`` from config.yaml.

    Returns ``False`` if no entry matches (e.g. a manifest record that predates this
    feature and was never in config.yaml to begin with).
    """
    with _lock:
        data = _load(config_path)
        papers = data.get("papers")
        if not papers:
            return False
        for i, entry in enumerate(papers):
            if str(entry.get("name")) == name:
                del papers[i]
                _write(config_path, data)
                return True
        return False
