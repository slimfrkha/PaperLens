import type { Paper } from "./api";

/** Mirrors `ChatAgent._resolve_paper_ids`'s tag/paper-intersection + manifest-fallback
 *  logic (`src/server/agent.py`) client-side, over the already-fetched paper list — so
 *  Compare mode can disable itself below 2 resolved papers, and warn before running over
 *  a large one, without a new API round-trip. `papers` must carry each entry's `tags` (the
 *  `Paper` type already does), so this needs no extra fetch beyond what the composer's
 *  tag/paper filters already loaded. */
export function resolveScopeSize(
  papers: Paper[],
  tags: string[],
  selectedPapers: string[],
): number {
  const tagIds = tags.length
    ? papers.filter((p) => p.tags.some((t) => tags.includes(t))).map((p) => p.paper_id)
    : null;
  const selected = selectedPapers.length ? selectedPapers : null;
  let ids: string[] | null;
  if (tagIds === null) ids = selected;
  else if (selected === null) ids = tagIds;
  else {
    const wanted = new Set(selected);
    ids = tagIds.filter((id) => wanted.has(id));
  }
  // Compare always needs a concrete scope to count, unlike a normal Ask turn (which is
  // fine leaving an inactive filter as "the entire library" with no explicit list) —
  // mirrors agent.py's `fallback_to_manifest=True` path.
  return (ids ?? papers.map((p) => p.paper_id)).length;
}
