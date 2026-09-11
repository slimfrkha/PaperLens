import type { Paper } from "./api";

/** The resolved in-scope paper ids for the current tag/paper filters, or `null` when no
 *  filter is active (the entire library — no explicit list). Mirrors
 *  `ChatAgent._resolve_paper_ids`'s tag/paper-intersection logic (`src/server/agent.py`)
 *  client-side, over the already-fetched paper list, so callers need no API round-trip.
 *  `papers` must carry each entry's `tags` (the `Paper` type already does). */
export function resolveScopeIds(
  papers: Paper[],
  tags: string[],
  selectedPapers: string[],
): string[] | null {
  const tagIds = tags.length
    ? papers.filter((p) => p.tags.some((t) => tags.includes(t))).map((p) => p.paper_id)
    : null;
  const selected = selectedPapers.length ? selectedPapers : null;
  if (tagIds === null) return selected;
  if (selected === null) return tagIds;
  const wanted = new Set(selected);
  return tagIds.filter((id) => wanted.has(id));
}

export function resolveScopeSize(
  papers: Paper[],
  tags: string[],
  selectedPapers: string[],
): number {
  // Compare always needs a concrete scope to count, unlike a normal Ask turn (which is
  // fine leaving an inactive filter as "the entire library" with no explicit list) —
  // mirrors agent.py's `fallback_to_manifest=True` path.
  const ids = resolveScopeIds(papers, tags, selectedPapers);
  return (ids ?? papers.map((p) => p.paper_id)).length;
}
