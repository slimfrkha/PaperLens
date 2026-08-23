import type { LibraryAnnotation } from "./api";

/** Groups annotations by paper, sorted alphabetically by paper title, each group's
 *  notes newest-first by `created_at`. Neither `manifest.papers()`'s order (arbitrary
 *  manifest-file insertion order) nor `AnnotationStore.list_all`'s array order (not
 *  guaranteed newest-first) is a fit for display — leaving it to whatever the backend
 *  happens to return would make the Notes page reorder unpredictably as annotations are
 *  edited. `NotesPage` and `notesToMarkdown` both call this, so the on-screen list and
 *  the exported markdown always agree. */
export function groupByPaper(
  notes: LibraryAnnotation[],
): { paperId: string; paperTitle: string; arxivId?: string; notes: LibraryAnnotation[] }[] {
  const byPaper = new Map<string, LibraryAnnotation[]>();
  for (const n of notes) {
    if (!byPaper.has(n.paper_id)) byPaper.set(n.paper_id, []);
    byPaper.get(n.paper_id)!.push(n);
  }
  const groups = [...byPaper.entries()].map(([paperId, paperNotes]) => ({
    paperId,
    paperTitle: paperNotes[0].paper_title,
    arxivId: paperNotes[0].arxiv_id,
    notes: [...paperNotes].sort((a, b) => b.created_at.localeCompare(a.created_at)),
  }));
  return groups.sort((a, b) => a.paperTitle.localeCompare(b.paperTitle));
}

/** One `## Paper Title (arXiv:ID)` heading per paper, each annotation as a quoted
 *  snippet + note. Returns `""` for an empty list, same "nothing to export" convention
 *  `answerToMarkdown` uses for an uncited answer. */
export function notesToMarkdown(notes: LibraryAnnotation[]): string {
  if (notes.length === 0) return "";
  const groups = groupByPaper(notes);
  const sections = groups.map(({ paperTitle, arxivId, notes: paperNotes }) => {
    const heading = arxivId ? `${paperTitle} (arXiv:${arxivId})` : paperTitle;
    const items = paperNotes.map((n) => {
      const quote = `> "${n.snippet}" — ${n.section_title}`;
      return n.note ? `${quote}\n\n${n.note}` : quote;
    });
    return `## ${heading}\n\n${items.join("\n\n")}`;
  });
  return `# My Notes\n\n${sections.join("\n\n")}`;
}
