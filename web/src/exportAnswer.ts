import type { Citation } from "./api";

// The `[rN]` marker pattern — the one place it's defined. `Answer.tsx` imports this
// (rather than keeping its own copy) so the two can't silently drift apart on what
// counts as a citation marker.
export const REF_MARKER = /\[(r\d+)\]/g;

/** Ordered, deduped refs from `text` that are real citations (present in `byRef`) —
 *  a turn's citations array can include registry entries the model never actually
 *  cited in its final answer. `Answer.tsx` uses this same helper to decide which
 *  `[rN]` markers become clickable badges. */
export function extractCitedRefs(text: string, byRef: Map<string, Citation>): string[] {
  const seen = new Set<string>();
  const refs: string[] = [];
  for (const m of text.matchAll(REF_MARKER)) {
    const ref = m[1];
    if (byRef.has(ref) && !seen.has(ref)) {
      seen.add(ref);
      refs.push(ref);
    }
  }
  return refs;
}

/** Citations actually cited inline in `text`, in order of first appearance — the
 *  shared "what was actually cited in this answer" primitive every export below
 *  needs (as opposed to every citation the search calls this turn ever surfaced). */
export function citedCitations(text: string, citations: Citation[]): Citation[] {
  const byRef = new Map(citations.map((c) => [c.ref, c]));
  return extractCitedRefs(text, byRef).map((ref) => byRef.get(ref)!);
}

// The number shown on-screen for a ref (`Answer.tsx`'s citation badge and this
// module's footnotes/BibTeX all use the same bare number, not the "rN" form).
export function refNumber(c: Citation): string {
  return c.ref.replace(/^r/, "");
}

/** Converts an answer's `[rN]` markers to `[^N]` footnotes — same number already
 *  shown on-screen, not renumbered — plus a `## References` block linking out to
 *  arXiv. Returns `text` unchanged when nothing was cited, so a small-talk turn
 *  doesn't get a dangling empty References heading. */
export function answerToMarkdown(text: string, cited: Citation[]): string {
  if (cited.length === 0) return text;
  const byRef = new Map(cited.map((c) => [c.ref, c]));
  const processed = text.replace(REF_MARKER, (m, ref: string) => {
    const c = byRef.get(ref);
    return c ? `[^${refNumber(c)}]` : m;
  });
  const sorted = [...cited].sort((a, b) => Number(refNumber(a)) - Number(refNumber(b)));
  const lines = sorted.map((c) => {
    const link = c.arxiv_id ? `, https://arxiv.org/abs/${c.arxiv_id}` : "";
    return `[^${refNumber(c)}]: **${c.title}** — ${c.section_title}${link}`;
  });
  return `${processed}\n\n## References\n\n${lines.join("\n")}`;
}

// New-style arXiv IDs only (YYMM.NNNNN, post-2007) — old-style ("hep-th/9901001")
// don't encode a year this way and are left without one rather than guessed at.
const NEW_STYLE_ARXIV_ID = /^(\d{2})\d{2}\.\d{4,5}$/;

function deriveYear(arxivId: string | null | undefined): string | undefined {
  const m = arxivId?.match(NEW_STYLE_ARXIV_ID);
  return m ? `20${m[1]}` : undefined;
}

/** Escapes BibTeX/LaTeX-special characters. Paper titles come from a parsed
 *  markdown heading — arbitrary paper-authored text, not trusted config — and can
 *  plausibly contain `{`, `}`, `&`, `%`, or `\\`; left raw, any of those breaks the
 *  entry's brace balance or field syntax and the exported `.bib` fails to parse. */
export function escapeBibtex(s: string): string {
  return s.replace(/\\/g, "\\textbackslash ").replace(/([{}&%])/g, "\\$1");
}

/** One `@misc` entry per distinct cited paper (a paper can be cited via several
 *  refs/passages — dedupe to one entry, in order of first appearance). Minimal,
 *  fully offline stub: title + arxiv_id + a year derived from the arXiv ID's YYMM
 *  prefix — no author field, since the manifest doesn't store one. `paper_id` (the
 *  config-name slug) is the citation key, for the same reason. */
export function answerToBibtex(cited: Citation[]): string {
  const byPaper = new Map<string, Citation>();
  for (const c of cited) if (!byPaper.has(c.paper_id)) byPaper.set(c.paper_id, c);

  const entries = [...byPaper.values()].map((c) => {
    const fields = [`  title = {${escapeBibtex(c.title)}}`];
    if (c.arxiv_id) {
      fields.push(`  eprint = {${c.arxiv_id}}`);
      fields.push(`  archivePrefix = {arXiv}`);
    }
    const year = deriveYear(c.arxiv_id);
    if (year) fields.push(`  year = {${year}}`);
    return `@misc{${c.paper_id},\n${fields.join(",\n")},\n}`;
  });
  return entries.join("\n\n");
}
