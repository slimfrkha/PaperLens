// Builds a table-of-contents from a rendered paper's heading elements.
//
// rehype-slug (Markdown.tsx) has already assigned every heading a stable, disambiguated
// `id` by the time this runs — those real DOM ids are used directly, not reconstructed,
// so there's no risk of drifting from rehype-slug's own slugging algorithm.

export interface OutlineEntry {
  id: string;
  text: string;
  depth: number;
}

// A heading like "2.1.1 Multi-Head Latent Attention" -> depth 3. Mirrors
// src/rag/chunking.py's _NUMBERED, independently in TypeScript — a small, self-contained
// regex duplication is preferable to reimplementing rehype-slug's slugger server-side
// just to keep two heading-id algorithms in sync (see product-features-spec.md § 4
// "Design decision: client-side only").
const NUMBERED = /^(\d+(?:\.\d+)*)\.?\s+(.*)$/;

/** Walk `root`'s heading elements in document order and derive a flat, depth-annotated
 *  outline. Headings with no `id` (shouldn't happen — rehype-slug assigns one to every
 *  heading it renders) are skipped rather than included with a broken jump target.
 *
 *  Not every heading is numbered — Docling still emits a bolded subsection as a plain
 *  `##`, alongside its numbered siblings — and nothing survives to say where in the true
 *  hierarchy an unnumbered heading belongs. An unnumbered heading's depth is derived from
 *  its nearest numbered neighbors on BOTH sides, not just the one before it (a look-behind-
 *  only rule was tried and reverted: it confidently misnests a trailing unnumbered section,
 *  e.g. an "Appendix" after "5 Conclusion", as that section's child):
 *  - nothing numbered has appeared yet (the title, "Abstract") -> depth 0.
 *  - nothing numbered follows (a trailing "Appendix"/"References") -> a sibling of the
 *    last numbered heading, not its child.
 *  - the next numbered heading is deeper than the previous one -> match the next one (this
 *    heading is introducing what follows, e.g. an unnumbered "Contributions" right before
 *    "1.1 Related Work").
 *  - otherwise -> one level under the previous numbered heading (this heading is that
 *    section's own child, with nothing deeper following it). */
export function buildOutline(root: HTMLElement): OutlineEntry[] {
  const headings = Array.from(root.querySelectorAll("h1, h2, h3, h4, h5, h6")).filter(
    (h): h is HTMLElement => !!h.id,
  );
  const parsed = headings.map((h) => {
    const text = (h.textContent ?? "").trim();
    const m = NUMBERED.exec(text);
    return { id: h.id, text, numberedDepth: m ? m[1].split(".").length : null };
  });

  const prevNumbered: (number | null)[] = [];
  let runningPrev: number | null = null;
  for (const p of parsed) {
    prevNumbered.push(runningPrev);
    if (p.numberedDepth !== null) runningPrev = p.numberedDepth;
  }
  const nextNumbered: (number | null)[] = new Array<number | null>(parsed.length).fill(null);
  let runningNext: number | null = null;
  for (let i = parsed.length - 1; i >= 0; i--) {
    nextNumbered[i] = runningNext;
    if (parsed[i].numberedDepth !== null) runningNext = parsed[i].numberedDepth;
  }

  return parsed.map(({ id, text, numberedDepth }, i) => {
    if (numberedDepth !== null) return { id, text, depth: numberedDepth };
    const prev = prevNumbered[i];
    const next = nextNumbered[i];
    let depth: number;
    if (prev === null) depth = 0;
    else if (next === null) depth = prev;
    else if (next > prev) depth = next;
    else depth = prev + 1;
    return { id, text, depth };
  });
}
