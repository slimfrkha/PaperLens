// Best-effort highlight of a cited passage inside a rendered paper.
// Uses the CSS Custom Highlight API so we can span multiple DOM nodes without
// mutating the tree. Falls back to scrolling to the nearest section heading.
//
// Two independent named highlight groups live here:
// - "citation": one transient range at a time, for the click-a-citation jump.
// - "annotation": many persistent ranges at once, for saved user annotations.

function normalize(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/** Build a flat map of the container's text plus (node, offset) back-references. */
function textMap(container: HTMLElement) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let text = "";
  const spans: { node: Text; start: number; end: number }[] = [];
  let n: Node | null;
  while ((n = walker.nextNode())) {
    const t = n as Text;
    const start = text.length;
    text += t.data;
    spans.push({ node: t, start, end: text.length });
  }
  return { text, spans };
}

function rangeFor(container: HTMLElement, query: string, bounds?: [number, number]): Range | null {
  const { text, spans } = textMap(container);
  const [from, to] = bounds ?? [0, text.length];
  const window = text.slice(from, to);
  // Try decreasing prefixes of the snippet for resilience to markdown reflow.
  const norm = normalize(query);
  for (const len of [220, 160, 110, 70, 40]) {
    const needle = norm.slice(0, len);
    if (needle.length < 20) break;
    // Search on a whitespace-collapsed copy while keeping an index back-map.
    const idx = collapsedIndexOf(window, needle);
    if (idx < 0) continue;
    const startAbs = from + idx;
    const endAbs = startAbs + rawLengthFor(window, idx, needle.length);
    const s = spans.find((sp) => startAbs >= sp.start && startAbs < sp.end);
    const e = spans.find((sp) => endAbs > sp.start && endAbs <= sp.end);
    if (!s || !e) continue;
    const range = document.createRange();
    range.setStart(s.node, startAbs - s.start);
    range.setEnd(e.node, endAbs - e.start);
    return range;
  }
  return null;
}

// Find `needle` (already whitespace-collapsed) inside raw `text`, returning the
// raw start index, tolerating runs of whitespace in the source.
function collapsedIndexOf(text: string, needle: string): number {
  const re = new RegExp(
    needle
      .split(" ")
      .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("\\s+"),
  );
  const m = re.exec(text);
  return m ? m.index : -1;
}

function rawLengthFor(text: string, start: number, approxLen: number): number {
  // Extend to the end of the collapsed match by re-running the regex from start.
  const slice = text.slice(start, start + approxLen * 3);
  return Math.min(slice.length, approxLen * 3);
}

// Find the [start, end) character bounds (in textMap's flat text) of the section headed
// by `sectionSlug` — from its own heading text up to (not including) the next heading at
// the same or shallower level, or the end of the container for the last section. Markdown
// headings/paragraphs are flat siblings (no wrapping <section> element), so this is done by
// offset arithmetic over the flat text rather than DOM containment. Returns null if the
// heading id no longer exists (e.g. the paper was re-extracted with different headings).
function sectionBounds(
  container: HTMLElement,
  sectionSlug: string,
  text: string,
  spans: { node: Text; start: number; end: number }[],
): [number, number] | null {
  const heading = document.getElementById(sectionSlug);
  if (!heading || !container.contains(heading)) return null;
  const level = Number(heading.tagName[1]) || 6;
  const headings = Array.from(container.querySelectorAll("h1, h2, h3, h4, h5, h6"));
  const idx = headings.indexOf(heading);

  const startSpans = spans.filter((sp) => heading.contains(sp.node));
  if (startSpans.length === 0) return null;
  const start = Math.min(...startSpans.map((sp) => sp.start));

  const next = headings.slice(idx + 1).find((h) => Number(h.tagName[1]) <= level);
  let end = text.length;
  if (next) {
    const endSpans = spans.filter((sp) => next.contains(sp.node));
    if (endSpans.length > 0) end = Math.min(...endSpans.map((sp) => sp.start));
  }
  return [start, end];
}

/** Resolve an annotation's anchor, scoped to its section when possible so a phrase
 *  recurring elsewhere in the paper can't cause it to attach to the wrong occurrence.
 *  Falls back to a whole-document search if the section heading no longer resolves. */
function rangeForAnnotation(
  container: HTMLElement,
  snippet: string,
  sectionSlug?: string,
): Range | null {
  if (sectionSlug) {
    const { text, spans } = textMap(container);
    const bounds = sectionBounds(container, sectionSlug, text, spans);
    if (bounds) {
      const scoped = rangeFor(container, snippet, bounds);
      if (scoped) return scoped;
    }
  }
  return rangeFor(container, snippet);
}

// TS's DOM lib types HighlightRegistry with only `forEach`, missing the set()/delete()
// methods the CSS Custom Highlight API actually defines — fill the gap via declaration merging.
declare global {
  interface HighlightRegistry {
    set(name: string, highlight: Highlight): HighlightRegistry;
    delete(name: string): boolean;
  }
}

export function highlightPassage(container: HTMLElement, snippet: string, sectionSlug?: string) {
  const HL = window.Highlight as typeof Highlight | undefined; // absent in older browsers
  const range = rangeFor(container, snippet);
  if (range && HL && CSS.highlights) {
    CSS.highlights.set("citation", new HL(range));
    (range.startContainer.parentElement || container).scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
    return;
  }
  if (sectionSlug) {
    const el = document.getElementById(sectionSlug);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      el.style.backgroundColor = "rgba(255, 241, 158, 0.18)";
    }
  }
}

export function clearHighlight() {
  if (CSS.highlights) CSS.highlights.delete("citation");
}

// --- Persistent, multi-range "annotation" highlight group ---
// Unlike "citation" (one transient range, replaced wholesale each call), many annotation
// ranges must render at once. `Highlight` is Set-like (`.add()`/`.delete()`/`.clear()`), so
// one instance is registered once and mutated as annotations are added/removed.
const annotationRanges = new Map<string, Range>();
let annotationHighlight: Highlight | undefined;

function ensureAnnotationHighlight(): Highlight | undefined {
  const HL = window.Highlight as typeof Highlight | undefined;
  if (!HL || !CSS.highlights) return undefined;
  if (!annotationHighlight) {
    annotationHighlight = new HL();
    CSS.highlights.set("annotation", annotationHighlight);
  }
  return annotationHighlight;
}

/** Resolve and register an annotation's highlight range. Re-registering an id (e.g. after
 *  a re-render) replaces its previous range. Returns the resolved range, or null if the
 *  snippet no longer matches anywhere in the container. */
export function addAnnotationHighlight(
  id: string,
  container: HTMLElement,
  snippet: string,
  sectionSlug?: string,
): Range | null {
  const highlight = ensureAnnotationHighlight();
  if (!highlight) return null;
  const range = rangeForAnnotation(container, snippet, sectionSlug);
  if (!range) return null;
  removeAnnotationHighlight(id);
  annotationRanges.set(id, range);
  highlight.add(range);
  return range;
}

/** Register an already-resolved Range directly (e.g. the browser's live Selection range
 *  right after creating an annotation), bypassing snippet re-resolution — the exact text
 *  the user selected is already known, no ambiguity to resolve. */
export function registerAnnotationRange(id: string, range: Range): void {
  const highlight = ensureAnnotationHighlight();
  if (!highlight) return;
  removeAnnotationHighlight(id);
  annotationRanges.set(id, range);
  highlight.add(range);
}

export function removeAnnotationHighlight(id: string): void {
  const range = annotationRanges.get(id);
  if (range && annotationHighlight) annotationHighlight.delete(range);
  annotationRanges.delete(id);
}

export function getAnnotationRange(id: string): Range | undefined {
  return annotationRanges.get(id);
}

export function clearAnnotationHighlights(): void {
  annotationHighlight?.clear();
  annotationRanges.clear();
}

/** The nearest heading at or before `node` in document order — used to tag a fresh
 *  selection with the section it was made in, so the annotation can later be re-anchored
 *  scoped to that section instead of searching the whole document. */
export function sectionForNode(
  container: HTMLElement,
  node: Node,
): { slug: string; title: string } | null {
  const headings = Array.from(
    container.querySelectorAll("h1, h2, h3, h4, h5, h6"),
  ) as HTMLElement[];
  let heading: HTMLElement | null = null;
  for (const h of headings) {
    const pos = node.compareDocumentPosition(h);
    if (pos & Node.DOCUMENT_POSITION_FOLLOWING) break;
    heading = h;
  }
  if (!heading || !heading.id) return null;
  return { slug: heading.id, title: heading.textContent?.trim() ?? "" };
}

/** Find which registered annotation (if any) contains the given viewport point —
 *  used to open the view/edit popover on click, since a Custom Highlight isn't a DOM
 *  node a click event can target directly. */
export function annotationIdAtPoint(clientX: number, clientY: number): string | undefined {
  const doc = document as Document & {
    caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
  };
  let node: Node | null = null;
  let offset = 0;
  const caret = doc.caretPositionFromPoint?.(clientX, clientY);
  if (caret) {
    node = caret.offsetNode;
    offset = caret.offset;
  } else if (document.caretRangeFromPoint) {
    const r = document.caretRangeFromPoint(clientX, clientY);
    if (r) {
      node = r.startContainer;
      offset = r.startOffset;
    }
  }
  if (!node) return undefined;
  for (const [id, range] of annotationRanges) {
    if (range.isPointInRange(node, offset)) return id;
  }
  return undefined;
}
