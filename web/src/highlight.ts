// Best-effort highlight of a cited passage inside a rendered paper.
// Uses the CSS Custom Highlight API so we can span multiple DOM nodes without
// mutating the tree. Falls back to scrolling to the nearest section heading.

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

function rangeFor(container: HTMLElement, query: string): Range | null {
  const { text, spans } = textMap(container);
  // Try decreasing prefixes of the snippet for resilience to markdown reflow.
  const norm = normalize(query);
  for (const len of [220, 160, 110, 70, 40]) {
    const needle = norm.slice(0, len);
    if (needle.length < 20) break;
    // Search on a whitespace-collapsed copy while keeping an index back-map.
    const idx = collapsedIndexOf(text, needle);
    if (idx < 0) continue;
    const startAbs = idx;
    const endAbs = idx + rawLengthFor(text, idx, needle.length);
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
