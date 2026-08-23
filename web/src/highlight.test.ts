import { afterEach, beforeAll, describe, expect, it } from "vitest";
import {
  addAnnotationHighlight,
  annotationIdAtPoint,
  clearAnnotationHighlights,
  getAnnotationRange,
  registerAnnotationRange,
  removeAnnotationHighlight,
  sectionForNode,
} from "./highlight";

// jsdom doesn't implement the CSS Custom Highlight API — install a minimal Set-like
// polyfill (matching the real Highlight/HighlightRegistry contract) so the module's
// guards (`window.Highlight`, `CSS.highlights`) don't short-circuit every call.
class FakeHighlight {
  ranges = new Set<Range>();
  constructor(...ranges: Range[]) {
    ranges.forEach((r) => this.ranges.add(r));
  }
  add(range: Range) {
    this.ranges.add(range);
    return this;
  }
  delete(range: Range) {
    return this.ranges.delete(range);
  }
  clear() {
    this.ranges.clear();
  }
  has(range: Range) {
    return this.ranges.has(range);
  }
  get size() {
    return this.ranges.size;
  }
}

beforeAll(() => {
  (window as unknown as { Highlight: typeof FakeHighlight }).Highlight = FakeHighlight;
  const registry = new Map<string, FakeHighlight>();
  (CSS as unknown as { highlights: unknown }).highlights = {
    set(name: string, hl: FakeHighlight) {
      registry.set(name, hl);
      return this;
    },
    delete: (name: string) => registry.delete(name),
    get: (name: string) => registry.get(name),
  };
});

afterEach(() => {
  clearAnnotationHighlights();
  document.body.innerHTML = "";
});

function buildContainer(html: string): HTMLDivElement {
  const el = document.createElement("div");
  el.innerHTML = html;
  document.body.appendChild(el);
  return el;
}

describe("addAnnotationHighlight", () => {
  it("ends the range exactly at the matched text, not past it into what follows", () => {
    // Regression: rangeFor used to size the match as `needle.length * 3` raw
    // characters (a blanket guess meant to tolerate whitespace differences) instead of
    // the regex's actual matched length, overshooting into unrelated trailing text
    // whenever the container had more text after the snippet.
    const el = buildContainer(
      "<p>A short snippet worth remembering. And then a lot more unrelated trailing text follows it.</p>",
    );
    const range = addAnnotationHighlight("a1", el, "A short snippet worth remembering.");
    expect(range).not.toBeNull();
    expect(range!.toString()).toBe("A short snippet worth remembering.");
  });

  it("resolves a snippet against the whole document when no section is given", () => {
    const el = buildContainer("<p>Some passage worth remembering here.</p>");
    const range = addAnnotationHighlight("a1", el, "Some passage worth remembering here.");
    expect(range).not.toBeNull();
    expect(getAnnotationRange("a1")).toBe(range);
  });

  it("scopes re-anchoring to the annotated section, not the first occurrence in the document", () => {
    // Regression: a phrase repeated verbatim in two sections used to always resolve to
    // the FIRST occurrence (rangeFor's plain first-match search) — silently attaching a
    // note made in section two to the wrong (section one) passage on every reload.
    const el = buildContainer(`
      <h2 id="section-one">Section One</h2>
      <p>This is a repeated sentence for testing.</p>
      <h2 id="section-two">Section Two</h2>
      <p>This is a repeated sentence for testing.</p>
    `);

    const range = addAnnotationHighlight(
      "a1",
      el,
      "This is a repeated sentence for testing.",
      "section-two",
    );
    expect(range).not.toBeNull();

    const sectionTwoHeading = el.querySelector("#section-two")!;
    const afterSectionTwo = document.createRange();
    afterSectionTwo.selectNode(sectionTwoHeading);
    afterSectionTwo.setEndAfter(el.lastElementChild!);

    expect(afterSectionTwo.isPointInRange(range!.startContainer, range!.startOffset)).toBe(true);
  });

  it("falls back to a whole-document search when the section heading no longer exists", () => {
    const el = buildContainer("<p>Only one occurrence of this passage exists.</p>");
    const range = addAnnotationHighlight(
      "a1",
      el,
      "Only one occurrence of this passage exists.",
      "section-that-was-removed",
    );
    expect(range).not.toBeNull();
  });

  it("returns null when the snippet no longer matches anywhere", () => {
    const el = buildContainer("<p>Completely different text.</p>");
    const range = addAnnotationHighlight("a1", el, "This text was never here at all really.");
    expect(range).toBeNull();
  });

  it("re-registering the same id replaces its range rather than accumulating", () => {
    const el = buildContainer("<p>First passage here. Second passage here.</p>");
    addAnnotationHighlight("a1", el, "First passage here.");
    const second = addAnnotationHighlight("a1", el, "Second passage here.");

    expect(getAnnotationRange("a1")).toBe(second);
  });
});

describe("registerAnnotationRange", () => {
  it("stores the exact Range given, without re-resolving it against the document", () => {
    const el = buildContainer("<p>abc abc abc</p>"); // ambiguous text on purpose
    const textNode = el.querySelector("p")!.firstChild!;
    const liveRange = document.createRange();
    liveRange.setStart(textNode, 4);
    liveRange.setEnd(textNode, 7);

    registerAnnotationRange("a1", liveRange);
    expect(getAnnotationRange("a1")).toBe(liveRange);
  });
});

describe("removeAnnotationHighlight", () => {
  it("removes the annotation from the registry", () => {
    const el = buildContainer("<p>Passage to remove later.</p>");
    addAnnotationHighlight("a1", el, "Passage to remove later.");
    expect(getAnnotationRange("a1")).toBeDefined();

    removeAnnotationHighlight("a1");
    expect(getAnnotationRange("a1")).toBeUndefined();
  });
});

describe("clearAnnotationHighlights", () => {
  it("removes every registered range, not just one", () => {
    const el = buildContainer(
      "<p>The first passage worth remembering. The second passage worth remembering.</p>",
    );
    addAnnotationHighlight("a1", el, "The first passage worth remembering.");
    addAnnotationHighlight("a2", el, "The second passage worth remembering.");
    expect(getAnnotationRange("a1")).toBeDefined();
    expect(getAnnotationRange("a2")).toBeDefined();

    clearAnnotationHighlights();

    expect(getAnnotationRange("a1")).toBeUndefined();
    expect(getAnnotationRange("a2")).toBeUndefined();
  });
});

describe("sectionForNode", () => {
  it("finds the nearest preceding heading", () => {
    const el = buildContainer(`
      <h2 id="intro">Introduction</h2>
      <p>Some text in the intro.</p>
      <h2 id="method">Method</h2>
      <p>Some text in the method section.</p>
    `);
    const methodParagraph = el.querySelectorAll("p")[1].firstChild!;

    expect(sectionForNode(el, methodParagraph)).toEqual({ slug: "method", title: "Method" });
  });

  it("returns null when no heading precedes the node", () => {
    const el = buildContainer("<p>Text with no heading before it.</p>");
    const node = el.querySelector("p")!.firstChild!;

    expect(sectionForNode(el, node)).toBeNull();
  });
});

describe("annotationIdAtPoint", () => {
  it("finds the registered annotation whose range contains the point", () => {
    const el = buildContainer("<p>Find me at this exact point.</p>");
    const range = addAnnotationHighlight("a1", el, "Find me at this exact point.")!;

    document.caretRangeFromPoint = () =>
      (() => {
        const r = document.createRange();
        r.setStart(range.startContainer, range.startOffset);
        r.setEnd(range.startContainer, range.startOffset);
        return r;
      })();

    expect(annotationIdAtPoint(0, 0)).toBe("a1");
  });

  it("returns undefined when the point falls outside every registered range", () => {
    const el = buildContainer("<p>Some unrelated paragraph text.</p>");
    const node = el.querySelector("p")!.firstChild!;
    document.caretRangeFromPoint = () => {
      const r = document.createRange();
      r.setStart(node, 0);
      r.setEnd(node, 0);
      return r;
    };

    expect(annotationIdAtPoint(0, 0)).toBeUndefined();
  });
});
