import { afterEach, describe, expect, it } from "vitest";
import { buildOutline } from "./outline";

afterEach(() => {
  document.body.innerHTML = "";
});

function buildContainer(html: string): HTMLDivElement {
  const el = document.createElement("div");
  el.innerHTML = html;
  document.body.appendChild(el);
  return el;
}

describe("buildOutline", () => {
  it("derives depth from the count of dot-segments in a numbered heading", () => {
    const el = buildContainer(`
      <h2 id="s1">1 Introduction</h2>
      <h2 id="s1-1">1.1 Related Work</h2>
      <h2 id="s1-1-1">1.1.1 A Subsection</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "s1", text: "1 Introduction", depth: 1 },
      { id: "s1-1", text: "1.1 Related Work", depth: 2 },
      { id: "s1-1-1", text: "1.1.1 A Subsection", depth: 3 },
    ]);
  });

  it("renders an unnumbered heading at depth 0 before any numbered heading has appeared", () => {
    const el = buildContainer(`
      <h2 id="abstract">Abstract</h2>
      <h2 id="s1">1 Introduction</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "abstract", text: "Abstract", depth: 0 },
      { id: "s1", text: "1 Introduction", depth: 1 },
    ]);
  });

  it("matches the next numbered heading's depth when it's deeper than the previous one", () => {
    // An unnumbered "Contributions" heading right before "1.1 Related Work" is introducing
    // that deeper content, so it takes on the same depth rather than sitting shallower.
    const el = buildContainer(`
      <h2 id="s1">1 Introduction</h2>
      <h2 id="contrib">Contributions</h2>
      <h2 id="s1-1">1.1 Related Work</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "s1", text: "1 Introduction", depth: 1 },
      { id: "contrib", text: "Contributions", depth: 2 },
      { id: "s1-1", text: "1.1 Related Work", depth: 2 },
    ]);
  });

  it("nests one level under the previous numbered heading when the next one is same/shallower", () => {
    const el = buildContainer(`
      <h2 id="s2">2 Method</h2>
      <h2 id="overview">Overview</h2>
      <h2 id="s3">3 Results</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "s2", text: "2 Method", depth: 1 },
      { id: "overview", text: "Overview", depth: 2 },
      { id: "s3", text: "3 Results", depth: 1 },
    ]);
  });

  it("nests consecutive unnumbered headings as siblings under the same rule", () => {
    const el = buildContainer(`
      <h2 id="s1">1 Introduction</h2>
      <h2 id="contrib">Contributions</h2>
      <h2 id="summary">Summary of Evaluation Results</h2>
      <h2 id="s2">2 Related Work</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "s1", text: "1 Introduction", depth: 1 },
      { id: "contrib", text: "Contributions", depth: 2 },
      { id: "summary", text: "Summary of Evaluation Results", depth: 2 },
      { id: "s2", text: "2 Related Work", depth: 1 },
    ]);
  });

  it("treats a trailing unnumbered heading as a sibling of the last numbered one, not its child", () => {
    // Regression guard for a rejected heuristic (look-behind only): nesting a trailing
    // unnumbered section one level under whatever numbered heading precedes it confidently
    // misplaces things like an "Appendix"/"References" after "5 Conclusion" as that
    // section's own child, when they're really top-level entries in their own right.
    const el = buildContainer(`
      <h2 id="s5">5 Conclusion</h2>
      <h2 id="appendix">Appendix</h2>
      <h2 id="refs">References</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "s5", text: "5 Conclusion", depth: 1 },
      { id: "appendix", text: "Appendix", depth: 1 },
      { id: "refs", text: "References", depth: 1 },
    ]);
  });

  it("renders every heading at depth 0 when a paper has no numbered sections at all", () => {
    const el = buildContainer(`
      <h2 id="overview">Overview</h2>
      <h2 id="details">Details</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "overview", text: "Overview", depth: 0 },
      { id: "details", text: "Details", depth: 0 },
    ]);
  });

  it("still parses depth with a trailing period after the number", () => {
    const el = buildContainer(`<h2 id="s2-1">2.1. Method</h2>`);
    expect(buildOutline(el)).toEqual([{ id: "s2-1", text: "2.1. Method", depth: 2 }]);
  });

  it("skips a heading with no id", () => {
    const el = buildContainer(`
      <h2>No Id Here</h2>
      <h2 id="s1">1 Introduction</h2>
    `);
    expect(buildOutline(el)).toEqual([{ id: "s1", text: "1 Introduction", depth: 1 }]);
  });

  it("keeps both entries of a duplicate heading, rehype-slug-disambiguated ids untouched", () => {
    const el = buildContainer(`
      <h2 id="overview">Overview</h2>
      <h2 id="overview-1">Overview</h2>
    `);
    expect(buildOutline(el)).toEqual([
      { id: "overview", text: "Overview", depth: 0 },
      { id: "overview-1", text: "Overview", depth: 0 },
    ]);
  });

  it("returns an empty list for a container with no headings", () => {
    const el = buildContainer(`<p>Just a paragraph, no headings.</p>`);
    expect(buildOutline(el)).toEqual([]);
  });
});
