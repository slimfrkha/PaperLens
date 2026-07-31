import { describe, expect, it } from "vitest";
import type { Citation } from "./api";
import {
  answerToBibtex,
  answerToMarkdown,
  citedCitations,
  escapeBibtex,
  extractCitedRefs,
} from "./exportAnswer";

function citation(overrides: Partial<Citation> & { ref: string; paper_id: string }): Citation {
  return {
    title: "A Paper",
    breadcrumb: "b",
    section_title: "Method",
    snippet: "sn",
    ...overrides,
  };
}

describe("extractCitedRefs / citedCitations", () => {
  it("keeps only refs present in the map, deduped, in first-appearance order", () => {
    const byRef = new Map([
      ["r1", citation({ ref: "r1", paper_id: "p1" })],
      ["r2", citation({ ref: "r2", paper_id: "p2" })],
    ]);
    const text = "See [r2] and again [r2], also [r1], but not [r9].";
    expect(extractCitedRefs(text, byRef)).toEqual(["r2", "r1"]);
  });

  it("ignores registry entries that were retrieved but never inline-cited", () => {
    const citations = [
      citation({ ref: "r1", paper_id: "p1" }),
      citation({ ref: "r2", paper_id: "p2" }),
    ];
    const text = "Only [r1] is actually cited in the answer.";
    expect(citedCitations(text, citations).map((c) => c.ref)).toEqual(["r1"]);
  });
});

describe("answerToMarkdown", () => {
  it("returns the text unchanged when nothing was cited", () => {
    const text = "Hi there, how can I help?";
    expect(answerToMarkdown(text, [])).toBe(text);
  });

  it("converts [rN] to [^N] footnotes and appends a References block with an arXiv link", () => {
    const cited = [
      citation({
        ref: "r3",
        paper_id: "deepseek-v3",
        title: "DeepSeek-V3",
        section_title: "Attention",
        arxiv_id: "2412.19437",
      }),
    ];
    const out = answerToMarkdown("MLA shrinks the cache [r3].", cited);
    expect(out).toContain("MLA shrinks the cache [^3].");
    expect(out).toContain("## References");
    expect(out).toContain("[^3]: **DeepSeek-V3** — Attention, https://arxiv.org/abs/2412.19437");
  });

  it("omits the arXiv link when arxiv_id is missing, and sorts footnotes numerically", () => {
    const cited = [
      citation({ ref: "r10", paper_id: "p10", title: "Ten" }),
      citation({ ref: "r2", paper_id: "p2", title: "Two" }),
    ];
    const out = answerToMarkdown("[r10] then [r2]", cited);
    const refLines = out.split("\n").filter((l) => /^\[\^\d+\]:/.test(l));
    expect(refLines).toEqual(["[^2]: **Two** — Method", "[^10]: **Ten** — Method"]);
  });
});

describe("escapeBibtex", () => {
  it("escapes braces, ampersand, and percent so the entry stays balanced", () => {
    expect(escapeBibtex("A {B} & C% Report")).toBe("A \\{B\\} \\& C\\% Report");
  });
});

describe("answerToBibtex", () => {
  it("emits one entry per distinct paper_id, deduped across multiple refs", () => {
    const cited = [
      citation({ ref: "r1", paper_id: "p1", title: "Paper One", arxiv_id: "2412.19437" }),
      citation({ ref: "r2", paper_id: "p1", title: "Paper One", arxiv_id: "2412.19437" }),
    ];
    const out = answerToBibtex(cited);
    expect(out.match(/@misc\{/g)).toHaveLength(1);
    expect(out).toContain("@misc{p1,");
    expect(out).toContain("title = {Paper One}");
    expect(out).toContain("eprint = {2412.19437}");
    expect(out).toContain("archivePrefix = {arXiv}");
    expect(out).toContain("year = {2024}");
  });

  it("omits eprint/archivePrefix/year gracefully when arxiv_id is absent or old-style", () => {
    const cited = [citation({ ref: "r1", paper_id: "p1", title: "No arXiv ID" })];
    const out = answerToBibtex(cited);
    expect(out).not.toContain("eprint");
    expect(out).not.toContain("archivePrefix");
    expect(out).not.toContain("year");

    const oldStyle = [
      citation({ ref: "r1", paper_id: "p2", title: "Old style", arxiv_id: "hep-th/9901001" }),
    ];
    expect(answerToBibtex(oldStyle)).not.toContain("year");
  });

  it("escapes special characters in the title", () => {
    const cited = [citation({ ref: "r1", paper_id: "p1", title: "50% Faster {Attention}" })];
    expect(answerToBibtex(cited)).toContain("title = {50\\% Faster \\{Attention\\}}");
  });
});
