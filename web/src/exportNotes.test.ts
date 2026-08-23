import { describe, expect, it } from "vitest";
import type { LibraryAnnotation } from "./api";
import { groupByPaper, notesToMarkdown } from "./exportNotes";

function note(overrides: Partial<LibraryAnnotation> & { id: string }): LibraryAnnotation {
  return {
    snippet: "a passage",
    section_title: "Method",
    section_slug: "method",
    note: "",
    created_at: "2026-01-01T00:00:00",
    updated_at: "2026-01-01T00:00:00",
    paper_id: "paper-a",
    paper_title: "Paper A",
    ...overrides,
  };
}

describe("groupByPaper", () => {
  it("groups by paper, sorted alphabetically by title, notes newest-first", () => {
    const notes = [
      note({ id: "1", paper_id: "b", paper_title: "Paper B", created_at: "2026-01-01T00:00:00" }),
      note({ id: "2", paper_id: "a", paper_title: "Paper A", created_at: "2026-01-01T00:00:00" }),
      note({ id: "3", paper_id: "a", paper_title: "Paper A", created_at: "2026-01-03T00:00:00" }),
    ];
    const groups = groupByPaper(notes);
    expect(groups.map((g) => g.paperTitle)).toEqual(["Paper A", "Paper B"]);
    expect(groups[0].notes.map((n) => n.id)).toEqual(["3", "2"]); // newest first
  });

  it("returns an empty list for no notes", () => {
    expect(groupByPaper([])).toEqual([]);
  });
});

describe("notesToMarkdown", () => {
  it("returns an empty string for no notes", () => {
    expect(notesToMarkdown([])).toBe("");
  });

  it("renders a single paper/note with the arXiv id in the heading", () => {
    const notes = [
      note({
        id: "1",
        paper_title: "DeepSeek-V3",
        arxiv_id: "2412.19437",
        snippet: "MLA shrinks the cache",
        section_title: "Attention",
        note: "check against Table 3",
      }),
    ];
    const out = notesToMarkdown(notes);
    expect(out).toContain("# My Notes");
    expect(out).toContain("## DeepSeek-V3 (arXiv:2412.19437)");
    expect(out).toContain('> "MLA shrinks the cache" — Attention');
    expect(out).toContain("check against Table 3");
  });

  it("omits the arXiv parenthetical when arxiv_id is absent", () => {
    const out = notesToMarkdown([note({ id: "1", paper_title: "No ID Paper" })]);
    expect(out).toContain("## No ID Paper\n");
    expect(out).not.toContain("(arXiv:");
  });

  it("renders a note-less annotation without a dangling blank note line", () => {
    const out = notesToMarkdown([note({ id: "1", note: "" })]);
    expect(out).toContain('> "a passage" — Method');
    // No trailing blank-then-empty-line artifact from an empty note body.
    expect(out.trimEnd().endsWith("— Method")).toBe(true);
  });

  it("renders one heading per paper, in alphabetical order", () => {
    const notes = [
      note({ id: "1", paper_id: "b", paper_title: "Paper B" }),
      note({ id: "2", paper_id: "a", paper_title: "Paper A" }),
    ];
    const out = notesToMarkdown(notes);
    expect(out.indexOf("## Paper A")).toBeLessThan(out.indexOf("## Paper B"));
  });
});
