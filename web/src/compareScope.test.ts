import { describe, expect, it } from "vitest";
import type { Paper } from "./api";
import { resolveScopeIds, resolveScopeSize } from "./compareScope";

const papers: Paper[] = [
  { paper_id: "a", title: "A", tags: ["x"] },
  { paper_id: "b", title: "B", tags: ["x", "y"] },
  { paper_id: "c", title: "C", tags: ["y"] },
];

describe("resolveScopeIds", () => {
  it("returns null when no filter is active (the entire library)", () => {
    expect(resolveScopeIds(papers, [], [])).toBeNull();
  });

  it("returns the selected papers when only a paper filter is set", () => {
    expect(resolveScopeIds(papers, [], ["a", "c"])).toEqual(["a", "c"]);
  });

  it("returns the papers carrying a tag when only a tag filter is set", () => {
    expect(resolveScopeIds(papers, ["y"], [])).toEqual(["b", "c"]);
  });

  it("intersects tag and paper filters", () => {
    // tag "y" -> {b, c}; paper filter {a, b} -> intersection {b}
    expect(resolveScopeIds(papers, ["y"], ["a", "b"])).toEqual(["b"]);
  });
});

describe("resolveScopeSize", () => {
  it("counts the whole library when no filter is active", () => {
    expect(resolveScopeSize(papers, [], [])).toBe(3);
  });

  it("counts the resolved intersection when filters are active", () => {
    expect(resolveScopeSize(papers, ["y"], ["a", "b"])).toBe(1);
  });
});
