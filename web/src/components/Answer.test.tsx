import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { Citation } from "../api";
import Answer from "./Answer";

function cite(ref: string, paperId: string, title: string): Citation {
  return {
    ref,
    paper_id: paperId,
    title,
    breadcrumb: "",
    section_title: "Method",
    snippet: "some passage",
  };
}

function renderAnswer(text: string, citations: Citation[]) {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <Answer text={text} citations={citations} />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("Answer citation marker rendering", () => {
  it("renders a single ASCII [rN] marker as a clickable citation", () => {
    renderAnswer("A claim [r1].", [cite("r1", "p1", "Paper One")]);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("renders a fullwidth 【rN】 marker as a clickable citation", () => {
    // A real local model has been observed emitting 【rN】 instead of [rN].
    renderAnswer("A claim【r1】.", [cite("r1", "p1", "Paper One")]);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("renders a comma-bunched [rN, rM] marker as two separate clickable citations", () => {
    // Real observed case: a per-paper Compare row answer wrote [r10, r12] instead of
    // [r10][r12] — must render as two distinct citation badges, same as if the model
    // had written them separately.
    renderAnswer("284 B total, 13 B activated [r10, r12].", [
      cite("r10", "p1", "Paper One"),
      cite("r12", "p2", "Paper Two"),
    ]);
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
  });

  it("falls back to plain text for a marker with no matching citation", () => {
    // Answer.tsx must degrade gracefully for an unrecognized ref, not crash — e.g. a
    // synthesized answer hallucinating a ref that isn't in the union citations list.
    renderAnswer("A claim [r99].", []);
    expect(screen.getByText(/A claim/)).toBeInTheDocument();
    expect(screen.queryByText("99")).not.toBeInTheDocument();
  });

  it("renders every ref in a bunch when all of them resolve", () => {
    renderAnswer("[r10, r12] combined.", [
      cite("r10", "p1", "Paper One"),
      cite("r12", "p2", "Paper Two"),
    ]);
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
  });

  it("falls back to plain text for the whole bunch when one ref in it doesn't resolve", () => {
    // A mixed bunch (one real ref, one hallucinated) must not silently drop the
    // unresolved ref's text while linking the rest — same "never erase text the
    // model wrote" principle as the lone-unresolved-ref fallback above. The whole
    // bracket falls back to plain text, not a partial render.
    renderAnswer("[r10, r99] combined.", [cite("r10", "p1", "Paper One")]);
    expect(screen.queryByText("10")).not.toBeInTheDocument();
    expect(screen.getByText(/r10, r99/)).toBeInTheDocument();
  });
});
