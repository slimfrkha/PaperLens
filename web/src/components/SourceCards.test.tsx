import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { Citation } from "../api";
import SourceCards from "./SourceCards";

const cite = (
  ref: string,
  paper_id: string,
  title: string,
  source?: Citation["source"],
): Citation => ({
  ref,
  paper_id,
  title,
  breadcrumb: "",
  section_title: "Method",
  snippet: "some passage",
  source,
});

function renderCards(citations: Citation[]) {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <SourceCards citations={citations} />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("SourceCards", () => {
  it("renders nothing when there are no citations", () => {
    renderCards([]);
    expect(screen.queryByText("Sources")).not.toBeInTheDocument();
  });

  it("collapses multiple citations of one paper into a single card", () => {
    renderCards([cite("r1", "p1", "Paper One"), cite("r3", "p1", "Paper One")]);
    // One title, both citation numbers shown on it.
    expect(screen.getAllByText("Paper One")).toHaveLength(1);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("renders one card per distinct paper", () => {
    renderCards([cite("r1", "p1", "Paper One"), cite("r2", "p2", "Paper Two")]);
    expect(screen.getByText("Paper One")).toBeInTheDocument();
    expect(screen.getByText("Paper Two")).toBeInTheDocument();
  });

  it("renders a lexical/hybrid citation's number with no visible marker", () => {
    renderCards([
      cite("r1", "p1", "Paper One", "sparse"),
      cite("r2", "p2", "Paper Two", "both"),
      cite("r3", "p3", "Paper Three", "dense"),
      cite("r4", "p4", "Paper Four"), // no `source` at all — old persisted chats
    ]);
    // `source` never changes what's rendered next to the number — that signal lives only in
    // the number's hover tooltip, not in the DOM as text (see SourceCards.tsx for why: the
    // faithfulness `!` flag already owns the number's color/glyph, and a second glyph for
    // `source` reads as noise stacked next to it, e.g. "4!K").
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });
});
