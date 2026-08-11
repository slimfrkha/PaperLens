import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { Citation } from "../api";
import SourceCards from "./SourceCards";

const cite = (
  ref: string,
  paper_id: string,
  title: string,
  source?: Citation["source"],
  section_title = "Method",
  snippet = "some passage",
): Citation => ({
  ref,
  paper_id,
  title,
  breadcrumb: "",
  section_title,
  snippet,
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

// Probe for the destination route: dumps the `highlight`/`section` nav state as text so a
// click's target can be asserted on without mocking `useNavigate`.
function LocationProbe() {
  const location = useLocation() as { state?: { highlight?: string; section?: string } };
  return <div data-testid="probe">{JSON.stringify(location.state)}</div>;
}

function renderCardsWithRoute(citations: Citation[]) {
  return render(
    <MantineProvider>
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route path="/" element={<SourceCards citations={citations} />} />
          <Route path="/papers/:id" element={<LocationProbe />} />
        </Routes>
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

  it("opens each number at its own section, not the card's first-cited one", () => {
    renderCardsWithRoute([
      cite("r3", "p1", "Paper One", undefined, "5.1 Post-Training Pipeline", "snippet for r3"),
      cite("r6", "p1", "Paper One", undefined, "1 Introduction", "snippet for r6"),
    ]);

    fireEvent.click(screen.getByText("6"));
    expect(screen.getByTestId("probe")).toHaveTextContent(
      JSON.stringify({ highlight: "snippet for r6", section: "1 Introduction" }),
    );
  });

  it("opens the paper with no highlight when the card itself is clicked (not a number)", () => {
    renderCardsWithRoute([
      cite("r3", "p1", "Paper One", undefined, "5.1 Post-Training Pipeline", "snippet for r3"),
      cite("r6", "p1", "Paper One", undefined, "1 Introduction", "snippet for r6"),
    ]);

    fireEvent.click(screen.getByText("Paper One"));
    // react-router gives an un-navigated-with-state location a `null` state, not `undefined`.
    expect(screen.getByTestId("probe").textContent).toBe(JSON.stringify(null));
  });
});
