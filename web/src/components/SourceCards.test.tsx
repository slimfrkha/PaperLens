import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { Citation } from "../api";
import SourceCards from "./SourceCards";

const cite = (ref: string, paper_id: string, title: string): Citation => ({
  ref,
  paper_id,
  title,
  breadcrumb: "",
  section_title: "Method",
  snippet: "some passage",
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
});
