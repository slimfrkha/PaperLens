import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { CompareRow } from "../api";
import ComparePanel from "./ComparePanel";

function row(paperId: string, title: string, text: string): CompareRow {
  return {
    paper_id: paperId,
    title,
    arxiv_id: null,
    text,
    citations: [
      {
        ref: `r-${paperId}`,
        paper_id: paperId,
        title,
        breadcrumb: "",
        section_title: "Method",
        snippet: "some passage",
      },
    ],
    trace: [{ type: "action", query: "q" }],
  };
}

function renderPanel(props: Partial<React.ComponentProps<typeof ComparePanel>> = {}) {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <ComparePanel rows={[]} totalPapers={2} {...props} />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("ComparePanel progress badge", () => {
  it("shows 'Searching paper N of M…' while rows are still incomplete", () => {
    renderPanel({ rows: [row("p1", "Paper One", "Answer one.")], totalPapers: 3, streaming: true });
    expect(screen.getByText("Searching paper 2 of 3…")).toBeInTheDocument();
  });

  it("shows 'Compared N papers' once the turn is done", () => {
    renderPanel({
      rows: [row("p1", "Paper One", "Answer one."), row("p2", "Paper Two", "Answer two.")],
      totalPapers: 2,
      streaming: false,
    });
    expect(screen.getByText("Compared 2 papers")).toBeInTheDocument();
  });

  it("uses singular wording for a single-paper result", () => {
    renderPanel({
      rows: [row("p1", "Paper One", "Answer one.")],
      totalPapers: 1,
      streaming: false,
    });
    expect(screen.getByText("Compared 1 paper")).toBeInTheDocument();
  });

  it("does not get stuck on 'Searching...' when the turn ends with zero completed rows", () => {
    // Reachable via Stop clicked before the first paper finishes (the turn still
    // persists and reloads with rows: [] — see ComparePanel.tsx's `done` comment) or a
    // backend error before any row completes. `done` must track "stopped streaming,"
    // not "did anything complete," or the badge/spinner never resolves.
    renderPanel({ rows: [], totalPapers: 2, streaming: false });
    expect(screen.getByText("Compare stopped")).toBeInTheDocument();
    expect(screen.queryByText(/Searching paper/)).not.toBeInTheDocument();
  });
});

describe("ComparePanel carousel", () => {
  const rows = [
    row("p1", "Paper One", "Answer one [r-p1]."),
    row("p2", "Paper Two", "Answer two [r-p2]."),
    row("p3", "Paper Three", "Answer three [r-p3]."),
  ];

  it("shows the first paper's title and answer by default", () => {
    renderPanel({ rows, totalPapers: 3, streaming: false });
    expect(screen.getByText("Paper One")).toBeInTheDocument();
    expect(screen.getByText(/Answer one/)).toBeInTheDocument();
  });

  it("Previous is disabled on the first slide, Next is not", () => {
    renderPanel({ rows, totalPapers: 3, streaming: false });
    expect(screen.getByLabelText("Previous paper")).toBeDisabled();
    expect(screen.getByLabelText("Next paper")).not.toBeDisabled();
  });

  it("Next advances to the next paper; Previous goes back", () => {
    renderPanel({ rows, totalPapers: 3, streaming: false });

    fireEvent.click(screen.getByLabelText("Next paper"));
    expect(screen.getByText("Paper Two")).toBeInTheDocument();
    expect(screen.queryByText("Paper One")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Previous paper"));
    expect(screen.getByText("Paper One")).toBeInTheDocument();
  });

  it("Next is disabled on the last slide, no wraparound", () => {
    renderPanel({ rows, totalPapers: 3, streaming: false });
    fireEvent.click(screen.getByLabelText("Next paper"));
    fireEvent.click(screen.getByLabelText("Next paper"));
    expect(screen.getByText("Paper Three")).toBeInTheDocument();
    expect(screen.getByLabelText("Next paper")).toBeDisabled();

    fireEvent.click(screen.getByLabelText("Next paper")); // no-op past the last slide
    expect(screen.getByText("Paper Three")).toBeInTheDocument();
  });

  it("renders each slide's own citation via Answer, not another slide's", () => {
    renderPanel({ rows, totalPapers: 3, streaming: false });
    // Paper One's answer cites r-p1 -> renders a "1" citation badge (refNumber of r-p1).
    expect(screen.getByText(/Answer one/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Next paper"));
    expect(screen.getByText(/Answer two/)).toBeInTheDocument();
    expect(screen.queryByText(/Answer one/)).not.toBeInTheDocument();
  });

  it("shows a waiting message before the first row has arrived", () => {
    renderPanel({ rows: [], totalPapers: 2, streaming: true });
    expect(screen.getByText("Waiting for the first paper…")).toBeInTheDocument();
  });
});
