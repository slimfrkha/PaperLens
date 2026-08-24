import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TraceEntry } from "../api";
import TraceBox from "./TraceBox";

function renderTrace(entries: TraceEntry[]) {
  return render(
    <MantineProvider>
      <TraceBox entries={entries} />
    </MantineProvider>,
  );
}

describe("TraceBox per-paper badge", () => {
  it("renders a per-paper badge when an action entry has per_paper: true", () => {
    renderTrace([{ type: "action", query: "q", per_paper: true }]);
    expect(screen.getByText("per-paper")).toBeInTheDocument();
  });

  it("does not render the badge when per_paper is false", () => {
    renderTrace([{ type: "action", query: "q", per_paper: false }]);
    expect(screen.queryByText("per-paper")).not.toBeInTheDocument();
  });

  it("does not render the badge when per_paper is omitted (old persisted chats)", () => {
    renderTrace([{ type: "action", query: "q" }]);
    expect(screen.queryByText("per-paper")).not.toBeInTheDocument();
  });
});

// Rendering coverage for the entry list itself (moved into TraceEntries, extracted for
// reuse by ComparePanel) — asserted through TraceBox, its one existing caller, so this
// also doubles as a regression check that the extraction didn't change what renders.
describe("TraceBox entry rendering", () => {
  it("renders nothing for an empty entry list", () => {
    renderTrace([]);
    expect(screen.queryByText("Reasoning")).not.toBeInTheDocument();
  });

  it("renders a thought entry's text", () => {
    renderTrace([{ type: "thought", text: "I should search for this." }]);
    expect(screen.getByText("I should search for this.")).toBeInTheDocument();
  });

  it("renders an action entry's query and optional paper badge", () => {
    renderTrace([{ type: "action", query: "latent attention", paper: "paper-a" }]);
    expect(screen.getByText("latent attention")).toBeInTheDocument();
    expect(screen.getByText("paper-a")).toBeInTheDocument();
  });

  it("renders an action entry with no paper filter without a paper badge", () => {
    renderTrace([{ type: "action", query: "latent attention" }]);
    expect(screen.queryByText("paper-a")).not.toBeInTheDocument();
  });

  it("renders an observation entry's text", () => {
    renderTrace([{ type: "observation", text: "[r1] paper=paper-a\nsome passage" }]);
    expect(screen.getByText(/some passage/)).toBeInTheDocument();
  });

  it("renders the N searches badge and shows the reasoning header", () => {
    renderTrace([
      { type: "thought", text: "Searching." },
      { type: "action", query: "q1" },
      { type: "observation", text: "result 1" },
      { type: "action", query: "q2" },
      { type: "observation", text: "result 2" },
    ]);
    expect(screen.getByText("Reasoning")).toBeInTheDocument();
    expect(screen.getByText("2 searches")).toBeInTheDocument();
  });
});
