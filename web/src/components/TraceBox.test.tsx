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
