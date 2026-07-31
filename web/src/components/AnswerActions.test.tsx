import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Citation } from "../api";
import AnswerActions from "./AnswerActions";

function citation(ref: string, paper_id: string, arxiv_id?: string): Citation {
  return {
    ref,
    paper_id,
    title: "A Paper",
    breadcrumb: "",
    section_title: "Method",
    snippet: "sn",
    arxiv_id,
  };
}

function renderActions(text: string, citations: Citation[]) {
  return render(
    <MantineProvider>
      <AnswerActions text={text} citations={citations} />
    </MantineProvider>,
  );
}

describe("AnswerActions", () => {
  beforeEach(() => {
    vi.mocked(navigator.clipboard.writeText).mockClear();
  });

  it("copies the answer as markdown with a footnote and references block", () => {
    renderActions("MLA shrinks the cache [r1].", [citation("r1", "p1", "2412.19437")]);
    fireEvent.click(screen.getByLabelText("Copy as Markdown"));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("MLA shrinks the cache [^1]."),
    );
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("https://arxiv.org/abs/2412.19437"),
    );
  });

  it("disables Copy BibTeX when the turn cited no sources", () => {
    renderActions("Hi there!", []);
    expect(screen.getByLabelText("Copy BibTeX")).toBeDisabled();
  });

  it("enables Copy BibTeX and copies a @misc entry when a source was cited", () => {
    renderActions("MLA shrinks the cache [r1].", [citation("r1", "p1", "2412.19437")]);
    const btn = screen.getByLabelText("Copy BibTeX");
    expect(btn).not.toBeDisabled();
    fireEvent.click(btn);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("@misc{p1,"),
    );
  });
});
