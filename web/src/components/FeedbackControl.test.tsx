import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import FeedbackControl from "./FeedbackControl";

function renderControl(
  vote: "up" | "down" | null,
  note: string | null,
  onChange: (vote: "up" | "down" | null, note: string | null) => void,
) {
  return render(
    <MantineProvider>
      <FeedbackControl vote={vote} note={note} onChange={onChange} />
    </MantineProvider>,
  );
}

describe("FeedbackControl", () => {
  it("clicking the active vote clears it", () => {
    const onChange = vi.fn();
    renderControl("up", null, onChange);
    fireEvent.click(screen.getByLabelText("Good answer"));
    expect(onChange).toHaveBeenCalledWith(null, null);
  });

  it("switching votes while the note box is open commits the unblurred draft", () => {
    // Regression: toggle() used to read the stale `note` prop instead of the live
    // draft, silently discarding whatever the user had typed but not yet blurred.
    const onChange = vi.fn();
    renderControl("up", null, onChange);

    fireEvent.click(screen.getByText("+ note"));
    fireEvent.change(screen.getByPlaceholderText(/what went wrong or right/i), {
      target: { value: "cited the wrong section" },
    });
    // Switch to thumbs-down without blurring the textarea first.
    fireEvent.click(screen.getByLabelText("Bad answer"));

    expect(onChange).toHaveBeenCalledWith("down", "cited the wrong section");
  });

  it("clearing a vote always clears the note, even with an open unblurred draft", () => {
    const onChange = vi.fn();
    renderControl("up", "existing note", onChange);

    fireEvent.click(screen.getByText("Edit note"));
    fireEvent.change(screen.getByPlaceholderText(/what went wrong or right/i), {
      target: { value: "something else" },
    });
    fireEvent.click(screen.getByLabelText("Good answer")); // toggle off

    expect(onChange).toHaveBeenCalledWith(null, null);
  });
});
