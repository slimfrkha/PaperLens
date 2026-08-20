import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AnnotationPopover, { type AnnotationPopoverProps } from "./AnnotationPopover";

function renderPopover(overrides: Partial<AnnotationPopoverProps> = {}) {
  const props: AnnotationPopoverProps = {
    position: { x: 10, y: 20 },
    mode: "create",
    onHighlight: vi.fn(),
    onSaveNote: vi.fn(),
    onDelete: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };
  render(
    <MantineProvider>
      <AnnotationPopover {...props} />
    </MantineProvider>,
  );
  return props;
}

describe("AnnotationPopover", () => {
  it("renders nothing when closed", () => {
    renderPopover({ position: null });
    expect(screen.queryByText("Highlight")).not.toBeInTheDocument();
  });

  it("create mode shows Highlight and Add note actions", () => {
    const props = renderPopover({ mode: "create" });
    fireEvent.click(screen.getByText("Highlight"));
    expect(props.onHighlight).toHaveBeenCalled();
  });

  it("create mode: Add note expands a textarea and Save posts the typed note", () => {
    const props = renderPopover({ mode: "create" });
    fireEvent.click(screen.getByText("Add note"));

    const textarea = screen.getByPlaceholderText("Add a note...");
    fireEvent.change(textarea, { target: { value: "a note to remember" } });
    fireEvent.click(screen.getByText("Save"));

    expect(props.onSaveNote).toHaveBeenCalledWith("a note to remember");
  });

  it("view mode shows the existing note with Edit/Delete actions", () => {
    const props = renderPopover({ mode: "view", note: "an existing note" });
    expect(screen.getByText("an existing note")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Delete"));
    expect(props.onDelete).toHaveBeenCalled();
  });

  it("view mode shows a placeholder when there is no note", () => {
    renderPopover({ mode: "view", note: "" });
    expect(screen.getByText("No note")).toBeInTheDocument();
  });

  it("clicking outside the popover calls onClose", () => {
    const props = renderPopover({ mode: "create" });
    fireEvent.mouseDown(document.body);
    expect(props.onClose).toHaveBeenCalled();
  });

  it("clicking outside while editing autosaves the draft instead of discarding it", () => {
    const props = renderPopover({ mode: "create" });
    fireEvent.click(screen.getByText("Add note"));

    const textarea = screen.getByPlaceholderText("Add a note...");
    fireEvent.change(textarea, { target: { value: "an unsaved draft" } });
    fireEvent.mouseDown(document.body);

    expect(props.onSaveNote).toHaveBeenCalledWith("an unsaved draft");
    expect(props.onClose).not.toHaveBeenCalled();
  });

  it("view mode: Edit reuses the Save flow to update the note", () => {
    const props = renderPopover({ mode: "view", note: "old note" });
    fireEvent.click(screen.getByText("Edit"));

    const textarea = screen.getByPlaceholderText("Add a note...");
    expect(textarea).toHaveValue("old note");
    fireEvent.change(textarea, { target: { value: "updated note" } });
    fireEvent.click(screen.getByText("Save"));

    expect(props.onSaveNote).toHaveBeenCalledWith("updated note");
  });
});
