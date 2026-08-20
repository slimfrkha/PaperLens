import { useState } from "react";
import { Button, Group, Popover, Stack, Text, Textarea } from "@mantine/core";

export interface AnnotationPopoverProps {
  /** Viewport coordinates to anchor the popover at; null closes it. */
  position: { x: number; y: number } | null;
  /** "create": a fresh selection with Highlight/Add-note actions.
   *  "view": an existing annotation, showing its note with Edit/Delete. */
  mode: "create" | "view";
  note?: string;
  onHighlight: () => void;
  onSaveNote: (note: string) => void;
  onDelete: () => void;
  onClose: () => void;
}

/** Selection toolbar / annotation editor, anchored to a point rather than a DOM element
 *  (a text selection can span multiple nodes, so there's no single element to target).
 *  The caller must pass a `key` that changes each time a *different* selection/annotation
 *  opens the popover, so `editing`/`draft` reset via remount rather than an effect. */
export default function AnnotationPopover({
  position,
  mode,
  note,
  onHighlight,
  onSaveNote,
  onDelete,
  onClose,
}: AnnotationPopoverProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(note ?? "");

  if (!position) return null;

  // Outside click (Mantine's own close-on-click-outside) dismisses the popover — while
  // editing, that must autosave the in-progress draft like the Save button, not discard it
  // like Cancel; the user didn't choose to cancel, they clicked somewhere else on the page.
  function handleDismiss() {
    if (editing) {
      onSaveNote(draft);
    } else {
      onClose();
    }
  }

  return (
    <Popover defaultOpened position="top" withArrow shadow="md" trapFocus onClose={handleDismiss}>
      <Popover.Target>
        <div
          style={{
            position: "fixed",
            left: position.x,
            top: position.y,
            width: 1,
            height: 1,
            pointerEvents: "none",
          }}
        />
      </Popover.Target>
      <Popover.Dropdown>
        <Stack gap={6} miw={220}>
          {mode === "create" && !editing && (
            <Group gap={6} wrap="nowrap">
              <Button size="xs" variant="light" onClick={onHighlight}>
                Highlight
              </Button>
              <Button size="xs" variant="light" onClick={() => setEditing(true)}>
                Add note
              </Button>
            </Group>
          )}
          {mode === "view" && !editing && (
            <>
              <Text size="sm" c={note ? undefined : "dimmed"}>
                {note || "No note"}
              </Text>
              <Group gap={6}>
                <Button size="xs" variant="subtle" onClick={() => setEditing(true)}>
                  Edit
                </Button>
                <Button size="xs" variant="subtle" color="red" onClick={onDelete}>
                  Delete
                </Button>
              </Group>
            </>
          )}
          {editing && (
            <Stack gap={6}>
              <Textarea
                autosize
                minRows={2}
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.currentTarget.value)}
                placeholder="Add a note..."
              />
              <Group gap={6} justify="flex-end">
                <Button
                  size="xs"
                  variant="subtle"
                  onClick={() => (mode === "create" ? onClose() : setEditing(false))}
                >
                  Cancel
                </Button>
                <Button size="xs" onClick={() => onSaveNote(draft)}>
                  Save
                </Button>
              </Group>
            </Stack>
          )}
        </Stack>
      </Popover.Dropdown>
    </Popover>
  );
}
