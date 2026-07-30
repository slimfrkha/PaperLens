import { useState } from "react";
import { ActionIcon, Collapse, Group, Text, Textarea, UnstyledButton } from "@mantine/core";
import { IconThumbDown, IconThumbUp } from "./Icons";
import type { Feedback } from "../api";

/** 👍/👎 + optional note on one assistant turn. Clicking the active vote again clears
 *  it; clicking the other one switches. The note field only appears once a vote is
 *  set, and commits on blur — there's no toast in this app, so the buttons' own
 *  filled/colored state is the only confirmation. */
export default function FeedbackControl({
  vote,
  note,
  onChange,
}: {
  vote: Feedback["vote"];
  note: string | null;
  onChange: (vote: Feedback["vote"], note: string | null) => void;
}) {
  const [draft, setDraft] = useState(note ?? "");
  const [noteOpen, setNoteOpen] = useState(false);

  function toggle(next: "up" | "down") {
    // Commit whatever's in the draft (even if the note box is still open, unblurred)
    // instead of the stale `note` prop — otherwise switching votes mid-edit silently
    // discards what the user just typed.
    const trimmed = draft.trim();
    onChange(vote === next ? null : next, vote === next ? null : trimmed || null);
  }

  function commitNote() {
    const trimmed = draft.trim();
    onChange(vote, trimmed || null);
  }

  return (
    <Group gap={4} mt="sm" wrap="wrap">
      <ActionIcon
        size="sm"
        variant="subtle"
        color={vote === "up" ? "accent" : "gray"}
        aria-label="Good answer"
        onClick={() => toggle("up")}
      >
        <IconThumbUp
          size={15}
          filled={vote === "up" ? "var(--mantine-color-accent-filled)" : undefined}
        />
      </ActionIcon>
      <ActionIcon
        size="sm"
        variant="subtle"
        color={vote === "down" ? "red" : "gray"}
        aria-label="Bad answer"
        onClick={() => toggle("down")}
      >
        <IconThumbDown
          size={15}
          filled={vote === "down" ? "var(--mantine-color-red-filled)" : undefined}
        />
      </ActionIcon>
      {vote && !noteOpen && (
        <UnstyledButton onClick={() => setNoteOpen(true)}>
          <Text size="xs" c="dimmed">
            {note ? "Edit note" : "+ note"}
          </Text>
        </UnstyledButton>
      )}
      {vote && (
        <Collapse in={noteOpen} style={{ width: "100%" }}>
          <Textarea
            autosize
            minRows={1}
            maxRows={4}
            size="xs"
            placeholder="What went wrong or right? (optional)"
            value={draft}
            onChange={(e) => setDraft(e.currentTarget.value)}
            onBlur={() => {
              commitNote();
              setNoteOpen(false);
            }}
          />
        </Collapse>
      )}
    </Group>
  );
}
