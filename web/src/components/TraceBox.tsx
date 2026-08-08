import { useState } from "react";
import { Badge, Box, Collapse, Group, Loader, Text, UnstyledButton } from "@mantine/core";
import { IconChevron, IconSearch, IconSpark } from "./Icons";
import type { TraceEntry } from "../api";

/** The agent's work between question and answer — reasoning (thought), searches
 *  (action), and results (observation) — as a stepped timeline on a rail.
 *  Auto-expands while the agent is streaming; collapsible once it settles. */
export default function TraceBox({
  entries,
  streaming,
}: {
  entries: TraceEntry[];
  streaming?: boolean;
}) {
  const [open, setOpen] = useState<boolean>(!!streaming);
  // Adjust state during render (not in an effect) when `streaming` flips on —
  // see https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes.
  const [prevStreaming, setPrevStreaming] = useState(streaming);
  if (streaming !== prevStreaming) {
    setPrevStreaming(streaming);
    if (streaming) setOpen(true);
  }

  if (entries.length === 0) return null;
  const nActions = entries.filter((e) => e.type === "action").length;

  return (
    <Box
      mb="sm"
      style={{
        border: "1px solid var(--pl-border)",
        borderRadius: "var(--mantine-radius-md)",
        background: "var(--pl-surface)",
        overflow: "hidden",
      }}
    >
      <UnstyledButton
        onClick={() => setOpen((o) => !o)}
        style={{ width: "100%", padding: "8px 12px" }}
      >
        <Group gap={8} justify="space-between" wrap="nowrap">
          <Group gap={8} wrap="nowrap">
            <IconSpark size={14} />
            <Text size="xs" fw={600} c="dimmed" tt="uppercase" style={{ letterSpacing: "0.04em" }}>
              Reasoning
            </Text>
            {nActions > 0 && (
              <Badge size="xs" variant="light" radius="sm">
                {nActions} search{nActions > 1 ? "es" : ""}
              </Badge>
            )}
            {streaming && <Loader size={12} color="accent" />}
          </Group>
          <Box
            style={{
              transform: open ? "rotate(90deg)" : "none",
              transition: "transform 150ms ease",
              color: "var(--mantine-color-dimmed)",
            }}
          >
            <IconChevron size={15} />
          </Box>
        </Group>
      </UnstyledButton>
      <Collapse in={open}>
        <Box
          px="md"
          pb="sm"
          pt={4}
          ml="md"
          style={{ borderLeft: "1px solid var(--pl-border-strong)" }}
        >
          {entries.map((e, i) => (
            <TraceLine key={i} e={e} />
          ))}
        </Box>
      </Collapse>
    </Box>
  );
}

/** A dot on the rail, aligned to the row's first line. */
function Dot({ color }: { color: string }) {
  return (
    <Box
      style={{
        position: "absolute",
        left: -21,
        top: 7,
        width: 7,
        height: 7,
        borderRadius: 999,
        background: `var(--mantine-color-${color})`,
        boxShadow: "0 0 0 3px var(--pl-surface)",
      }}
    />
  );
}

function TraceLine({ e }: { e: TraceEntry }) {
  if (e.type === "thought")
    return (
      <Box pos="relative" pl="md" mt={8}>
        <Dot color="dimmed" />
        <Text
          size="sm"
          c="dimmed"
          ff="'Newsreader', Georgia, serif"
          fs="italic"
          style={{ whiteSpace: "pre-wrap", lineHeight: 1.55 }}
        >
          {e.text}
        </Text>
      </Box>
    );

  if (e.type === "action")
    return (
      <Box pos="relative" pl="md" mt={8}>
        <Dot color="accent-filled" />
        <Group gap={6} wrap="nowrap" align="center">
          <IconSearch size={13} />
          <Text size="sm" fw={500} style={{ color: "var(--mantine-color-accent-light-color)" }}>
            {e.query}
          </Text>
          {e.paper && (
            <Badge size="xs" variant="outline" color="gray" radius="sm">
              {e.paper}
            </Badge>
          )}
          {e.per_paper && (
            <Badge size="xs" variant="light" color="accent" radius="sm">
              per-paper
            </Badge>
          )}
        </Group>
      </Box>
    );

  // observation
  return (
    <Box pos="relative" pl="md" mt={4}>
      <Text
        size="xs"
        c="dimmed"
        style={{
          whiteSpace: "pre-wrap",
          fontFamily: "var(--mantine-font-family-monospace)",
          lineHeight: 1.5,
        }}
      >
        {e.text}
      </Text>
    </Box>
  );
}
