import { useState } from "react";
import { Badge, Box, Collapse, Group, Loader, Text, UnstyledButton } from "@mantine/core";
import type { TraceEntry } from "../api";

/** Collapsible view of everything the agent did between the question and the
 *  final answer: reasoning (thought), tool calls (action), and results
 *  (observation). Collapsed by default — click to watch what's happening. */
export default function TraceBox({
  entries,
  streaming,
}: {
  entries: TraceEntry[];
  streaming?: boolean;
}) {
  const [open, setOpen] = useState<boolean>(false);
  if (entries.length === 0) return null;
  const nActions = entries.filter((e) => e.type === "action").length;

  return (
    <Box
      mb="sm"
      style={{ border: "1px solid var(--mantine-color-gray-3)", borderRadius: 6 }}
    >
      <UnstyledButton onClick={() => setOpen((o) => !o)} style={{ width: "100%", padding: "6px 10px" }}>
        <Group gap={8}>
          <Text size="xs" c="dimmed">
            {open ? "▾" : "▸"} Agent trace
          </Text>
          {nActions > 0 && (
            <Badge size="xs" variant="light" color="grape">
              {nActions} search{nActions > 1 ? "es" : ""}
            </Badge>
          )}
          {streaming && <Loader size={12} />}
        </Group>
      </UnstyledButton>
      <Collapse in={open}>
        <Box px="md" pb="sm">
          {entries.map((e, i) => (
            <TraceLine key={i} e={e} />
          ))}
        </Box>
      </Collapse>
    </Box>
  );
}

function TraceLine({ e }: { e: TraceEntry }) {
  if (e.type === "thought")
    return (
      <Text size="xs" c="dimmed" fs="italic" mt={6} style={{ whiteSpace: "pre-wrap" }}>
        🧠 {e.text}
      </Text>
    );
  if (e.type === "action")
    return (
      <Group gap={4} mt={6}>
        <Text size="xs">🔍</Text>
        <Badge size="xs" variant="dot" color="grape">
          {e.query}
          {e.paper ? ` · ${e.paper}` : ""}
        </Badge>
      </Group>
    );
  // observation
  return (
    <Box mt={2} pl="sm" style={{ borderLeft: "2px solid var(--mantine-color-gray-3)" }}>
      <Text
        size="xs"
        c="dimmed"
        style={{ whiteSpace: "pre-wrap", fontFamily: "var(--mantine-font-family-monospace)" }}
      >
        {e.text}
      </Text>
    </Box>
  );
}
