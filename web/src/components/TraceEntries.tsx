import { Badge, Box, Group, Text } from "@mantine/core";
import { IconSearch } from "./Icons";
import type { TraceEntry } from "../api";

/** Renders a list of Thought→Action→Observation entries as a stepped timeline on a
 *  rail — shared by TraceBox (a normal turn's own trace) and each ComparePanel carousel
 *  slide (that paper's own trace), since both need to render the identical entry shapes.
 *  The caller owns the rail's border/indent (TraceBox and ComparePanel each wrap this in
 *  their own bordered Box) — this component only lays out the entries themselves. */
export default function TraceEntries({ entries }: { entries: TraceEntry[] }) {
  return (
    <>
      {entries.map((e, i) => (
        <TraceLine key={i} e={e} />
      ))}
    </>
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
