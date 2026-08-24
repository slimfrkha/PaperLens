import { useState } from "react";
import {
  ActionIcon,
  Anchor,
  Badge,
  Box,
  Collapse,
  Group,
  Loader,
  Stack,
  Text,
  UnstyledButton,
} from "@mantine/core";
import { IconChevron, IconExternal, IconSpark } from "./Icons";
import type { CompareRow } from "../api";
import Answer from "./Answer";
import TraceEntries from "./TraceEntries";

/** The Compare-mode counterpart to TraceBox: a collapsible "Compared N papers" section
 *  holding a carousel, one slide per paper — that paper's own title, its own
 *  Thought→Action→Observation trace, then its own individual answer (rendered via the
 *  same, unmodified `Answer`, so its citation links and faithfulness flags work exactly
 *  like a normal answer's).
 *
 *  Auto-expands the moment the per-paper sub-runs start streaming, and auto-collapses
 *  the moment the synthesis pass starts — this collapse-on-finish half is new behavior,
 *  not a mirror of TraceBox (which never auto-collapses once opened). */
export default function ComparePanel({
  rows,
  totalPapers,
  streaming,
  synthesizing,
}: {
  rows: CompareRow[];
  totalPapers: number;
  streaming?: boolean;
  synthesizing?: boolean;
}) {
  const inProgress = !!streaming && !synthesizing;
  const [open, setOpen] = useState<boolean>(inProgress);
  // Adjust state during render (not in an effect) on a streaming/synthesizing edge —
  // same pattern TraceBox already uses for its own (expand-only) case.
  const [prevInProgress, setPrevInProgress] = useState(inProgress);
  if (inProgress !== prevInProgress) {
    setPrevInProgress(inProgress);
    if (inProgress) setOpen(true);
    else if (synthesizing) setOpen(false);
  }

  const [idx, setIdx] = useState(0);
  // Track the latest completed paper while sub-runs are still in progress, so the
  // carousel shows what just finished rather than staying pinned to slide 0; once
  // synthesis starts (or the turn reloads from history), leave the slide wherever the
  // user last navigated it to.
  const [prevRowCount, setPrevRowCount] = useState(rows.length);
  if (rows.length !== prevRowCount) {
    setPrevRowCount(rows.length);
    if (inProgress) setIdx(rows.length - 1);
  }

  const clampedIdx = Math.min(Math.max(idx, 0), Math.max(rows.length - 1, 0));
  const row = rows[clampedIdx];
  // `done` tracks "the turn stopped streaming," not "did any paper complete" — a Compare
  // turn that ends (successfully or via Stop/error) before a single row finishes still
  // needs `done` true, or the badge/spinner is stuck showing "Searching..." forever, on a
  // turn that in the Stop case even persists and reloads in that same stuck-looking state.
  const done = !streaming;
  const badgeLabel = done
    ? rows.length > 0
      ? `Compared ${rows.length} paper${rows.length === 1 ? "" : "s"}`
      : "Compare stopped"
    : `Searching paper ${Math.min(rows.length + 1, totalPapers)} of ${totalPapers}…`;

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
              Compare
            </Text>
            <Badge size="xs" variant="light" radius="sm">
              {badgeLabel}
            </Badge>
            {!done && <Loader size={12} color="accent" />}
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
        <Box px="md" pb="md" pt={4}>
          {row ? (
            <Stack gap="xs">
              <Group gap={8} justify="space-between" wrap="nowrap">
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="sm"
                  disabled={clampedIdx === 0}
                  aria-label="Previous paper"
                  onClick={() => setIdx((i) => Math.max(i - 1, 0))}
                >
                  <IconChevron size={14} style={{ transform: "rotate(180deg)" }} />
                </ActionIcon>
                <Group gap={6} wrap="nowrap" style={{ flex: 1, justifyContent: "center" }}>
                  <Text size="sm" fw={600}>
                    {row.title}
                  </Text>
                  {row.arxiv_id && (
                    <Anchor
                      href={`https://arxiv.org/abs/${row.arxiv_id}`}
                      target="_blank"
                      rel="noreferrer"
                      size="xs"
                    >
                      <Group gap={2} wrap="nowrap">
                        arXiv:{row.arxiv_id}
                        <IconExternal size={11} />
                      </Group>
                    </Anchor>
                  )}
                  <Text size="xs" c="dimmed">
                    {clampedIdx + 1} / {rows.length}
                  </Text>
                </Group>
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="sm"
                  disabled={clampedIdx === rows.length - 1}
                  aria-label="Next paper"
                  onClick={() => setIdx((i) => Math.min(i + 1, rows.length - 1))}
                >
                  <IconChevron size={14} />
                </ActionIcon>
              </Group>
              {row.trace.length > 0 && (
                <Box pl="md" style={{ borderLeft: "1px solid var(--pl-border-strong)" }}>
                  <TraceEntries entries={row.trace} />
                </Box>
              )}
              <Answer text={row.text} citations={row.citations} />
            </Stack>
          ) : (
            <Text size="sm" c="dimmed">
              Waiting for the first paper…
            </Text>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}
