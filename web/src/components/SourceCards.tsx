import { Group, Stack, Text, UnstyledButton } from "@mantine/core";
import { useNavigate } from "react-router-dom";
import type { Citation } from "../api";

interface Source {
  paper_id: string;
  title: string;
  section_title: string;
  snippet: string;
  nums: string[]; // citation numbers into this paper, matching the inline [n] markers
}

/** Groups an answer's citations by paper and renders one compact, clickable card
 *  per source — the papers this answer stood on. Clicking opens the paper at the
 *  first cited passage (same target as the inline [n] markers in Answer). */
export default function SourceCards({ citations }: { citations: Citation[] }) {
  const navigate = useNavigate();
  if (citations.length === 0) return null;

  // Group by paper, preserving first-seen order; several [n] can hit one paper.
  const byPaper = new Map<string, Source>();
  for (const c of citations) {
    const num = c.ref.replace(/^r/, "");
    const s = byPaper.get(c.paper_id);
    if (s) s.nums.push(num);
    else
      byPaper.set(c.paper_id, {
        paper_id: c.paper_id,
        title: c.title,
        section_title: c.section_title,
        snippet: c.snippet,
        nums: [num],
      });
  }
  const sources = [...byPaper.values()];

  return (
    <Stack gap={8} mt="lg">
      <Text size="xs" c="dimmed" fw={600} tt="uppercase" style={{ letterSpacing: "0.04em" }}>
        Sources
      </Text>
      <Group gap="sm" align="stretch">
        {sources.map((s) => (
          <UnstyledButton
            key={s.paper_id}
            className="paper-card"
            onClick={() =>
              navigate(`/papers/${s.paper_id}`, {
                state: { highlight: s.snippet, section: s.section_title },
              })
            }
            style={{
              flex: "1 1 200px",
              maxWidth: 280,
              padding: "10px 12px",
              border: "1px solid var(--pl-border)",
              borderRadius: 12,
              background: "var(--pl-surface)",
            }}
          >
            <Group gap={4} mb={6}>
              {s.nums.map((n) => (
                <Text key={n} span className="cite" style={{ cursor: "inherit" }}>
                  {n}
                </Text>
              ))}
            </Group>
            <Text size="sm" fw={500} lh={1.25} lineClamp={2} ff="'Newsreader', Georgia, serif">
              {s.title}
            </Text>
            <Text size="xs" c="dimmed" mt={4} lineClamp={1}>
              {s.section_title}
            </Text>
          </UnstyledButton>
        ))}
      </Group>
    </Stack>
  );
}
