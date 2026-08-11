import { Badge, Group, Stack, Text, Tooltip, UnstyledButton } from "@mantine/core";
import { Fragment, type MouseEvent } from "react";
import { useNavigate } from "react-router-dom";
import type { Citation, FaithfulnessLabel, RetrievalSource } from "../api";
import {
  faithfulnessColor,
  faithfulnessMessage,
  summarizeFaithfulness,
  worstLabel,
} from "../faithfulness";

/** Plain-language tooltip for a citation found via the sparse/hybrid lane. `"dense"` (the
 *  common case, and the only value when hybrid retrieval is off) is never rendered — see
 *  the `!== "dense"` guard at the call site. */
function sourceMessage(source: RetrievalSource): string {
  return source === "both" ? "found via keyword + semantic match" : "found via keyword match";
}

interface SourceNum {
  num: string; // citation number, matching the inline [n] markers
  label: FaithfulnessLabel | undefined; // worst-of-per-ref verdict; undefined = unchecked
  source: RetrievalSource | undefined; // which retrieval pool(s) surfaced it; undefined = dense/unknown
  section_title: string; // this ref's own section — several refs can share a paper but not a section
  snippet: string; // this ref's own passage, for the highlight-on-open behavior
}

interface Source {
  paper_id: string;
  title: string;
  nums: SourceNum[];
}

/** Groups an answer's citations by paper and renders one compact card per source —
 *  the papers this answer stood on. Each number opens the paper at its own cited
 *  passage (same target as that number's inline [n] marker in Answer); the rest of
 *  the card opens the paper plain, with no passage highlighted — several numbers on
 *  one card can point at different passages, so the card itself picks none of them. */
export default function SourceCards({ citations }: { citations: Citation[] }) {
  const navigate = useNavigate();
  if (citations.length === 0) return null;

  // Group by paper, preserving first-seen order; several [n] can hit one paper.
  const byPaper = new Map<string, Source>();
  for (const c of citations) {
    const n = {
      num: c.ref.replace(/^r/, ""),
      label: worstLabel(c.faithfulness),
      source: c.source,
      section_title: c.section_title,
      snippet: c.snippet,
    };
    const s = byPaper.get(c.paper_id);
    if (s) s.nums.push(n);
    else byPaper.set(c.paper_id, { paper_id: c.paper_id, title: c.title, nums: [n] });
  }
  const sources = [...byPaper.values()];
  const summary = summarizeFaithfulness(citations);
  const flagged = summary ? summary.total - summary.counts.entailment : 0;

  return (
    <Stack gap={8} mt="lg">
      <Group gap={8}>
        <Text size="xs" c="dimmed" fw={600} tt="uppercase" style={{ letterSpacing: "0.04em" }}>
          Sources
        </Text>
        {summary && flagged > 0 && (
          <Tooltip
            multiline
            w={240}
            label="Some citations don't clearly support (or may contradict) the claim they're attached to — an automated check, not a guarantee. Hover a flagged number below for detail."
          >
            <Badge size="xs" variant="light" radius="sm" color={faithfulnessColor(summary.worst)}>
              {flagged}/{summary.total}{" "}
              {summary.worst === "contradiction"
                ? "may contradict source"
                : "not clearly supported"}
            </Badge>
          </Tooltip>
        )}
      </Group>
      <Group gap="sm" align="stretch">
        {sources.map((s) => (
          <UnstyledButton
            key={s.paper_id}
            className="paper-card"
            onClick={() => navigate(`/papers/${s.paper_id}`)}
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
              {s.nums.map((n) => {
                const flag = n.label && n.label !== "entailment" ? n.label : undefined;
                const lexical = n.source && n.source !== "dense" ? n.source : undefined;
                const num = (
                  <Text
                    span
                    component="a"
                    className={flag ? `cite cite-${flag}` : "cite"}
                    aria-label={
                      flag
                        ? `citation ${n.num}: this source ${faithfulnessMessage(flag)}`
                        : undefined
                    }
                    onClick={(e: MouseEvent) => {
                      // Own passage, not the card's — several numbers on one card can
                      // point at different sections of the same paper.
                      e.stopPropagation();
                      navigate(`/papers/${s.paper_id}`, {
                        state: { highlight: n.snippet, section: n.section_title },
                      });
                    }}
                  >
                    {n.num}
                    {flag && (
                      <Text component="span" className="cite-flag" inherit aria-hidden>
                        !
                      </Text>
                    )}
                  </Text>
                );
                // No visual marker for a lexical/hybrid hit — the number's color already carries
                // the higher-stakes faithfulness signal, and stacking a second glyph there reads
                // as noise (e.g. "4!K"). The keyword-match info still exists, just quieter: a
                // hover tooltip on the number itself, same affordance the number already has for
                // the "click to open the paper" action.
                if (!lexical) return <Fragment key={n.num}>{num}</Fragment>;
                return (
                  <Tooltip key={n.num} label={sourceMessage(lexical)}>
                    {num}
                  </Tooltip>
                );
              })}
            </Group>
            <Text size="sm" fw={500} lh={1.25} lineClamp={2} ff="'Newsreader', Georgia, serif">
              {s.title}
            </Text>
          </UnstyledButton>
        ))}
      </Group>
    </Stack>
  );
}
