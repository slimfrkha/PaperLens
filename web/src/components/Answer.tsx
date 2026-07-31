import { Box, Text, Tooltip } from "@mantine/core";
import { useNavigate } from "react-router-dom";
import type { Components } from "react-markdown";
import Markdown from "./Markdown";
import type { Citation, FaithfulnessClaim } from "../api";
import { REF_MARKER, extractCitedRefs, refNumber } from "../exportAnswer";
import {
  createClaimResolver,
  faithfulnessColor,
  faithfulnessMessage,
  sentenceAt,
  splitSentencesWithOffsets,
} from "../faithfulness";

/** Renders an assistant answer as markdown, turning [rN] markers into clickable
 *  citation badges that open the cited paper with the passage highlighted. */
export default function Answer({ text, citations }: { text: string; citations: Citation[] }) {
  const navigate = useNavigate();
  const byRef = new Map(citations.map((c) => [c.ref, c]));
  const citedRefs = new Set(extractCitedRefs(text, byRef));
  const sentences = splitSentencesWithOffsets(text);
  const resolveClaim = createClaimResolver(citations);
  const instanceClaims: (FaithfulnessClaim | undefined)[] = [];

  // Rewrite [rN] into a markdown link (cite:rN:instanceIdx) so it survives
  // markdown parsing; instanceIdx picks out this specific marker's own
  // faithfulness claim (a ref cited in several sentences can carry different
  // verdicts per sentence — not one collapsed color for the whole ref).
  const processed = text.replace(REF_MARKER, (m, ref, offset: number) => {
    if (!citedRefs.has(ref)) return m;
    const claim = resolveClaim(ref, sentenceAt(sentences, offset));
    const idx = instanceClaims.push(claim) - 1;
    return `[${ref}](cite:${ref}:${idx})`;
  });

  const components: Components = {
    a({ href, children }) {
      if (href && href.startsWith("cite:")) {
        const [ref, idxStr] = href.slice(5).split(":");
        const c = byRef.get(ref);
        if (!c) return <>{children}</>;
        const n = refNumber(c);
        const claim = instanceClaims[Number(idxStr)];
        // Stay silent on entailment — the thresholds behind it are a starting
        // calibration, not a validated guarantee, so only flag concerns.
        const flagged = claim && claim.label !== "entailment" ? claim : undefined;
        return (
          <Tooltip
            color="dark.8"
            label={
              <Box style={{ maxWidth: 300 }}>
                <Text size="xs" fw={600} lh={1.3} c="white">
                  {c.title}
                </Text>
                <Text size="xs" c="gray.4" mt={2}>
                  {c.section_title}
                </Text>
                {c.snippet && (
                  <Text size="xs" c="gray.3" mt={6} lineClamp={3} fs="italic">
                    “{c.snippet}”
                  </Text>
                )}
                {flagged && (
                  <Text size="xs" c={`${faithfulnessColor(flagged.label)}.4`} mt={6}>
                    ⚠ This source {faithfulnessMessage(flagged.label)} (
                    {(flagged.score * 100).toFixed(0)}% supported)
                  </Text>
                )}
                <Text size="10px" c="gray.5" mt={6}>
                  Click to open the passage
                </Text>
              </Box>
            }
            multiline
            withArrow
            radius="md"
          >
            <Text
              component="a"
              className={flagged ? `cite cite-${flagged.label}` : "cite"}
              aria-label={
                flagged
                  ? `citation ${n}: this source ${faithfulnessMessage(flagged.label)}`
                  : undefined
              }
              onClick={() =>
                navigate(`/papers/${c.paper_id}`, {
                  state: { highlight: c.snippet, section: c.section_title },
                })
              }
            >
              {n}
              {flagged && (
                <Text component="span" className="cite-flag" aria-hidden>
                  !
                </Text>
              )}
            </Text>
          </Tooltip>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };

  // Keep the internal `cite:` scheme; react-markdown's default sanitizer drops it.
  return (
    <Markdown components={components} urlTransform={(url) => url}>
      {processed}
    </Markdown>
  );
}
