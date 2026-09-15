import { Box, Text, Tooltip } from "@mantine/core";
import { useNavigate } from "react-router-dom";
import type { Components } from "react-markdown";
import Markdown from "./Markdown";
import type { Citation, FaithfulnessClaim } from "../api";
import { isSafeExternalUrl, isWebCitation } from "../api";
import { REF_ID, REF_MARKER, extractCitedRefs, refNumber } from "../exportAnswer";
import { normalizeLatexMath } from "../markdownMath";
import {
  createClaimResolver,
  faithfulnessColor,
  faithfulnessMessage,
  sentenceAt,
  splitSentencesWithOffsets,
} from "../faithfulness";

/** A cited passage to open. Structurally shared with SourceCards and ChatPage's
 *  side-by-side handler (no named import needed — the shapes match). */
export interface OpenCitationTarget {
  paperId: string;
  ref?: string;
  snippet?: string;
  section?: string;
}

/** Renders an assistant answer as markdown, turning [rN] markers into clickable
 *  citation badges that open the cited paper with the passage highlighted. When
 *  `onOpenCitation` is given (the chat side-by-side view), a click opens the paper in the
 *  panel; otherwise it navigates to the standalone paper route. */
export default function Answer({
  text,
  citations,
  onOpenCitation,
}: {
  text: string;
  citations: Citation[];
  onOpenCitation?: (target: OpenCitationTarget) => void;
}) {
  const navigate = useNavigate();
  const byRef = new Map(citations.map((c) => [c.ref, c]));
  const citedRefs = new Set(extractCitedRefs(text, byRef));
  const sentences = splitSentencesWithOffsets(text);
  const resolveClaim = createClaimResolver(citations);
  const instanceClaims: (FaithfulnessClaim | undefined)[] = [];

  // Rewrite [rN] into a markdown link (cite:rN:instanceIdx) so it survives
  // markdown parsing; instanceIdx picks out this specific marker's own
  // faithfulness claim (a ref cited in several sentences can carry different
  // verdicts per sentence — not one collapsed color for the whole ref). A bunched
  // bracket ([r10, r12] — REF_MARKER's group 1 holds the whole bunch) becomes one
  // link per ref, same visual result as if the model had written [r10][r12]. If
  // any ref in the bunch doesn't resolve (e.g. a hallucinated one alongside a real
  // one), the whole bracket falls back to plain text unchanged — same as a lone
  // unresolved ref already does — rather than silently dropping just that ref's
  // text while linking the rest.
  const processed = text.replace(REF_MARKER, (m, group: string, offset: number) => {
    const refsInGroup = [...group.matchAll(REF_ID)].map((refMatch) => refMatch[0]);
    if (!refsInGroup.every((ref) => citedRefs.has(ref))) return m;
    return refsInGroup
      .map((ref) => {
        const claim = resolveClaim(ref, sentenceAt(sentences, offset));
        const idx = instanceClaims.push(claim) - 1;
        return `[${ref}](cite:${ref}:${idx})`;
      })
      .join("");
  });

  const components: Components = {
    a({ href, children }) {
      if (href && href.startsWith("cite:")) {
        const [ref, idxStr] = href.slice(5).split(":");
        const c = byRef.get(ref);
        if (!c) return <>{children}</>;
        const n = refNumber(c);
        // A web citation opens its source URL in a new tab — never the paper panel — and
        // carries no faithfulness/section metadata, so it gets its own simpler badge.
        if (isWebCitation(c)) {
          // The backend rejects non-HTTP(S) results. Keep this second check for old or
          // manually-edited saved chats, which still flow into this renderer.
          if (!isSafeExternalUrl(c.url)) {
            return (
              <Text
                component="span"
                className="cite cite-web"
                aria-label={`web citation ${n}: invalid source URL omitted`}
              >
                {n}
              </Text>
            );
          }
          return (
            <Tooltip
              color="dark.8"
              label={
                <Box style={{ maxWidth: 300 }}>
                  <Text size="xs" fw={600} lh={1.3} c="white">
                    {c.title}
                  </Text>
                  <Text size="xs" c="gray.4" mt={2}>
                    Web source · {c.url}
                  </Text>
                  {c.snippet && (
                    <Text size="xs" c="gray.3" mt={6} lineClamp={3} fs="italic">
                      “{c.snippet}”
                    </Text>
                  )}
                  <Text size="10px" c="gray.5" mt={6}>
                    Click to open the source
                  </Text>
                </Box>
              }
              multiline
              withArrow
              radius="md"
            >
              <Text
                component="a"
                href={c.url}
                target="_blank"
                rel="noreferrer"
                className="cite cite-web"
                aria-label={`web citation ${n}: opens ${c.url}`}
              >
                {n}
              </Text>
            </Tooltip>
          );
        }
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
                onOpenCitation
                  ? onOpenCitation({
                      paperId: c.paper_id,
                      ref: c.ref,
                      snippet: c.snippet,
                      section: c.section_title,
                    })
                  : navigate(`/papers/${c.paper_id}`, {
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
      {normalizeLatexMath(processed)}
    </Markdown>
  );
}
