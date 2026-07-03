import { Box, Text, Tooltip } from "@mantine/core";
import { useNavigate } from "react-router-dom";
import type { Components } from "react-markdown";
import Markdown from "./Markdown";
import type { Citation } from "../api";

/** Renders an assistant answer as markdown, turning [rN] markers into clickable
 *  citation badges that open the cited paper with the passage highlighted. */
export default function Answer({ text, citations }: { text: string; citations: Citation[] }) {
  const navigate = useNavigate();
  const byRef = new Map(citations.map((c) => [c.ref, c]));

  // Rewrite [rN] into a markdown link (cite:rN) so it survives markdown parsing.
  const processed = text.replace(/\[(r\d+)\]/g, (m, ref) =>
    byRef.has(ref) ? `[${ref}](cite:${ref})` : m
  );

  const components: Components = {
    a({ href, children }) {
      if (href && href.startsWith("cite:")) {
        const c = byRef.get(href.slice(5));
        if (!c) return <>{children}</>;
        const n = c.ref.replace(/^r/, "");
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
              className="cite"
              onClick={() =>
                navigate(`/papers/${c.paper_id}`, {
                  state: { highlight: c.snippet, section: c.section_title },
                })
              }
            >
              {n}
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
