import { Badge, Tooltip } from "@mantine/core";
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
        return (
          <Tooltip label={`${c.title} — ${c.section_title}`} multiline w={280} withArrow>
            <Badge
              component="a"
              variant="light"
              radius="sm"
              size="sm"
              style={{ cursor: "pointer", marginInline: 2 }}
              onClick={() =>
                navigate(`/papers/${c.paper_id}`, {
                  state: { highlight: c.snippet, section: c.section_title },
                })
              }
            >
              {c.ref}
            </Badge>
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
