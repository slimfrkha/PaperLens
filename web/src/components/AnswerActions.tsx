import { ActionIcon, CopyButton, Group, Tooltip } from "@mantine/core";
import type { Citation } from "../api";
import { answerToBibtex, answerToMarkdown, citedCitations } from "../exportAnswer";
import { IconBook, IconCheck, IconCopy } from "./Icons";

/** "Copy as Markdown" (citations as [^N] footnotes + a References block) and "Copy
 *  BibTeX" (one @misc entry per cited paper) for one assistant answer. BibTeX is
 *  disabled when nothing was cited — a small-talk turn has no sources to export. */
export default function AnswerActions({
  text,
  citations,
}: {
  text: string;
  citations: Citation[];
}) {
  const cited = citedCitations(text, citations);
  const noCitations = cited.length === 0;

  return (
    <Group gap={4} mt="sm">
      <CopyButton value={answerToMarkdown(text, cited)}>
        {({ copied, copy }) => (
          <Tooltip label={copied ? "Copied!" : "Copy as Markdown"}>
            <ActionIcon
              size="sm"
              variant="subtle"
              color="gray"
              aria-label="Copy as Markdown"
              onClick={copy}
            >
              {copied ? <IconCheck size={15} /> : <IconCopy size={15} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
      <CopyButton value={answerToBibtex(cited)}>
        {({ copied, copy }) => (
          <Tooltip label={noCitations ? "No cited sources" : copied ? "Copied!" : "Copy BibTeX"}>
            {/* A disabled ActionIcon drops pointer events, so wrap in a span for the
                Tooltip to still trigger on hover (Mantine's documented workaround). */}
            <span>
              <ActionIcon
                size="sm"
                variant="subtle"
                color="gray"
                aria-label="Copy BibTeX"
                disabled={noCitations}
                onClick={copy}
              >
                {copied ? <IconCheck size={15} /> : <IconBook size={15} />}
              </ActionIcon>
            </span>
          </Tooltip>
        )}
      </CopyButton>
    </Group>
  );
}
