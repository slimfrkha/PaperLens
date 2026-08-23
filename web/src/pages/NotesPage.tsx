import { useEffect, useState } from "react";
import {
  ActionIcon,
  Alert,
  Anchor,
  Badge,
  Box,
  CopyButton,
  Group,
  MultiSelect,
  Stack,
  Text,
  TextInput,
  Title,
  Tooltip,
  UnstyledButton,
} from "@mantine/core";
import { Link, useNavigate } from "react-router-dom";
import { deleteAnnotation, getAllAnnotations, getPapers, type LibraryAnnotation } from "../api";
import { groupByPaper, notesToMarkdown } from "../exportNotes";
import { IconCheck, IconCopy, IconExternal, IconSearch, IconTrash } from "../components/Icons";

/** Hand-rolled — no relative-time dependency exists in this codebase (checked
 *  package.json and every other page) and one label doesn't earn adding one. */
function relativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

export default function NotesPage() {
  const navigate = useNavigate();
  const [notes, setNotes] = useState<LibraryAnnotation[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [query, setQuery] = useState("");
  const [paperFilter, setPaperFilter] = useState<string[]>([]);
  const [paperOptions, setPaperOptions] = useState<{ value: string; label: string }[]>([]);

  useEffect(() => {
    getAllAnnotations().then((n) => {
      setNotes(n);
      setLoaded(true);
    });
    getPapers().then((p) => setPaperOptions(p.map((x) => ({ value: x.paper_id, label: x.title }))));
  }, []);

  const q = query.trim().toLowerCase();
  const filtered = notes.filter(
    (n) =>
      (paperFilter.length === 0 || paperFilter.includes(n.paper_id)) &&
      (q === "" || n.snippet.toLowerCase().includes(q) || n.note.toLowerCase().includes(q)),
  );
  const groups = groupByPaper(filtered);

  const openAnnotation = (n: LibraryAnnotation) =>
    navigate(`/papers/${n.paper_id}`, {
      state: { highlight: n.snippet, section: n.section_title },
    });

  const handleDelete = async (n: LibraryAnnotation) => {
    if (!window.confirm("Delete this note?")) return;
    try {
      await deleteAnnotation(n.paper_id, n.id);
      setNotes((prev) => prev.filter((x) => x.id !== n.id));
    } catch (e) {
      alert(e instanceof Error ? e.message : "failed to delete note");
    }
  };

  return (
    <Stack gap="lg">
      <Group align="baseline" gap="sm">
        <Title order={2}>Notes</Title>
        <Text c="dimmed" className="tnum">
          {notes.length}
        </Text>
      </Group>

      {loaded && notes.length === 0 && (
        <Alert color="yellow" variant="light" radius="md">
          No notes yet — open a paper and select text to add one. See{" "}
          <Anchor component={Link} to="/papers">
            Papers
          </Anchor>
          .
        </Alert>
      )}

      {notes.length > 0 && (
        <>
          <Group gap="sm" wrap="wrap">
            <TextInput
              placeholder="Search notes…"
              value={query}
              onChange={(e) => setQuery(e.currentTarget.value)}
              leftSection={<IconSearch size={15} />}
              size="xs"
              style={{ maxWidth: 260, flex: "0 1 260px" }}
              aria-label="Search notes"
            />
            <MultiSelect
              data={paperOptions}
              value={paperFilter}
              onChange={setPaperFilter}
              placeholder={paperFilter.length ? "" : "All papers"}
              searchable
              clearable
              size="xs"
              variant="filled"
              style={{ maxWidth: 300, flex: "0 1 300px" }}
              aria-label="Restrict to papers"
            />
            <CopyButton value={notesToMarkdown(filtered)}>
              {({ copied, copy }) => (
                <Tooltip label={copied ? "Copied!" : "Copy as Markdown"}>
                  <span>
                    <ActionIcon
                      size="md"
                      variant="subtle"
                      color="gray"
                      aria-label="Copy as Markdown"
                      disabled={filtered.length === 0}
                      onClick={copy}
                    >
                      {copied ? <IconCheck size={15} /> : <IconCopy size={15} />}
                    </ActionIcon>
                  </span>
                </Tooltip>
              )}
            </CopyButton>
          </Group>

          {filtered.length === 0 && (
            <Text size="sm" c="dimmed">
              No notes match your filters.
            </Text>
          )}

          {groups.map((g) => (
            <Stack key={g.paperId} gap="xs">
              <Group gap={8}>
                <Anchor component={Link} to={`/papers/${g.paperId}`} fw={500}>
                  {g.paperTitle}
                </Anchor>
                {g.arxivId && (
                  <Anchor
                    href={`https://arxiv.org/abs/${g.arxivId}`}
                    target="_blank"
                    rel="noreferrer"
                    size="sm"
                  >
                    <Group gap={4} wrap="nowrap">
                      arXiv:{g.arxivId}
                      <IconExternal size={13} />
                    </Group>
                  </Anchor>
                )}
              </Group>
              <Stack gap="sm" pl="sm">
                {g.notes.map((n) => (
                  // Delete icon is a sibling of the clickable content, not nested inside
                  // it — an ActionIcon (button) inside an UnstyledButton (also a button)
                  // is invalid HTML, so it's positioned on top instead (same pattern
                  // PapersPage's stretched-link + overlaid remove button uses).
                  <Box key={n.id} pos="relative">
                    <UnstyledButton
                      onClick={() => openAnnotation(n)}
                      style={{ textAlign: "left", display: "block", width: "100%" }}
                    >
                      <Stack gap={2} pr={28}>
                        <Text size="xs" c="dimmed" fs="italic" lineClamp={2}>
                          “{n.snippet}”
                        </Text>
                        {n.note && <Text size="sm">{n.note}</Text>}
                        <Group gap={6}>
                          {n.section_title && (
                            <Badge variant="light" color="gray" size="xs" radius="sm" fw={500}>
                              {n.section_title}
                            </Badge>
                          )}
                          <Text size="xs" c="dimmed" className="tnum">
                            {relativeTime(n.created_at)}
                          </Text>
                        </Group>
                      </Stack>
                    </UnstyledButton>
                    <ActionIcon
                      size="sm"
                      variant="subtle"
                      color="gray"
                      aria-label="Delete note"
                      pos="absolute"
                      top={0}
                      right={0}
                      onClick={() => void handleDelete(n)}
                    >
                      <IconTrash size={13} />
                    </ActionIcon>
                  </Box>
                ))}
              </Stack>
            </Stack>
          ))}
        </>
      )}
    </Stack>
  );
}
