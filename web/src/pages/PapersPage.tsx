import { useEffect, useState } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Card,
  Group,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { Link } from "react-router-dom";
import { getPapers, removePaper, type Paper } from "../api";
import { IconTrash } from "../components/Icons";

export default function PapersPage() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loaded, setLoaded] = useState(false);

  const load = () =>
    getPapers().then((p) => {
      setPapers(p);
      setLoaded(true);
    });

  useEffect(() => {
    load();
  }, []);

  const handleRemove = async (paperId: string, title: string) => {
    if (
      !window.confirm(`Remove "${title}"? This deletes its index, cached files, and config entry.`)
    ) {
      return;
    }
    try {
      await removePaper(paperId);
      load();
    } catch (e) {
      alert(e instanceof Error ? e.message : "failed to remove paper");
    }
  };

  return (
    <Stack gap="lg">
      <Group align="baseline" gap="sm">
        <Title order={2}>Papers</Title>
        <Text c="dimmed" className="tnum">
          {papers.length}
        </Text>
      </Group>

      {loaded && papers.length === 0 && (
        <Alert color="yellow" variant="light" radius="md">
          No papers yet — ingestion may still be running (see Admin).
        </Alert>
      )}

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
        {papers.map((p) => (
          <Card
            key={p.paper_id}
            className="paper-card"
            withBorder
            radius="md"
            padding="md"
            pos="relative"
            style={{ height: "100%" }}
          >
            {/* Stretched-link pattern: an invisible full-cover anchor catches clicks
                for the whole card without nesting a <button> inside an <a> (invalid
                HTML, unreliable for screen readers/keyboard nav). It sits above the
                (unstyled) card content for hit-testing but paints nothing, so the
                visible title/tags underneath still show through; the remove button
                sits in a higher z-index layer still so its own clicks aren't
                intercepted by the link. */}
            <Link
              to={`/papers/${p.paper_id}`}
              aria-label={p.title}
              style={{ position: "absolute", inset: 0, zIndex: 1 }}
            />
            <ActionIcon
              size="sm"
              variant="subtle"
              color="gray"
              aria-label="Remove paper"
              pos="absolute"
              top={8}
              right={8}
              style={{ zIndex: 2 }}
              onClick={() => void handleRemove(p.paper_id, p.title)}
            >
              <IconTrash size={15} />
            </ActionIcon>
            <Stack gap="xs" h="100%" justify="space-between">
              <div>
                <Text
                  fw={500}
                  lineClamp={2}
                  ff="'Newsreader', Georgia, serif"
                  fz="1.05rem"
                  lh={1.3}
                >
                  {p.title}
                </Text>
                <Text size="xs" c="dimmed" mt={6} className="tnum">
                  {p.paper_id} · {p.n_chunks ?? 0} chunks
                </Text>
              </div>
              {(p.tags ?? []).length > 0 && (
                <Group gap={5}>
                  {(p.tags ?? []).slice(0, 5).map((t) => (
                    <Badge key={t} variant="light" color="gray" size="xs" radius="sm" fw={500}>
                      {t}
                    </Badge>
                  ))}
                </Group>
              )}
            </Stack>
          </Card>
        ))}
      </SimpleGrid>
    </Stack>
  );
}
