import { useEffect, useState } from "react";
import { Alert, Badge, Card, Container, Group, SimpleGrid, Text, Title } from "@mantine/core";
import { Link } from "react-router-dom";
import { getPapers, type Paper } from "../api";

export default function PapersPage() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getPapers().then((p) => {
      setPapers(p);
      setLoaded(true);
    });
  }, []);

  return (
    <Container size="lg">
      <Title order={3} mb="md">
        Papers ({papers.length})
      </Title>
      {loaded && papers.length === 0 && (
        <Alert color="yellow">
          No papers yet — ingestion may still be running (see Admin).
        </Alert>
      )}
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
        {papers.map((p) => (
          <Card
            key={p.paper_id}
            component={Link}
            to={`/papers/${p.paper_id}`}
            withBorder
            radius="md"
            padding="md"
          >
            <Text fw={600} lineClamp={2}>
              {p.title}
            </Text>
            <Text size="xs" c="dimmed" mt={4}>
              {p.paper_id} · {p.n_chunks ?? 0} chunks
            </Text>
            <Group gap={4} mt="sm">
              {(p.tags ?? []).slice(0, 6).map((t) => (
                <Badge key={t} variant="light" size="xs">
                  {t}
                </Badge>
              ))}
            </Group>
          </Card>
        ))}
      </SimpleGrid>
    </Container>
  );
}
