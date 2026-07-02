import { useEffect, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Container,
  Group,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { getStatus, rescan, type AdminStatus } from "../api";

export default function AdminPage() {
  const [status, setStatus] = useState<AdminStatus | null>(null);

  useEffect(() => {
    const load = () => getStatus().then(setStatus).catch(() => {});
    load();
    const iv = setInterval(load, 1500);
    return () => clearInterval(iv);
  }, []);

  if (!status)
    return (
      <Container>
        <Text>Loading…</Text>
      </Container>
    );

  const ing = status.ingestion;
  const pct = ing.total
    ? Math.round(((ing.done + (ing.current?.pct ?? 0)) / ing.total) * 100)
    : 0;

  return (
    <Container size="lg">
      <Group justify="space-between" mb="md">
        <Title order={3}>Admin</Title>
        <Button variant="light" onClick={() => rescan()}>
          Re-scan config
        </Button>
      </Group>

      <SimpleGrid cols={{ base: 1, sm: 3 }} mb="md">
        <Card withBorder>
          <Text size="xs" c="dimmed">
            Papers
          </Text>
          <Text fz={28} fw={700}>
            {status.db.n_papers}
          </Text>
        </Card>
        <Card withBorder>
          <Text size="xs" c="dimmed">
            Chunks
          </Text>
          <Text fz={28} fw={700}>
            {status.db.n_chunks}
          </Text>
        </Card>
        <Card withBorder>
          <Text size="xs" c="dimmed">
            Pending
          </Text>
          <Text fz={28} fw={700}>
            {status.pending.length}
          </Text>
        </Card>
      </SimpleGrid>

      <Card withBorder mb="md">
        <Group justify="space-between">
          <Text fw={600}>Ingestion</Text>
          <Badge
            color={ing.state === "running" ? "blue" : ing.state === "error" ? "red" : "gray"}
          >
            {ing.state}
          </Badge>
        </Group>
        {ing.state === "running" && (
          <Stack gap={4} mt="sm">
            <Text size="sm">
              {ing.current ? `${ing.current.name} — ${ing.current.stage}` : "starting…"} (
              {ing.done}/{ing.total})
            </Text>
            <Progress value={pct} animated />
          </Stack>
        )}
        {status.pending.length > 0 && (
          <Text size="sm" c="dimmed" mt="sm">
            Pending: {status.pending.join(", ")}
          </Text>
        )}
        {ing.errors.length > 0 && (
          <Alert color="red" mt="sm" title="Errors">
            {ing.errors.map((e, i) => (
              <Text key={i} size="xs">
                {e.name}: {e.error}
              </Text>
            ))}
          </Alert>
        )}
      </Card>

      <Card withBorder>
        <Text fw={600} mb="sm">
          Tags ({status.tags.length})
        </Text>
        <Group gap={6}>
          {status.tags.map((t) => (
            <Badge key={t.tag} variant="light">
              {t.tag} · {t.count}
            </Badge>
          ))}
        </Group>
      </Card>
    </Container>
  );
}
