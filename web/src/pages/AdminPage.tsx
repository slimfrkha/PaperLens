import { useEffect, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Center,
  Group,
  Loader,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { IconPlus, IconRescan } from "../components/Icons";
import { addPaper, getStatus, rescan, type AdminStatus } from "../api";

export default function AdminPage() {
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [newPaper, setNewPaper] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  const load = () =>
    getStatus()
      .then(setStatus)
      .catch(() => {});

  useEffect(() => {
    load();
    const iv = setInterval(load, 1500);
    return () => clearInterval(iv);
  }, []);

  const handleAddPaper = async () => {
    const raw = newPaper.trim();
    if (!raw) return;
    setAdding(true);
    setAddError(null);
    try {
      await addPaper(raw);
      setNewPaper("");
      load(); // show the new pending paper without waiting for the next poll tick
    } catch (e) {
      setAddError(e instanceof Error ? e.message : "failed to add paper");
    } finally {
      setAdding(false);
    }
  };

  if (!status)
    return (
      <Center mih="60vh">
        <Loader color="accent" />
      </Center>
    );

  const ing = status.ingestion;
  const pct = ing.total ? Math.round(((ing.done + (ing.current?.pct ?? 0)) / ing.total) * 100) : 0;

  return (
    <Stack gap="lg">
      <Group justify="space-between">
        <Title order={2}>Admin</Title>
        <Button variant="default" leftSection={<IconRescan size={16} />} onClick={() => rescan()}>
          Re-scan config
        </Button>
      </Group>

      <Card withBorder radius="md">
        <Text fw={600} mb="sm">
          Add paper
        </Text>
        <Group align="flex-end" gap="sm">
          <TextInput
            placeholder="arXiv id or URL, e.g. 2412.19437"
            value={newPaper}
            onChange={(e) => setNewPaper(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddPaper()}
            disabled={adding}
            style={{ flex: 1 }}
          />
          <Button leftSection={<IconPlus size={16} />} onClick={handleAddPaper} loading={adding}>
            Add paper
          </Button>
        </Group>
        {addError && (
          <Text size="sm" c="red" mt="xs">
            {addError}
          </Text>
        )}
      </Card>

      <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
        <Stat label="Papers" value={status.db.n_papers} />
        <Stat label="Chunks" value={status.db.n_chunks} />
        <Stat label="Pending" value={status.pending.length} />
      </SimpleGrid>

      <Card withBorder radius="md">
        <Group justify="space-between">
          <Text fw={600}>Ingestion</Text>
          <Badge
            variant="light"
            color={ing.state === "running" ? "accent" : ing.state === "error" ? "red" : "gray"}
          >
            {ing.state}
          </Badge>
        </Group>
        {ing.state === "running" && (
          <Stack gap={6} mt="sm">
            <Text size="sm" className="tnum">
              {ing.current ? `${ing.current.name} — ${ing.current.stage}` : "starting…"} ({ing.done}
              /{ing.total})
            </Text>
            <Progress value={pct} color="accent" animated radius="xl" />
          </Stack>
        )}
        {status.pending.length > 0 && (
          <Text size="sm" c="dimmed" mt="sm">
            Pending: {status.pending.join(", ")}
          </Text>
        )}
        {ing.errors.length > 0 && (
          <Alert color="red" variant="light" mt="sm" title="Errors" radius="md">
            {ing.errors.map((e, i) => (
              <Text key={i} size="xs">
                {e.name}: {e.error}
              </Text>
            ))}
          </Alert>
        )}
      </Card>

      <Card withBorder radius="md">
        <Text fw={600} mb="sm">
          Tags ({status.tags.length})
        </Text>
        <Group gap={6}>
          {status.tags.map((t) => (
            <Badge key={t.tag} variant="light" color="gray" radius="sm" fw={500}>
              {t.tag} · {t.count}
            </Badge>
          ))}
        </Group>
      </Card>
    </Stack>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <Card withBorder radius="md">
      <Text size="xs" c="dimmed" tt="uppercase" style={{ letterSpacing: "0.05em" }}>
        {label}
      </Text>
      <Text fz={32} fw={600} className="tnum" mt={4} ff="'Newsreader', Georgia, serif">
        {value}
      </Text>
    </Card>
  );
}
