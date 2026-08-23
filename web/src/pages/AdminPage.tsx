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
  TagsInput,
  Text,
  Title,
} from "@mantine/core";
import { IconPlus, IconRescan } from "../components/Icons";
import { addPapers, getStatus, rescan, type AdminStatus, type AddPaperResult } from "../api";

const statusGlyph = (s: AddPaperResult["status"]) =>
  s === "queued" ? "✓" : s === "duplicate" ? "⚠" : "✗";
const statusColor = (s: AddPaperResult["status"]) =>
  s === "queued" ? "green" : s === "duplicate" ? "yellow" : "red";
const statusDetail = (r: AddPaperResult) => {
  if (r.status === "duplicate") return ` — already curated as ${r.existing_name}`;
  if (r.status === "invalid") return " — not a recognizable arXiv id or URL";
  if (r.status === "error") return ` — ${r.detail}`;
  return " — queued";
};

export default function AdminPage() {
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [paperIds, setPaperIds] = useState<string[]>([]);
  const [tagsSearch, setTagsSearch] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const [addResults, setAddResults] = useState<AddPaperResult[] | null>(null);
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

  // Text typed but not yet turned into a pill (no Enter/Tab/Space/comma pressed) —
  // included below in both the disabled check and the submitted list, so clicking
  // Add right after typing (the single most common path) isn't a silent no-op.
  const pendingId = tagsSearch.trim();

  const handleAdd = async () => {
    const ids = pendingId && !paperIds.includes(pendingId) ? [...paperIds, pendingId] : paperIds;
    if (ids.length === 0) return;
    setAdding(true);
    setAddError(null);
    setAddResults(null);
    setPaperIds(ids);
    setTagsSearch("");
    try {
      const { results } = await addPapers(ids);
      setAddResults(results);
      setPaperIds([]);
      load(); // show the new pending paper(s) without waiting for the next poll tick
    } catch (e) {
      setAddError(e instanceof Error ? e.message : "failed to add paper(s)");
    } finally {
      setAdding(false);
    }
  };

  // TagsInput's splitChars only covers single-character triggers (space, comma,
  // paste-newlines below) — Tab is a multi-character `event.key`, so it can't sit in
  // that array without corrupting paste-splitting (which reuses splitChars as a regex
  // character class). Commit it by hand instead.
  const handleTagsKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Tab") return;
    const raw = tagsSearch.trim();
    if (!raw) return;
    event.preventDefault();
    setPaperIds((prev) => (prev.includes(raw) ? prev : [...prev, raw]));
    setTagsSearch("");
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
          <TagsInput
            placeholder="Paste or type arXiv ids/URLs — space, tab, enter, or comma to add"
            value={paperIds}
            onChange={setPaperIds}
            searchValue={tagsSearch}
            onSearchChange={setTagsSearch}
            onKeyDown={handleTagsKeyDown}
            splitChars={[",", " ", "\n"]}
            disabled={adding}
            style={{ flex: 1 }}
          />
          <Button
            leftSection={<IconPlus size={16} />}
            onClick={handleAdd}
            loading={adding}
            disabled={paperIds.length === 0 && !pendingId}
          >
            Add paper(s)
          </Button>
        </Group>
        {addError && (
          <Text size="sm" c="red" mt="xs">
            {addError}
          </Text>
        )}
        {addResults && (
          <Stack gap={4} mt="sm">
            {addResults.map((r, i) => (
              <Text key={i} size="sm" c={statusColor(r.status)}>
                {statusGlyph(r.status)} {r.input}
                {statusDetail(r)}
              </Text>
            ))}
          </Stack>
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
