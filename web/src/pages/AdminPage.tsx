import { useEffect, useRef, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Center,
  Checkbox,
  Group,
  Loader,
  Progress,
  ScrollArea,
  SimpleGrid,
  Stack,
  TagsInput,
  Text,
  Title,
} from "@mantine/core";
import { IconPlus, IconRescan } from "../components/Icons";
import {
  addPapers,
  getStatus,
  getSuggestedPapers,
  rescan,
  type AdminStatus,
  type AddPaperResult,
  type SuggestedPaper,
} from "../api";

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
  const [suggested, setSuggested] = useState<SuggestedPaper[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [addingSuggested, setAddingSuggested] = useState(false);
  // Suggestions added this session but not yet reconciled away by a completion refresh.
  // Keeps a row greyed continuously from the click until the list is rebuilt, so it never
  // flickers back to selectable in the gap between `pending` clearing and the refetch.
  const [justAdded, setJustAdded] = useState<Set<string>>(new Set());

  const load = () =>
    getStatus()
      .then(setStatus)
      .catch(() => {});

  // Fetched once on mount and re-fetched after an add — deliberately NOT on the 1.5s
  // status poll: it re-reads and regexes every paper's markdown, cheap now but O(pool).
  const loadSuggested = () =>
    getSuggestedPapers()
      .then(setSuggested)
      .catch(() => setSuggested([]));

  useEffect(() => {
    load();
    loadSuggested();
    const iv = setInterval(load, 1500);
    return () => clearInterval(iv);
  }, []);

  // Refresh suggestions when an ingestion run finishes (running -> idle): a just-added
  // paper leaves the list (it's now in the pool), and the papers *it* cites join it.
  // The list content changes underfoot, so send the scroll box back to the top.
  const prevIngState = useRef<string | null>(null);
  const suggestedViewport = useRef<HTMLDivElement>(null);
  const ingState = status?.ingestion.state ?? null;
  useEffect(() => {
    if (prevIngState.current === "running" && ingState === "idle") {
      // Do the un-grey, scroll reset, and list swap together, once the new list is in —
      // not piecemeal as `pending` and the fetch land at different times.
      loadSuggested().then(() => {
        suggestedViewport.current?.scrollTo?.({ top: 0 });
        setJustAdded(new Set());
      });
    }
    prevIngState.current = ingState;
  }, [ingState]);

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

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
      loadSuggested(); // a manually-added paper drops off the suggested list
    } catch (e) {
      setAddError(e instanceof Error ? e.message : "failed to add paper(s)");
    } finally {
      setAdding(false);
    }
  };

  const handleAddSuggested = async () => {
    const ids = [...selected];
    if (ids.length === 0) return;
    setAddingSuggested(true);
    setAddError(null);
    setAddResults(null);
    try {
      const { results } = await addPapers(ids);
      setAddResults(results);
      setSelected(new Set());
      setJustAdded((prev) => new Set([...prev, ...ids])); // keep these rows greyed until reconciled
      load();
      loadSuggested();
    } catch (e) {
      setAddError(e instanceof Error ? e.message : "failed to add paper(s)");
    } finally {
      setAddingSuggested(false);
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
  // Papers added but not yet ingested — queued or in progress. `pending` lists them by
  // name, which for an added paper equals its arxiv_id, so it keys the suggested rows.
  const inFlight = new Set(status.pending);

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

      {suggested && suggested.length > 0 && (
        <Card withBorder radius="md">
          <Group justify="space-between" mb="sm">
            <Text fw={600}>Suggested from your pool ({suggested.length})</Text>
            <Button
              size="xs"
              leftSection={<IconPlus size={14} />}
              onClick={handleAddSuggested}
              loading={addingSuggested}
              disabled={selected.size === 0}
            >
              Add {selected.size} selected
            </Button>
          </Group>
          <Text size="xs" c="dimmed" mb="sm">
            arXiv papers cited by your pool but not yet in it, most-cited first.
          </Text>
          {/* Fixed-height scroll box: the full list is kept (a big pool has a long
              tail of papers cited once), just bounded so it never dominates the page. */}
          <ScrollArea.Autosize mah={320} type="auto" viewportRef={suggestedViewport}>
            <Stack gap={8} pr="sm">
              {suggested.map((s) => {
                // Already added and still ingesting: `pending` lists it (arxiv_id=name)
                // once the poll catches up, and `justAdded` covers the gap right after the
                // click. Grey it and block re-adding, which would just return a "duplicate".
                const queued = inFlight.has(s.arxiv_id) || justAdded.has(s.arxiv_id);
                return (
                  <Group
                    key={s.arxiv_id}
                    gap="sm"
                    wrap="nowrap"
                    align="flex-start"
                    style={{ opacity: queued ? 0.55 : 1 }}
                  >
                    <Checkbox
                      checked={selected.has(s.arxiv_id)}
                      onChange={() => toggle(s.arxiv_id)}
                      disabled={queued}
                      aria-label={`Select ${s.arxiv_id}`}
                      mt={2}
                    />
                    <Badge variant="light" color="gray" radius="sm" className="tnum">
                      {s.cited_by}×
                    </Badge>
                    <Text size="sm" style={{ flex: 1 }} lineClamp={2}>
                      {s.label}
                    </Text>
                    {queued ? (
                      <Badge variant="light" color="yellow" radius="sm">
                        queued
                      </Badge>
                    ) : (
                      <Text size="xs" c="dimmed" className="tnum">
                        {s.arxiv_id}
                      </Text>
                    )}
                  </Group>
                );
              })}
            </Stack>
          </ScrollArea.Autosize>
        </Card>
      )}

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
