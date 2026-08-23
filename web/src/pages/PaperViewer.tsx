import { useEffect, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import {
  ActionIcon,
  Affix,
  Anchor,
  Badge,
  Box,
  Center,
  Drawer,
  Group,
  Indicator,
  Loader,
  Paper,
  Stack,
  Text,
  Title,
  Tooltip,
  UnstyledButton,
} from "@mantine/core";
import Markdown from "../components/Markdown";
import { IconExternal, IconList, IconSidebar } from "../components/Icons";
import AnnotationPopover from "../components/AnnotationPopover";
import { buildOutline, type OutlineEntry } from "../outline";
import {
  type Annotation,
  createAnnotation,
  deleteAnnotation,
  getAnnotations,
  getPaper,
  updateAnnotation,
} from "../api";
import {
  addAnnotationHighlight,
  annotationIdAtPoint,
  clearAnnotationHighlights,
  clearHighlight,
  getAnnotationRange,
  highlightPassage,
  registerAnnotationRange,
  removeAnnotationHighlight,
  sectionForNode,
} from "../highlight";

type PaperData = Awaited<ReturnType<typeof getPaper>>;

// A fresh selection needs at least this many characters to open the annotation toolbar —
// matches the noise floor `rangeFor` itself enforces when resolving snippets.
const MIN_SELECTION_LENGTH = 20;

interface PopoverState {
  position: { x: number; y: number };
  mode: "create" | "view";
  annotationId?: string; // view mode
  range?: Range; // create mode: the live selection, cloned
  section: { slug: string; title: string } | null;
  seq: number; // bumped on every open, keys AnnotationPopover so it remounts fresh
}

export default function PaperViewer() {
  const { id } = useParams();
  const location = useLocation() as { state?: { highlight?: string; section?: string } };
  const [data, setData] = useState<PaperData | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  // Annotation ids whose snippet failed to re-anchor (paper re-extracted with different
  // text) — real state, not a render-time read of highlight.ts's registry, so the Notes
  // rail's "not found" label updates on its own instead of riding along on an unrelated
  // re-render that happens to occur after registration.
  const [unresolvedIds, setUnresolvedIds] = useState<Set<string>>(new Set());
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const [railOpen, setRailOpen] = useState(false);
  const [tocOpen, setTocOpen] = useState(false);
  const [outline, setOutline] = useState<OutlineEntry[]>([]);
  const ref = useRef<HTMLDivElement>(null);
  const popoverSeq = useRef(0);
  // Tracks which annotation ids already have a registered highlight range, so the
  // re-render-safety effect below only (re-)resolves ones it hasn't seen yet — a freshly
  // created annotation is registered directly from the live selection range (see
  // handleHighlight/handleSaveNote) and must not be clobbered by a snippet re-resolution.
  const registeredIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (id) getPaper(id).then(setData);
    return clearHighlight;
  }, [id]);

  useEffect(() => {
    // Annotation highlight state (unlike `data`/`annotations`) lives in module scope in
    // highlight.ts, not component state — React Router reuses this component instance
    // across an in-app paper-to-paper navigation, so without this the previous paper's
    // ranges (now pointing at detached DOM) would linger indefinitely.
    clearAnnotationHighlights();
    registeredIds.current = new Set();
    // Resetting local state in response to a route param changing (the user navigated to
    // a different paper) isn't derivable at render time — there's no prop this could be
    // computed from inline, it's synchronizing with an external event (navigation).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setUnresolvedIds(new Set());
    if (id) getAnnotations(id).then(setAnnotations);
  }, [id]);

  useEffect(() => {
    const snippet = location.state?.highlight;
    if (data && snippet && ref.current) {
      const el = ref.current;
      const t = setTimeout(() => highlightPassage(el, snippet), 200);
      return () => clearTimeout(t);
    }
  }, [data, location.state]);

  // Builds the "Contents" rail from the rendered headings once ReactMarkdown has
  // committed its DOM — same 200ms-after-mount wait as the citation-highlight effect
  // above, and for the same reason: headings don't exist until after this render commits.
  useEffect(() => {
    const el = ref.current;
    if (!data || !el) return;
    // Docling always emits the paper's own title as the first `##` heading; it's already
    // shown separately in the page header above the reading pane, so including it here
    // would just duplicate it as the outline's redundant first entry.
    const t = setTimeout(() => setOutline(buildOutline(el).slice(1)), 200);
    return () => clearTimeout(t);
  }, [data]);

  // Restores previously-saved annotation highlights. Depends on `data` (not just `id`) so
  // it re-registers if the reading DOM is ever recreated after mount, and on `annotations`
  // so newly-loaded ones get picked up — but `registeredIds` makes each id idempotent, so
  // this never re-resolves (and can't clobber) an annotation already registered elsewhere.
  useEffect(() => {
    const el = ref.current;
    if (!data || !el) return;
    const failed = new Set<string>();
    for (const a of annotations) {
      if (registeredIds.current.has(a.id)) continue;
      const range = addAnnotationHighlight(a.id, el, a.snippet, a.section_slug);
      if (!range) failed.add(a.id);
      registeredIds.current.add(a.id);
    }
    if (failed.size > 0) {
      // Reporting the outcome of `addAnnotationHighlight` — an imperative DOM/CSS Custom
      // Highlight API call that can only be made (and its result known) after this render
      // committed — is exactly the "calling setState in response to an external system"
      // case the rule's own guidance calls out as fine; it just can't tell that statically.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setUnresolvedIds((prev) => new Set([...prev, ...failed]));
    }
  }, [data, annotations]);

  // Selection toolbar: opens on a non-trivial text selection inside the reading pane.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // `selectionchange` fires repeatedly for what's still effectively the same selection
    // (every intermediate step of a drag-select in real browsers; more than once for a
    // single `addRange` in jsdom) — track the last range acted on so a redundant firing
    // doesn't bump `seq` and remount the popover, discarding any in-progress note draft.
    let lastRange: Range | null = null;

    function onSelectionChange() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
        // Don't close the popover here: focusing the "Add note" textarea itself collapses
        // the page selection (autoFocus moves focus off the selected text), which would
        // otherwise immediately self-close the popover the user just opened. Dismissal is
        // handled by AnnotationPopover's own outside-click `onClose` instead.
        lastRange = null;
        return;
      }
      const range = sel.getRangeAt(0);
      if (!el!.contains(range.commonAncestorContainer)) return;
      const text = sel.toString().replace(/\s+/g, " ").trim();
      if (text.length < MIN_SELECTION_LENGTH) return;
      if (
        lastRange &&
        range.compareBoundaryPoints(Range.START_TO_START, lastRange) === 0 &&
        range.compareBoundaryPoints(Range.END_TO_END, lastRange) === 0
      ) {
        return; // same selection as last time — nothing new to open/reposition
      }
      lastRange = range.cloneRange();
      const rect = range.getBoundingClientRect();
      setPopover({
        position: { x: rect.left + rect.width / 2, y: rect.top },
        mode: "create",
        range: range.cloneRange(),
        section: sectionForNode(el!, range.startContainer),
        seq: ++popoverSeq.current,
      });
    }

    document.addEventListener("selectionchange", onSelectionChange);
    return () => document.removeEventListener("selectionchange", onSelectionChange);
    // `data` (not `id`) is the real gating condition: `ref.current` is null until the
    // reading Box mounts, which only happens once `data` loads — keying on `id` alone
    // reruns this effect once, while `ref.current` is still null, and never again.
  }, [data]);

  // Click on an existing annotated passage: open it in view mode.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    function onClick(e: MouseEvent) {
      const annotationId = annotationIdAtPoint(e.clientX, e.clientY);
      if (!annotationId) return;
      setPopover({
        position: { x: e.clientX, y: e.clientY },
        mode: "view",
        annotationId,
        section: null,
        seq: ++popoverSeq.current,
      });
    }
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
    // Same `data`-gating reasoning as the selection-toolbar effect above; `annotationId`
    // is resolved live via `annotationIdAtPoint` at click time, so `annotations` itself
    // isn't a dependency the closure actually needs.
  }, [data]);

  function closePopover() {
    setPopover(null);
  }

  async function saveNewAnnotation(note: string) {
    if (!id || !popover?.range) return;
    const text = popover.range.toString().replace(/\s+/g, " ").trim();
    const created = await createAnnotation(
      id,
      text,
      popover.section?.title ?? "",
      popover.section?.slug ?? "",
      note,
    );
    registerAnnotationRange(created.id, popover.range);
    registeredIds.current.add(created.id);
    setAnnotations((prev) => [...prev, created]);
  }

  async function handleHighlight() {
    await saveNewAnnotation("");
    closePopover();
  }

  async function handleSaveNote(note: string) {
    if (!id || !popover) return;
    if (popover.mode === "create") {
      await saveNewAnnotation(note);
    } else if (popover.annotationId) {
      const updated = await updateAnnotation(id, popover.annotationId, note);
      setAnnotations((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));
    }
    closePopover();
  }

  async function handleDelete() {
    if (!id || popover?.mode !== "view" || !popover.annotationId) return;
    await deleteAnnotation(id, popover.annotationId);
    removeAnnotationHighlight(popover.annotationId);
    registeredIds.current.delete(popover.annotationId);
    setUnresolvedIds((prev) => {
      if (!prev.has(popover.annotationId!)) return prev;
      const next = new Set(prev);
      next.delete(popover.annotationId!);
      return next;
    });
    setAnnotations((prev) => prev.filter((a) => a.id !== popover.annotationId));
    closePopover();
  }

  function jumpToAnnotation(annotationId: string) {
    const range = getAnnotationRange(annotationId);
    range?.startContainer.parentElement?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  if (!data)
    return (
      <Center mih="60vh">
        <Loader color="accent" />
      </Center>
    );

  const viewingAnnotation =
    popover?.mode === "view" ? annotations.find((a) => a.id === popover.annotationId) : undefined;

  return (
    <Center>
      <Stack gap="lg" w="100%" maw={760} py="md">
        <Stack gap="sm">
          <Title order={1} lh={1.15}>
            {data.title}
          </Title>
          <Group gap={8}>
            {data.arxiv_id && (
              <Anchor
                href={`https://arxiv.org/abs/${data.arxiv_id}`}
                target="_blank"
                rel="noreferrer"
                size="sm"
              >
                <Group gap={4} wrap="nowrap">
                  arXiv:{data.arxiv_id}
                  <IconExternal size={13} />
                </Group>
              </Anchor>
            )}
            {data.tags.map((t) => (
              <Badge key={t} variant="light" color="gray" size="xs" radius="sm" fw={500}>
                {t}
              </Badge>
            ))}
          </Group>
        </Stack>
        <Box ref={ref} className="reading">
          <Markdown>{data.markdown}</Markdown>
        </Box>
      </Stack>

      {!tocOpen && !railOpen && (
        // Fixed to the viewport, not the document flow: a citation jump auto-scrolls the
        // page to the cited passage, which would otherwise carry the Notes toggle and hint
        // off-screen before the user ever saw them — and once scrolled past the title, a
        // non-floating toggle would stay out of reach for the rest of the page. top: 76 =
        // AppShell's 60px header (App.tsx:63) + 16px clearance, so it sits below the header
        // bar instead of overlapping the nav/theme-toggle. Hidden while the rail or the
        // Contents rail is open — either Drawer sits on top of it at the same corner, so
        // showing it then just layers a dead-looking icon over the open panel.
        <Affix position={{ top: 76, right: 16 }}>
          <Stack gap="xs" align="flex-end">
            <Tooltip label="Your notes for this paper">
              <Indicator
                label={annotations.length}
                size={16}
                color="accent"
                disabled={annotations.length === 0}
              >
                <ActionIcon
                  variant="light"
                  color="gray"
                  size="lg"
                  aria-label="Notes"
                  onClick={() => setRailOpen(true)}
                >
                  <IconSidebar size={16} />
                </ActionIcon>
              </Indicator>
            </Tooltip>
            {annotations.length === 0 && (
              <Paper shadow="sm" radius="md" p="xs" withBorder maw={220}>
                <Text size="xs" c="dimmed">
                  Select any passage in the paper to highlight it or add a note.
                </Text>
              </Paper>
            )}
          </Stack>
        </Affix>
      )}

      {!tocOpen && !railOpen && (
        // Mirrors the Notes toggle above (same positioning rationale) but on the left, so
        // the two floating buttons never stack.
        <Affix position={{ top: 76, left: 16 }}>
          <Tooltip label="Jump to a section">
            <ActionIcon
              variant="light"
              color="gray"
              size="lg"
              aria-label="Contents"
              onClick={() => setTocOpen(true)}
            >
              <IconList size={16} />
            </ActionIcon>
          </Tooltip>
        </Affix>
      )}

      <AnnotationPopover
        key={popover?.seq ?? "closed"}
        position={popover?.position ?? null}
        mode={popover?.mode ?? "create"}
        note={viewingAnnotation?.note}
        onHighlight={handleHighlight}
        onSaveNote={handleSaveNote}
        onDelete={handleDelete}
        onClose={closePopover}
      />

      <Drawer
        opened={railOpen}
        onClose={() => setRailOpen(false)}
        position="right"
        title="Notes"
        size="sm"
      >
        <Stack gap="sm">
          {annotations.length === 0 && (
            <Text size="sm" c="dimmed">
              No annotations yet — select text in the paper to add one.
            </Text>
          )}
          {annotations.map((a) => {
            const resolvable = !unresolvedIds.has(a.id);
            return (
              <UnstyledButton
                key={a.id}
                disabled={!resolvable}
                onClick={() => jumpToAnnotation(a.id)}
                style={{ opacity: resolvable ? 1 : 0.6, textAlign: "left" }}
              >
                <Stack gap={2}>
                  <Text size="xs" c="dimmed" lineClamp={1}>
                    {a.snippet}
                  </Text>
                  {a.note && <Text size="sm">{a.note}</Text>}
                  {!resolvable && (
                    <Text size="xs" c="red">
                      Not found in current text
                    </Text>
                  )}
                </Stack>
              </UnstyledButton>
            );
          })}
        </Stack>
      </Drawer>

      <Drawer
        opened={tocOpen}
        onClose={() => setTocOpen(false)}
        position="left"
        title="Contents"
        size="sm"
      >
        <Stack gap={4}>
          {outline.length === 0 && (
            <Text size="sm" c="dimmed">
              No headings found in this paper.
            </Text>
          )}
          {outline.map((entry) => (
            <UnstyledButton
              key={entry.id}
              pl={entry.depth * 12}
              onClick={() =>
                document
                  .getElementById(entry.id)
                  ?.scrollIntoView({ behavior: "smooth", block: "start" })
              }
              style={{ textAlign: "left" }}
            >
              <Text size="sm" lineClamp={2}>
                {entry.text}
              </Text>
            </UnstyledButton>
          ))}
        </Stack>
      </Drawer>
    </Center>
  );
}
