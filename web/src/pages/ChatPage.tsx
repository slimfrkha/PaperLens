import { useEffect, useRef, useState } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Chip,
  Group,
  Loader,
  MultiSelect,
  SegmentedControl,
  Stack,
  Text,
  Textarea,
  Title,
  Tooltip,
} from "@mantine/core";
import { Link, useNavigate, useParams } from "react-router-dom";
import { IconCheck, IconEdit, IconSend, IconSidebar, IconStop, IconX } from "../components/Icons";
import {
  chat,
  classifyMode,
  createChat,
  deleteChat,
  getChat,
  getPapers,
  getTags,
  listChats,
  setFeedback,
  stopChat,
  type ChatMessage,
  type ChatSummary,
  type Citation,
  type CompareRow,
  type Feedback,
  type Paper,
  type TagCount,
  type TraceEntry,
  type UsageInfo,
} from "../api";
import Answer from "../components/Answer";
import AnswerActions from "../components/AnswerActions";
import ChatSidebar from "../components/ChatSidebar";
import ComparePanel from "../components/ComparePanel";
import FeedbackControl from "../components/FeedbackControl";
import SourceCards from "../components/SourceCards";
import TraceBox from "../components/TraceBox";
import { resolveScopeSize } from "../compareScope";
import { citedCitations } from "../exportAnswer";

// Above this resolved-paper-count, Compare (N sequential search+answer sub-runs plus a
// synthesis pass) is confirmed before sending — a tooltip alone isn't a guard against an
// unfiltered send over a large pool. No backend cap: the backend only enforces the floor
// (<2 papers raises), a slow turn is an inconvenience, not an incident, in a local tool.
const COMPARE_CONFIRM_THRESHOLD = 12;

interface Turn {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  trace?: TraceEntry[];
  streaming?: boolean;
  feedback?: Feedback | null;
  usage?: UsageInfo;
  compare?: boolean;
  compareResults?: CompareRow[];
  compareTotal?: number; // resolved scope size at send time — only set while streaming live
  auto?: boolean; // this turn's Ask/Compare shape was resolved by Auto mode, not picked directly
}

export default function ChatPage() {
  const { chatId } = useParams();
  const navigate = useNavigate();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessions, setSessions] = useState<ChatSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [input, setInput] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagOptions, setTagOptions] = useState<string[]>([]);
  const [papers, setPapers] = useState<string[]>([]);
  const [paperOptions, setPaperOptions] = useState<{ value: string; label: string }[]>([]);
  const [allPapers, setAllPapers] = useState<Paper[]>([]); // for resolveScopeSize — carries tags
  const [perPaper, setPerPaper] = useState(false);
  const [mode, setMode] = useState<"auto" | "ask" | "compare">("auto");
  const [deciding, setDeciding] = useState(false); // Auto's classifyMode() pre-flight in flight
  const [busy, setBusy] = useState(false);
  const [empty, setEmpty] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const loadedId = useRef<string | null>(null); // which chat's turns are in state
  const bottomRef = useRef<HTMLDivElement>(null);
  // The in-flight request's abort handle + the chat_id it's running against — refs, not
  // state, because Stop needs the exact values runTurn started with (chatId's own state
  // can lag a beat behind on a brand-new chat, right after navigate() but before the
  // route param re-renders).
  const abortRef = useRef<AbortController | null>(null);
  const activeChatIdRef = useRef<string | null>(null);
  // onToken concatenates streamed text with no separator of its own — fine within one
  // uninterrupted stretch of tokens, but a trace event (a tool call) means the next batch
  // of tokens is a fresh chunk of prose from a new model round, not a continuation of the
  // same sentence. Without this, "...KV cache handling." and "DeepSeek-V3 reports..." land
  // glued together with no boundary. Set on every trace event, consumed by the next token.
  const pendingSeparatorRef = useRef(false);

  const refreshSessions = () => listChats().then(setSessions);

  useEffect(() => {
    getTags().then((t: TagCount[]) => setTagOptions(t.map((x) => x.tag)));
    getPapers().then((p) => {
      setEmpty(p.length === 0);
      setAllPapers(p);
      setPaperOptions(p.map((x) => ({ value: x.paper_id, label: x.title })));
    });
    refreshSessions();
  }, []);

  // Reset turns/filters synchronously during render when the URL's chatId changes away
  // from a chat — see https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes.
  const [prevChatId, setPrevChatId] = useState(chatId);
  if (chatId !== prevChatId) {
    setPrevChatId(chatId);
    if (!chatId) {
      setTurns([]);
      setTags([]);
      setPapers([]);
      setPerPaper(false);
      setMode("auto");
    }
  }

  // Refs may only be written outside of render (effects/handlers) — clear the
  // "loaded" marker here, once the chatId-cleared render above has committed.
  useEffect(() => {
    if (!chatId) loadedId.current = null;
  }, [chatId]);

  // Load the session named in the URL (restores conversation after navigation).
  useEffect(() => {
    if (!chatId || loadedId.current === chatId) return; // already have it (e.g. just created)
    getChat(chatId)
      .then((s) => {
        setTurns(
          s.messages.map((m, i) => ({
            role: m.role,
            content: m.content,
            citations: s.citations?.[i] ?? undefined,
            trace: s.traces?.[i] ?? undefined,
            feedback: s.feedback?.[i] ?? undefined,
            usage: s.usage?.[i] ?? undefined,
            compare: s.compare?.[i] ?? undefined,
            compareResults: s.compare_results?.[i] ?? undefined,
            auto: s.auto?.[i] ?? undefined,
          })),
        );
        // Filters aren't persisted per chat; reset so a reopened chat never shows
        // another session's stale (and now locked) filter values.
        setTags([]);
        setPapers([]);
        // Neither toggle is a locked filter — each should reflect what the conversation's
        // latest message actually used, not silently reset to a default that may not
        // match (e.g. reloading a chat whose last message used per-paper mode would
        // otherwise show the toggle off, misleading the user about what's about to happen
        // if they send another message without checking).
        const lastPerPaper = s.per_paper?.length ? s.per_paper[s.per_paper.length - 1] : null;
        setPerPaper(lastPerPaper ?? false);
        const lastCompare = s.compare?.length ? s.compare[s.compare.length - 1] : null;
        const lastAuto = s.auto?.length ? s.auto[s.auto.length - 1] : null;
        // Once Auto exists, compare[last] alone no longer says who picked the mode — it
        // can mean "Auto picked Compare." Restore to Auto whenever the last turn was
        // auto-decided, not to its resolved mode, so the control shows what it actually
        // was (same "reflect what the conversation actually used" principle as per_paper's
        // restore above).
        setMode(lastAuto ? "auto" : lastCompare ? "compare" : "ask");
        loadedId.current = chatId;
      })
      .catch(() => setTurns([]));
  }, [chatId]);

  // Block body (not a concise arrow): some smooth-scroll polyfills / browser
  // extensions make scrollIntoView return a Promise, and a concise body would
  // leak that as the effect's cleanup → "destroy is not a function" on unmount.
  //
  // Depend on the tail's own progress signals, not the whole `turns` array — every
  // mutation (patchAt/patchLast) replaces the array with a new reference, and keying
  // off `turns` itself meant setting feedback on ANY turn (even an old one) re-ran
  // this and yanked the view down to the bottom.
  const lastTurn = turns[turns.length - 1];
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, lastTurn?.content, lastTurn?.trace?.length, lastTurn?.streaming]);

  const patchLast = (fn: (t: Turn) => Turn) =>
    setTurns((prev) => {
      const next = [...prev];
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });

  const patchAt = (i: number, fn: (t: Turn) => Turn) =>
    setTurns((prev) => {
      const next = [...prev];
      next[i] = fn(next[i]);
      return next;
    });

  async function onFeedback(i: number, vote: Feedback["vote"], note: string | null) {
    if (!chatId) return;
    patchAt(i, (t) => ({ ...t, feedback: { vote, note } }));
    try {
      await setFeedback(chatId, i, vote, note);
    } catch (e) {
      console.error("Failed to save feedback", e);
    }
  }

  // Shared by send()/sendEdit(): builds history from `prefix` + `question`, appends the
  // new user turn + a streaming assistant placeholder, and runs the chat() SSE call.
  // `prefix` must be captured fresh by the caller right before this call — it's applied
  // directly (not via a setTurns functional updater), so an await between capturing it
  // and calling runTurn could hand it a stale snapshot.
  async function runTurn(
    prefix: Turn[],
    question: string,
    id: string,
    editIndex: number | undefined,
    sendCompare: boolean,
    sendAuto: boolean,
  ) {
    const scopeSize = sendCompare ? resolveScopeSize(allPapers, tags, papers) : 0;
    const history: ChatMessage[] = [
      ...prefix.map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: question },
    ];
    setTurns([
      ...prefix,
      { role: "user", content: question },
      {
        role: "assistant",
        content: "",
        streaming: true,
        compare: sendCompare,
        auto: sendAuto,
        compareResults: sendCompare ? [] : undefined,
        compareTotal: sendCompare ? scopeSize : undefined,
      },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    activeChatIdRef.current = id;
    pendingSeparatorRef.current = false;
    try {
      await chat(
        history,
        tags,
        papers,
        // The secondary knob only applies inside Ask — Compare's per-paper sub-runs
        // already search one paper at a time, so it's never sent under Compare.
        mode === "ask" ? perPaper : false,
        sendCompare,
        id,
        {
          onToken: (tok) =>
            patchLast((t) => {
              const sep = pendingSeparatorRef.current && t.content && !/\n\n$/.test(t.content);
              pendingSeparatorRef.current = false;
              return { ...t, content: t.content + (sep ? "\n\n" : "") + tok };
            }),
          onCitations: (c) => patchLast((t) => ({ ...t, citations: c })),
          onTrace: (e) => {
            pendingSeparatorRef.current = true;
            patchLast((t) => ({ ...t, trace: [...(t.trace ?? []), e] }));
          },
          onUsage: (u) => patchLast((t) => ({ ...t, usage: u })),
          onCompareRow: (row) =>
            patchLast((t) => ({ ...t, compareResults: [...(t.compareResults ?? []), row] })),
          onMeta: () => refreshSessions(),
          onError: (e) => patchLast((t) => ({ ...t, content: t.content + `\n\n_Error: ${e}_` })),
          onDone: () => patchLast((t) => ({ ...t, streaming: false })),
        },
        editIndex,
        controller.signal,
        sendAuto,
      );
    } finally {
      setBusy(false);
      refreshSessions();
    }
  }

  // Compare is N sequential search+answer sub-runs plus a synthesis pass — materially
  // slower than Ask on a large scope. Checked (and, if declined, bailed out of) before
  // send()/sendEdit() touch any state, so a cancel leaves the composer/edit box untouched
  // instead of silently discarding what the user typed. Takes explicit args (not read from
  // `mode`/scopeSize via closure) so both the manually-selected-Compare path and Auto's
  // resolved-to-Compare path share this one threshold check.
  function confirmLargeCompareIfNeeded(isCompare: boolean, n: number): boolean {
    if (!isCompare) return true;
    if (n <= COMPARE_CONFIRM_THRESHOLD) return true;
    return window.confirm(
      `Compare will run ${n} separate paper searches plus a synthesis pass — this can take a while. Continue?`,
    );
  }

  // Resolves what a send should actually do. For explicit Ask/Compare: just the existing
  // confirm gate against the client-computed scopeSize — unchanged behavior, zero added
  // latency. For Auto: first calls classifyMode() (full conversation history) to learn the
  // resolved mode + its own scope_size — a separate pre-flight round trip, since SSE can't
  // pause mid-stream for a confirm — then runs the same gate against that. Bounded by a
  // client-side timeout (no Stop/abort wiring for this single, fast pre-flight call): a
  // classify failure or timeout falls back to sending as Ask, matching the backend's own
  // default-to-ask-on-any-failure rule for classify_mode itself. A decline from the confirm
  // dialog itself still blocks the send either way.
  async function resolveSendMode(
    history: ChatMessage[],
  ): Promise<{ send: boolean; compare: boolean; auto: boolean }> {
    if (mode !== "auto") {
      const isCompare = mode === "compare";
      if (!confirmLargeCompareIfNeeded(isCompare, scopeSize)) {
        return { send: false, compare: false, auto: false };
      }
      return { send: true, compare: isCompare, auto: false };
    }
    setDeciding(true);
    try {
      const timeout = new Promise<never>((_resolve, reject) =>
        setTimeout(() => reject(new Error("classify timed out")), 10000),
      );
      const { mode: resolved, scope_size } = await Promise.race([
        classifyMode(history, tags, papers),
        timeout,
      ]);
      const isCompare = resolved === "compare";
      if (!confirmLargeCompareIfNeeded(isCompare, scope_size)) {
        return { send: false, compare: false, auto: false };
      }
      return { send: true, compare: isCompare, auto: true };
    } catch {
      return { send: true, compare: false, auto: true };
    } finally {
      setDeciding(false);
    }
  }

  // Stops the in-flight turn: aborts our side of the SSE fetch immediately (so the
  // composer unlocks right away, no waiting on the backend) and tells the backend to stop
  // generating too, at its next checkpoint — otherwise the abandoned turn keeps running
  // and holds the chat's single-flight lock until it finishes on its own, so a message
  // sent right after Stop would 409 for however long that takes.
  function stop() {
    abortRef.current?.abort();
    const chatId = activeChatIdRef.current;
    if (chatId) stopChat(chatId).catch((e) => console.error("Failed to stop generation", e));
    patchLast((t) => ({ ...t, streaming: false }));
    setBusy(false);
  }

  async function send() {
    const q = input.trim();
    if (!q || busy || deciding) return;
    const history: ChatMessage[] = [
      ...turns.map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: q },
    ];
    const resolved = await resolveSendMode(history);
    if (!resolved.send) return;
    setInput("");
    // A stale open edit box shouldn't survive an unrelated normal send — otherwise its
    // Save button becomes a silent no-op once `busy` flips true for this turn instead.
    cancelEdit();

    let id = chatId ?? null;
    if (!id) {
      const c = await createChat();
      id = c.id;
      loadedId.current = id; // prevent the load effect from clobbering our turns
      navigate(`/c/${id}`, { replace: true });
    }

    await runTurn(turns, q, id, undefined, resolved.compare, resolved.auto);
  }

  function startEdit(i: number, content: string) {
    setEditingIndex(i);
    setEditDraft(content);
  }

  function cancelEdit() {
    setEditingIndex(null);
    setEditDraft("");
  }

  async function sendEdit(index: number, newContent: string) {
    const q = newContent.trim();
    if (!q || busy || deciding || !chatId) return;

    // Exchanges strictly after this one (own reply + any full user/assistant pairs
    // beyond it) that editing here would discard — confirm only when it's more than
    // just regenerating the immediate reply. Cheap and synchronous, so it's checked
    // before resolveSendMode's possible classifyMode() round trip below — declining it
    // shouldn't have already paid for that call.
    const turnsAfter = turns.length - index - 1;
    if (turnsAfter > 1) {
      const exchanges = Math.floor((turnsAfter - 1) / 2);
      const noun = exchanges === 1 ? "exchange" : "exchanges";
      if (
        !window.confirm(
          `This will remove ${exchanges} later ${noun} in this conversation. Continue?`,
        )
      ) {
        return;
      }
    }

    const prefix = turns.slice(0, index);
    const history: ChatMessage[] = [
      ...prefix.map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: q },
    ];
    const resolved = await resolveSendMode(history);
    if (!resolved.send) return;

    setEditingIndex(null);
    await runTurn(prefix, q, chatId, index, resolved.compare, resolved.auto);
  }

  async function onDelete(id: string) {
    await deleteChat(id);
    await refreshSessions();
    if (id === chatId) navigate("/");
  }

  // New chat: clear both filters (also covers the case where we're already at "/",
  // where navigating wouldn't re-run the load effect).
  function newChat() {
    setTags([]);
    setPapers([]);
    setPerPaper(false);
    setMode("auto");
    navigate("/");
  }

  // Derived, not stored state — mirrors the backend's tag/paper-intersection +
  // manifest-fallback scope resolution (agent.py) over the already-fetched paper list,
  // so Compare can disable itself / warn without a new request per keystroke.
  const scopeSize = resolveScopeSize(allPapers, tags, papers);

  const composer = (
    <Box className="composer" p={6}>
      <Group align="flex-end" gap={6} wrap="nowrap">
        <Tooltip
          label={
            scopeSize < 2
              ? "Compare needs at least 2 papers in scope"
              : `Auto decides Ask vs Compare per question. Compare runs one guaranteed search+answer per paper (${scopeSize} in scope), then synthesizes them into one comparative answer — slower than Ask on a large scope.`
          }
          multiline
          w={260}
        >
          {/* alignSelf overrides the row's flex-end (which bottom-aligns against the
              taller textarea/send button) so the control centers instead. */}
          <Box style={{ alignSelf: "center" }}>
            <SegmentedControl
              size="xs"
              value={mode}
              onChange={(v) => setMode(v as "auto" | "ask" | "compare")}
              data={[
                { label: "Auto", value: "auto" },
                { label: "Ask", value: "ask" },
                { label: "Compare", value: "compare", disabled: scopeSize < 2 },
              ]}
            />
          </Box>
        </Tooltip>
        {mode === "ask" && (
          <Tooltip
            label="Runs retrieval once per paper instead of once over the whole library, so one paper with many relevant chunks can't crowd out the others before reranking."
            multiline
            w={260}
          >
            {/* Chip's own input is 0x0 (the visible surface is a sibling label), so
                Tooltip's hover handlers need this wrapper to land on a hoverable box. */}
            <Box style={{ alignSelf: "center" }}>
              <Chip
                size="xs"
                variant="light"
                checked={perPaper}
                onChange={setPerPaper}
                aria-label="Broaden recall per paper"
              >
                Broaden recall
              </Chip>
            </Box>
          </Tooltip>
        )}
        <Textarea
          flex={1}
          variant="unstyled"
          autosize
          minRows={1}
          maxRows={8}
          value={input}
          // Greyed out during Auto's classify round trip: send()/sendEdit() already
          // captured this question before the pre-flight call started, so further edits
          // here would silently have no effect on the in-flight decision.
          disabled={deciding}
          placeholder={
            mode === "compare"
              ? `Ask one thing to compare across ${scopeSize} papers…`
              : mode === "auto"
                ? "Ask about a paper or a concept — Auto will pick how to search…"
                : "Ask about a paper or a concept…"
          }
          styles={{ input: { paddingInline: 10, fontSize: "0.95rem" } }}
          onChange={(e) => setInput(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        {deciding && (
          <Group gap={4} wrap="nowrap" style={{ alignSelf: "center" }}>
            <Loader size="xs" type="dots" color="accent" />
            <Text size="xs" c="dimmed">
              Deciding…
            </Text>
          </Group>
        )}
        {busy ? (
          <ActionIcon size={38} radius="md" onClick={stop} aria-label="Stop generating">
            <IconStop size={16} />
          </ActionIcon>
        ) : (
          <ActionIcon
            size={38}
            radius="md"
            onClick={() => send()}
            disabled={!input.trim() || deciding}
            aria-label="Send"
          >
            <IconSend size={18} />
          </ActionIcon>
        )}
      </Group>
    </Box>
  );

  return (
    <Group
      align="stretch"
      gap="lg"
      wrap="nowrap"
      style={{ minHeight: "calc(100vh - 60px - 3rem)" }}
    >
      {sidebarOpen && (
        <ChatSidebar
          sessions={sessions}
          activeId={chatId}
          onNew={newChat}
          onSelect={(id) => navigate(`/c/${id}`)}
          onDelete={onDelete}
        />
      )}
      <Stack style={{ flex: 1, minWidth: 0 }} gap="md">
        <Group gap="xs" justify="space-between">
          <Tooltip label={sidebarOpen ? "Hide chats" : "Show chats"}>
            <ActionIcon variant="subtle" color="gray" onClick={() => setSidebarOpen((o) => !o)}>
              <IconSidebar size={18} />
            </ActionIcon>
          </Tooltip>
          <Tooltip
            label="Filters are fixed once the conversation starts — use New chat to change them"
            disabled={turns.length === 0}
            multiline
            w={240}
          >
            <Group gap="xs" wrap="wrap" justify="flex-end" style={{ flex: 1 }}>
              <MultiSelect
                data={paperOptions}
                value={papers}
                onChange={setPapers}
                placeholder={papers.length ? "" : "All papers"}
                disabled={turns.length > 0}
                searchable
                clearable
                size="xs"
                variant="filled"
                style={{ maxWidth: 300, flex: "0 1 300px" }}
                aria-label="Restrict search to papers"
              />
              <MultiSelect
                data={tagOptions}
                value={tags}
                onChange={setTags}
                placeholder={tags.length ? "" : "All tags"}
                disabled={turns.length > 0}
                searchable
                clearable
                size="xs"
                variant="filled"
                style={{ maxWidth: 260, flex: "0 1 260px" }}
                aria-label="Restrict search to tags"
              />
            </Group>
          </Tooltip>
        </Group>

        {empty && (
          <Alert color="yellow" variant="light" title="The library is empty" radius="md">
            No papers indexed yet — ingestion may still be running. Check the{" "}
            <Link to="/admin">Admin</Link> page.
          </Alert>
        )}

        {turns.length === 0 ? (
          <EmptyHero />
        ) : (
          <Stack gap="xl" style={{ flex: 1 }}>
            {turns.map((t, i) =>
              t.role === "user" ? (
                <Group key={i} justify="flex-end" align="center" gap={4}>
                  {editingIndex === i ? (
                    <Box style={{ maxWidth: "82%", width: "100%" }}>
                      <Textarea
                        autosize
                        minRows={1}
                        maxRows={8}
                        autoFocus
                        value={editDraft}
                        onChange={(e) => setEditDraft(e.currentTarget.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            sendEdit(i, editDraft);
                          } else if (e.key === "Escape") {
                            cancelEdit();
                          }
                        }}
                      />
                      <Group gap={4} justify="flex-end" mt={4}>
                        <ActionIcon
                          size="sm"
                          variant="subtle"
                          color="gray"
                          aria-label="Cancel edit"
                          onClick={cancelEdit}
                        >
                          <IconX size={14} />
                        </ActionIcon>
                        <ActionIcon
                          size="sm"
                          variant="subtle"
                          aria-label="Save and resend"
                          disabled={!editDraft.trim() || busy}
                          onClick={() => sendEdit(i, editDraft)}
                        >
                          <IconCheck size={14} />
                        </ActionIcon>
                      </Group>
                    </Box>
                  ) : (
                    <>
                      <ActionIcon
                        size="sm"
                        variant="subtle"
                        color="gray"
                        aria-label="Edit message"
                        disabled={busy}
                        onClick={() => startEdit(i, t.content)}
                      >
                        <IconEdit size={14} />
                      </ActionIcon>
                      <Box
                        px="md"
                        py="xs"
                        style={{
                          maxWidth: "82%",
                          background: "var(--pl-surface-2)",
                          border: "1px solid var(--pl-border)",
                          borderRadius: 14,
                          borderBottomRightRadius: 4,
                        }}
                      >
                        <Text style={{ whiteSpace: "pre-wrap" }}>{t.content}</Text>
                      </Box>
                    </>
                  )}
                </Group>
              ) : (
                <Box key={i}>
                  {t.auto && (
                    <Badge size="xs" variant="outline" color="gray" mb={4}>
                      Auto
                    </Badge>
                  )}
                  {t.compare ? (
                    <ComparePanel
                      rows={t.compareResults ?? []}
                      totalPapers={t.compareTotal ?? t.compareResults?.length ?? 0}
                      streaming={t.streaming}
                      // The synthesis pass's own text streams through the same `content`
                      // field a normal answer uses — the moment any of it has arrived,
                      // every per-paper sub-run is done and synthesis has started.
                      synthesizing={t.streaming && !!t.content}
                    />
                  ) : (
                    t.trace && <TraceBox entries={t.trace} streaming={t.streaming} />
                  )}
                  {t.content ? (
                    <Answer text={t.content} citations={t.citations ?? []} />
                  ) : t.streaming ? (
                    <Group gap="xs">
                      <Loader size="sm" type="dots" color="accent" />
                      <Text size="sm" c="dimmed">
                        Thinking…
                      </Text>
                    </Group>
                  ) : null}
                  {!t.streaming && t.citations && (
                    <SourceCards citations={citedCitations(t.content, t.citations)} />
                  )}
                  {!t.streaming && t.usage && (
                    <Text size="xs" c="dimmed" mt={4}>
                      {formatUsage(t.usage)}
                    </Text>
                  )}
                  {!t.streaming && t.content && chatId && (
                    <FeedbackControl
                      // Scoped to chatId, not just the array index — otherwise switching
                      // chats reuses this component instance (ChatPage doesn't remount on
                      // a chatId param change) and its local draft/note-open state leaks
                      // from the previous chat's turn at the same index.
                      key={`${chatId}-${i}`}
                      vote={t.feedback?.vote ?? null}
                      note={t.feedback?.note ?? null}
                      onChange={(vote, note) => onFeedback(i, vote, note)}
                    />
                  )}
                  {!t.streaming && t.content && (
                    <AnswerActions text={t.content} citations={t.citations ?? []} />
                  )}
                </Box>
              ),
            )}
            <div ref={bottomRef} />
          </Stack>
        )}

        <Box style={{ position: "sticky", bottom: 0, paddingBottom: 4 }}>{composer}</Box>
      </Stack>
    </Group>
  );
}

function formatUsage(u: UsageInfo): string {
  const parts: string[] = [];
  if (u.input_tokens != null && u.output_tokens != null) {
    const total = u.input_tokens + u.output_tokens;
    parts.push(`${total.toLocaleString()} token${total === 1 ? "" : "s"}`);
  }
  parts.push(`${(u.latency_ms / 1000).toFixed(1)}s`);
  return parts.join(" · ");
}

function EmptyHero() {
  return (
    <Stack align="center" justify="center" gap={6} style={{ flex: 1, textAlign: "center" }} py="xl">
      <Title order={1} fw={500} style={{ letterSpacing: "-0.02em" }}>
        What do you want to understand?
      </Title>
      <Text c="dimmed" maw={520}>
        Ask across the indexed arXiv papers. Answers cite the exact passage — click any{" "}
        <span className="cite">n</span> to open its source.
      </Text>
    </Stack>
  );
}
