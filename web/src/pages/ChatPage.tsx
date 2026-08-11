import { useEffect, useRef, useState } from "react";
import {
  ActionIcon,
  Alert,
  Box,
  Chip,
  Group,
  Loader,
  MultiSelect,
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
  type Feedback,
  type TagCount,
  type TraceEntry,
  type UsageInfo,
} from "../api";
import Answer from "../components/Answer";
import AnswerActions from "../components/AnswerActions";
import ChatSidebar from "../components/ChatSidebar";
import FeedbackControl from "../components/FeedbackControl";
import SourceCards from "../components/SourceCards";
import TraceBox from "../components/TraceBox";

interface Turn {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  trace?: TraceEntry[];
  streaming?: boolean;
  feedback?: Feedback | null;
  usage?: UsageInfo;
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
  const [perPaper, setPerPaper] = useState(false);
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

  const refreshSessions = () => listChats().then(setSessions);

  useEffect(() => {
    getTags().then((t: TagCount[]) => setTagOptions(t.map((x) => x.tag)));
    getPapers().then((p) => {
      setEmpty(p.length === 0);
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
          })),
        );
        // Filters aren't persisted per chat; reset so a reopened chat never shows
        // another session's stale (and now locked) filter values.
        setTags([]);
        setPapers([]);
        // Per-paper isn't a locked filter — it should reflect what the conversation's
        // latest message actually used, not silently reset to a default that may not
        // match (e.g. reloading a chat whose last message used per-paper mode would
        // otherwise show the toggle off, misleading the user about what's about to happen
        // if they send another message without checking).
        const lastPerPaper = s.per_paper?.length ? s.per_paper[s.per_paper.length - 1] : null;
        setPerPaper(lastPerPaper ?? false);
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
  async function runTurn(prefix: Turn[], question: string, id: string, editIndex?: number) {
    const history: ChatMessage[] = [
      ...prefix.map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: question },
    ];
    setTurns([
      ...prefix,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    activeChatIdRef.current = id;
    try {
      await chat(
        history,
        tags,
        papers,
        perPaper,
        id,
        {
          onToken: (tok) => patchLast((t) => ({ ...t, content: t.content + tok })),
          onCitations: (c) => patchLast((t) => ({ ...t, citations: c })),
          onTrace: (e) => patchLast((t) => ({ ...t, trace: [...(t.trace ?? []), e] })),
          onUsage: (u) => patchLast((t) => ({ ...t, usage: u })),
          onMeta: () => refreshSessions(),
          onError: (e) => patchLast((t) => ({ ...t, content: t.content + `\n\n_Error: ${e}_` })),
          onDone: () => patchLast((t) => ({ ...t, streaming: false })),
        },
        editIndex,
        controller.signal,
      );
    } finally {
      setBusy(false);
      refreshSessions();
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
    if (!q || busy) return;
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

    await runTurn(turns, q, id);
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
    if (!q || busy || !chatId) return;

    // Exchanges strictly after this one (own reply + any full user/assistant pairs
    // beyond it) that editing here would discard — confirm only when it's more than
    // just regenerating the immediate reply.
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

    setEditingIndex(null);
    await runTurn(turns.slice(0, index), q, chatId, index);
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
    navigate("/");
  }

  const composer = (
    <Box className="composer" p={6}>
      <Group align="flex-end" gap={6} wrap="nowrap">
        <Tooltip
          label="Runs retrieval once per paper instead of once over the whole library, so one paper with many relevant chunks can't crowd out the others before reranking."
          multiline
          w={260}
        >
          {/* Chip's own input is 0x0 (the visible surface is a sibling label), so
              Tooltip's hover handlers need this wrapper to land on a hoverable box.
              alignSelf overrides the row's flex-end (which bottom-aligns against the
              taller textarea/send button) so the short pill centers instead. */}
          <Box style={{ alignSelf: "center" }}>
            <Chip
              size="xs"
              variant="light"
              checked={perPaper}
              onChange={setPerPaper}
              aria-label="Search each paper separately"
            >
              Per-paper
            </Chip>
          </Box>
        </Tooltip>
        <Textarea
          flex={1}
          variant="unstyled"
          autosize
          minRows={1}
          maxRows={8}
          value={input}
          placeholder="Ask about a paper or a concept…"
          styles={{ input: { paddingInline: 10, fontSize: "0.95rem" } }}
          onChange={(e) => setInput(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        {busy ? (
          <ActionIcon size={38} radius="md" onClick={stop} aria-label="Stop generating">
            <IconStop size={16} />
          </ActionIcon>
        ) : (
          <ActionIcon
            size={38}
            radius="md"
            onClick={() => send()}
            disabled={!input.trim()}
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
                  {t.trace && <TraceBox entries={t.trace} streaming={t.streaming} />}
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
                  {!t.streaming && t.citations && <SourceCards citations={t.citations} />}
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
