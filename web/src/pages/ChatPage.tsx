import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Chip,
  Group,
  Loader,
  MultiSelect,
  SegmentedControl,
  Select,
  Stack,
  Text,
  Textarea,
  Title,
  Tooltip,
} from "@mantine/core";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  IconChevron,
  IconCheck,
  IconDownload,
  IconEdit,
  IconPanelCenter,
  IconPanelLeft,
  IconPanelRight,
  IconSend,
  IconSidebar,
  IconStop,
  IconX,
} from "../components/Icons";
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
  type Feedback,
  type Paper,
  type StoredTurn,
  type TagCount,
  type UsageInfo,
} from "../api";
import PaperPanel, { type HighlightTarget } from "../components/PaperPanel";
import Answer from "../components/Answer";
import AnswerActions from "../components/AnswerActions";
import ChatSidebar from "../components/ChatSidebar";
import ComparePanel from "../components/ComparePanel";
import FeedbackControl from "../components/FeedbackControl";
import SourceCards from "../components/SourceCards";
import TraceBox from "../components/TraceBox";
import { resolveScopeIds, resolveScopeSize } from "../compareScope";
import { citedCitations } from "../exportAnswer";
import { conversationToMarkdown } from "../exportConversation";
import { downloadTextFile, slugFilename } from "../download";

// Above this resolved-paper-count, Compare (N sequential search+answer sub-runs plus a
// synthesis pass) is confirmed before sending — a tooltip alone isn't a guard against an
// unfiltered send over a large pool. No backend cap: the backend only enforces the floor
// (<2 papers raises), a slow turn is an inconvenience, not an incident, in a local tool.
const COMPARE_CONFIRM_THRESHOLD = 12;

interface Turn extends StoredTurn {
  streaming?: boolean;
  compareTotal?: number; // resolved scope size at send time — only set while streaming live
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
  // Side-by-side paper panel. `openPaperId` is the paper the panel currently shows — its own
  // cursor, deliberately decoupled from the citation trace so the picker/scrolling can drift
  // from what the chat cited. `activeCitation` remembers which turn's citation opened the
  // panel, driving prev/next stepping and the "back to citation" affordance. `panelHighlight`
  // is the passage to scroll to; a fresh object (identity) on each focus re-triggers the jump.
  const [openPaperId, setOpenPaperId] = useState<string | null>(null);
  const [panelHighlight, setPanelHighlight] = useState<HighlightTarget | null>(null);
  const [activeCitation, setActiveCitation] = useState<{ turnIndex: number; ref: string } | null>(
    null,
  );
  // Divider snap state. `snap` is the named preset; `dragRatio` (paper's fraction of the
  // split width, 0..1) is a live drag that overrides it until the next snap. `readWidth`
  // remembers the width the user last annotated at, so "Read" reopens at their reading width.
  const [snap, setSnap] = useState<"chat" | "split" | "read">("split");
  const [dragRatio, setDragRatio] = useState<number | null>(null);
  const [readWidth, setReadWidth] = useState(0.7);
  const splitRef = useRef<HTMLDivElement>(null);
  // Column refs + a live paper-fraction ref: the divider drag writes flex-grow straight to
  // these nodes each pointermove (no per-frame React render — the paper markdown is
  // expensive to re-render), committing to state only on release.
  const chatColRef = useRef<HTMLDivElement>(null);
  const paperColRef = useRef<HTMLDivElement>(null);
  const paperFracRef = useRef(0.55);
  // Desktop-only side-by-side; below this a citation click falls back to navigating to the
  // standalone paper route (the split can't breathe on a narrow screen).
  const [wide, setWide] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia("(min-width: 900px)").matches : true,
  );
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

  // Leaving a chat abandons only that chat's active turn. The id equality matters for a
  // brand-new chat: navigate() changes `/` to `/c/:id` just before runTurn installs its
  // controller, and the cleanup for `/` must not abort the request that belongs to :id.
  useEffect(
    () => () => {
      const controller = abortRef.current;
      const activeId = activeChatIdRef.current;
      if (!controller || controller.signal.aborted || activeId !== chatId) return;
      controller.abort();
      stopChat(activeId).catch((e) => console.error("Failed to stop generation", e));
    },
    [chatId],
  );

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
    // The open paper panel and its citation pointer belong to the conversation you were in —
    // a switch (or a fresh chat) makes them stale against the new turns, so close it.
    setOpenPaperId(null);
    setPanelHighlight(null);
    setActiveCitation(null);
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
        setTurns(s.turns);
        // Restore the conversation's locked retrieval scope so a reload keeps searching the
        // same papers (empty for chats predating scope persistence, i.e. the whole library).
        setTags(s.tags ?? []);
        setPapers(s.papers ?? []);
        // Neither toggle is a locked filter — each should reflect what the conversation's
        // latest message actually used, not silently reset to a default that may not
        // match (e.g. reloading a chat whose last message used per-paper mode would
        // otherwise show the toggle off, misleading the user about what's about to happen
        // if they send another message without checking).
        const last = s.turns[s.turns.length - 1];
        setPerPaper(last?.per_paper ?? false);
        const lastCompare = last?.compare ?? false;
        const lastAuto = last?.auto ?? false;
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
  }, [turns.length, lastTurn?.answer, lastTurn?.trace.length, lastTurn?.streaming]);

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

  async function onFeedback(turnIndex: number, vote: Feedback["vote"], note: string | null) {
    if (!chatId) return;
    patchAt(turnIndex, (t) => ({ ...t, feedback: { vote, note } }));
    try {
      await setFeedback(chatId, turnIndex, vote, note);
    } catch (e) {
      console.error("Failed to save feedback", e);
    }
  }

  // Shared by send()/sendEdit(): builds history from `prefix` + `question`, appends one
  // streaming exchange, and runs the chat() SSE call.
  // `prefix` must be captured fresh by the caller right before this call — it's applied
  // directly (not via a setTurns functional updater), so an await between capturing it
  // and calling runTurn could hand it a stale snapshot.
  async function runTurn(
    prefix: Turn[],
    question: string,
    id: string,
    editTurn: number | undefined,
    sendCompare: boolean,
    sendAuto: boolean,
  ) {
    const scopeSize = sendCompare ? resolveScopeSize(allPapers, tags, papers) : 0;
    const history: ChatMessage[] = [
      ...prefix.flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.answer },
      ]),
      { role: "user", content: question },
    ];
    setTurns([
      ...prefix,
      {
        question,
        answer: "",
        citations: [],
        trace: [],
        usage: null,
        feedback: null,
        per_paper: mode === "ask" ? perPaper : false,
        compare: sendCompare,
        compare_results: sendCompare ? [] : null,
        auto: sendAuto,
        streaming: true,
        compareTotal: sendCompare ? scopeSize : undefined,
      },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    activeChatIdRef.current = id;
    pendingSeparatorRef.current = false;
    const isActiveTurn = () => abortRef.current === controller && !controller.signal.aborted;
    const patchActiveTurn = (fn: (t: Turn) => Turn) => {
      if (isActiveTurn()) patchLast(fn);
    };
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
            patchActiveTurn((t) => {
              const sep = pendingSeparatorRef.current && t.answer && !/\n\n$/.test(t.answer);
              pendingSeparatorRef.current = false;
              return { ...t, answer: t.answer + (sep ? "\n\n" : "") + tok };
            }),
          onCitations: (c) => patchActiveTurn((t) => ({ ...t, citations: c })),
          onTrace: (e) => {
            if (!isActiveTurn()) return;
            pendingSeparatorRef.current = true;
            patchActiveTurn((t) => ({ ...t, trace: [...t.trace, e] }));
          },
          onUsage: (u) => patchActiveTurn((t) => ({ ...t, usage: u })),
          onCompareRow: (row) =>
            patchActiveTurn((t) => ({
              ...t,
              compare_results: [...(t.compare_results ?? []), row],
            })),
          onMeta: () => refreshSessions(),
          onError: (e) =>
            patchActiveTurn((t) => ({ ...t, answer: t.answer + `\n\n_Error: ${e}_` })),
          onDone: () => patchActiveTurn((t) => ({ ...t, streaming: false })),
        },
        editTurn,
        controller.signal,
        sendAuto,
      );
    } finally {
      // Only the still-active turn owns the shared UI. If a newer turn has taken over
      // (`abortRef` reassigned), this stale finally must touch nothing — clearing the
      // global `busy` here would report the newer turn as idle while it's mid-stream.
      if (abortRef.current === controller) {
        abortRef.current = null;
        activeChatIdRef.current = null;
        setBusy(false);
        refreshSessions();
      }
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
      ...turns.flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.answer },
      ]),
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
    if (turnsAfter > 0) {
      const exchanges = turnsAfter;
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
      ...prefix.flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.answer },
      ]),
      { role: "user", content: q },
    ];
    const resolved = await resolveSendMode(history);
    if (!resolved.send) return;

    setEditingIndex(null);
    // A resend from `index` discards that turn and everything after it, so a citation
    // pointer into the discarded range no longer identifies the same answer — drop it (the
    // panel's own paper stays open; only prev/next context is cleared).
    if (activeCitation && activeCitation.turnIndex >= index) setActiveCitation(null);
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
  // The chat's active retrieval scope, for marking papers the picker can open but this
  // conversation won't search. `null` = no filter set (everything is in scope).
  const scopeIds = resolveScopeIds(allPapers, tags, papers);
  const scopeSet = scopeIds ? new Set(scopeIds) : null;

  // ---- Side-by-side paper panel ----
  const panelOpen = openPaperId !== null;

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 900px)");
    const onChange = () => setWide(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // The turn the panel was opened from, and its cited passages in order — the set prev/next
  // walks. Recomputed from the turn each render so an edit/resend that changes a turn's
  // citations is reflected (the pointer is by ref; it simply lands in the new set or drops).
  const activeTurn = activeCitation ? turns[activeCitation.turnIndex] : undefined;
  const activeCited = activeTurn ? citedCitations(activeTurn.answer, activeTurn.citations) : [];
  const activeIdx = activeCitation
    ? activeCited.findIndex((c) => c.ref === activeCitation.ref)
    : -1;
  const citedTarget = activeIdx >= 0 ? activeCited[activeIdx] : undefined;
  // Panel drifted off the cited paper (picker, or opening a card) — offer one click back.
  // Paper-level drift is what we can detect; scroll-away within a paper isn't tracked.
  const offCitation = !!citedTarget && openPaperId !== citedTarget.paper_id;
  // A newer finished turn cited papers after the one the panel is anchored to — a quiet,
  // opt-in nudge rather than yanking the panel away from what the user is reading.
  const lastIdx = turns.length - 1;
  const lastCited =
    lastIdx >= 0 && !turns[lastIdx].streaming
      ? citedCitations(turns[lastIdx].answer, turns[lastIdx].citations)
      : [];
  const hasNewerCitations =
    panelOpen && !!activeCitation && lastIdx > activeCitation.turnIndex && lastCited.length > 0;

  // paper's fraction of the split width for the current snap (or live committed drag).
  const paperFrac = dragRatio ?? (snap === "chat" ? 0 : snap === "read" ? readWidth : 0.55);
  // Keep the live ref in sync with the committed fraction (the drag writes it directly, and
  // `paperFrac` is stable through a drag, so this never fires mid-drag to clobber it).
  useEffect(() => {
    paperFracRef.current = paperFrac;
  }, [paperFrac]);

  function focusPaper(paperId: string, snippet?: string, section?: string) {
    setOpenPaperId(paperId);
    // A fresh object each call — its identity change is what re-triggers the panel's
    // scroll-and-highlight effect, even for the same snippet.
    setPanelHighlight(snippet ? { snippet, section } : null);
  }

  function openCitation(
    target: { paperId: string; ref?: string; snippet?: string; section?: string },
    turnIndex: number,
  ) {
    focusPaper(target.paperId, target.snippet, target.section);
    setActiveCitation(target.ref ? { turnIndex, ref: target.ref } : null);
    setSnap("split");
    setDragRatio(null);
  }

  function stepCitation(delta: number) {
    if (!activeCitation || activeIdx < 0) return;
    const next = activeCited[activeIdx + delta];
    if (!next) return;
    focusPaper(next.paper_id, next.snippet, next.section_title);
    setActiveCitation({ turnIndex: activeCitation.turnIndex, ref: next.ref });
  }

  function backToCitation() {
    if (!citedTarget) return;
    focusPaper(citedTarget.paper_id, citedTarget.snippet, citedTarget.section_title);
  }

  function closePanel() {
    setOpenPaperId(null);
    setPanelHighlight(null);
    setActiveCitation(null);
  }

  function snapTo(s: "chat" | "split" | "read") {
    setSnap(s);
    setDragRatio(null);
  }

  // "Read" reopens at the width the user last annotated at (only remember genuine reading
  // widths, not a sliver where a stray selection happened). Stable identity — it's passed to
  // the memoized PaperPanel; reads the live fraction from a ref instead of render scope.
  const rememberReadWidth = useCallback(() => {
    if (paperFracRef.current >= 0.4) setReadWidth(paperFracRef.current);
  }, []);

  // Apply a paper fraction straight to the two column nodes — the drag's hot path, kept off
  // React so a resize doesn't re-render the conversation or re-parse the paper each frame.
  function applyFrac(frac: number) {
    paperFracRef.current = frac;
    if (chatColRef.current) chatColRef.current.style.flexGrow = String(1 - frac);
    if (paperColRef.current) paperColRef.current.style.flexGrow = String(frac);
  }

  function startDrag(e: React.PointerEvent) {
    e.preventDefault();
    const container = splitRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    let latest = paperFracRef.current;
    function onMove(ev: PointerEvent) {
      const fromLeft = (ev.clientX - rect.left) / rect.width;
      latest = Math.min(0.92, Math.max(0.08, 1 - fromLeft));
      applyFrac(latest);
    }
    function onUp() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      setDragRatio(latest); // commit once, so subsequent renders agree with the DOM
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  // Snap shortcuts: [ widen chat · \ split · ] widen paper · Esc close. Ignored while typing.
  useEffect(() => {
    if (!panelOpen) return;
    function onKey(e: KeyboardEvent) {
      const el = document.activeElement as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable))
        return;
      if (e.key === "[") snapTo("chat");
      else if (e.key === "\\") snapTo("split");
      else if (e.key === "]") snapTo("read");
      else if (e.key === "Escape") closePanel();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [panelOpen]);

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
    <div style={{ maxWidth: panelOpen ? "none" : 1180, margin: "0 auto" }}>
      <Group
        ref={splitRef}
        align="stretch"
        gap={panelOpen ? 0 : "lg"}
        wrap="nowrap"
        style={
          panelOpen
            ? { height: "calc(100vh - 60px - 3rem)", overflow: "hidden" }
            : { minHeight: "calc(100vh - 60px - 3rem)" }
        }
      >
        {/* Session list collapses out of the way while the paper panel is open. */}
        {!panelOpen && sidebarOpen && (
          <ChatSidebar
            sessions={sessions}
            activeId={chatId}
            onNew={newChat}
            onSelect={(id) => navigate(`/c/${id}`)}
            onDelete={onDelete}
          />
        )}
        <Stack
          ref={chatColRef}
          gap="md"
          style={{
            flexGrow: panelOpen ? 1 - paperFrac : 1,
            flexBasis: 0,
            minWidth: 0,
            ...(panelOpen ? { height: "100%", overflowY: "auto" } : {}),
          }}
        >
          <Group gap="xs" justify="space-between">
            <Group gap="xs">
              <Tooltip label={sidebarOpen ? "Hide chats" : "Show chats"}>
                <ActionIcon variant="subtle" color="gray" onClick={() => setSidebarOpen((o) => !o)}>
                  <IconSidebar size={18} />
                </ActionIcon>
              </Tooltip>
              <Tooltip label="Export conversation as Markdown">
                {/* Disabled ActionIcon drops pointer events; wrap in a span so the
                    tooltip still fires on hover (same workaround as AnswerActions). */}
                <span>
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    aria-label="Export conversation as Markdown"
                    disabled={turns.length === 0 || busy}
                    onClick={() => {
                      const name = sessions.find((s) => s.id === chatId)?.name ?? "Chat";
                      downloadTextFile(
                        `${slugFilename(name, chatId ?? "chat")}.md`,
                        conversationToMarkdown(turns, name),
                      ).catch((e) => console.error("Export failed", e));
                    }}
                  >
                    <IconDownload size={18} />
                  </ActionIcon>
                </span>
              </Tooltip>
            </Group>
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
              {turns.map((t, i) => (
                <Box key={i}>
                  <Group justify="flex-end" align="center" gap={4}>
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
                          onClick={() => startEdit(i, t.question)}
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
                          <Text style={{ whiteSpace: "pre-wrap" }}>{t.question}</Text>
                        </Box>
                      </>
                    )}
                  </Group>
                  <Box mt="sm">
                    {t.auto && (
                      <Badge size="xs" variant="outline" color="gray" mb={4}>
                        Auto
                      </Badge>
                    )}
                    {t.compare ? (
                      <ComparePanel
                        rows={t.compare_results ?? []}
                        totalPapers={t.compareTotal ?? t.compare_results?.length ?? 0}
                        streaming={t.streaming}
                        // The synthesis pass's own text streams through the same `answer`
                        // field a normal answer uses — the moment any of it has arrived,
                        // every per-paper sub-run is done and synthesis has started.
                        synthesizing={t.streaming && !!t.answer}
                      />
                    ) : (
                      <TraceBox entries={t.trace} streaming={t.streaming} />
                    )}
                    {t.answer ? (
                      <Answer
                        text={t.answer}
                        citations={t.citations}
                        onOpenCitation={wide ? (target) => openCitation(target, i) : undefined}
                      />
                    ) : t.streaming ? (
                      <Group gap="xs">
                        <Loader size="sm" type="dots" color="accent" />
                        <Text size="sm" c="dimmed">
                          Thinking…
                        </Text>
                      </Group>
                    ) : null}
                    {!t.streaming && (
                      <SourceCards
                        citations={citedCitations(t.answer, t.citations)}
                        onOpenCitation={wide ? (target) => openCitation(target, i) : undefined}
                      />
                    )}
                    {!t.streaming && t.usage && (
                      <Text size="xs" c="dimmed" mt={4}>
                        {formatUsage(t.usage)}
                      </Text>
                    )}
                    {!t.streaming && t.answer && chatId && (
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
                    {!t.streaming && t.answer && (
                      <AnswerActions text={t.answer} citations={t.citations} />
                    )}
                  </Box>
                </Box>
              ))}
              <div ref={bottomRef} />
            </Stack>
          )}

          <Box style={{ position: "sticky", bottom: 0, paddingBottom: 4 }}>{composer}</Box>
        </Stack>
        {panelOpen && openPaperId && (
          <>
            {/* Divider: a hairline splitter with the controls floating on it, so they stay
              reachable at every snap (even "chat", where the paper column is 0-width). Drag
              anywhere on the rail to resize; the controls stop the drag from starting. */}
            <Box
              className="split-rail"
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize the paper panel"
              onPointerDown={startDrag}
              style={{
                width: 38,
                flexShrink: 0,
                position: "relative",
                userSelect: "none",
              }}
            >
              <Box
                className="split-rail-line"
                style={{
                  position: "absolute",
                  top: 0,
                  bottom: 0,
                  left: "50%",
                  width: 1,
                  transform: "translateX(-50%)",
                }}
              />
              <Box
                onPointerDown={(e) => e.stopPropagation()}
                style={{
                  position: "absolute",
                  top: 6,
                  left: "50%",
                  transform: "translateX(-50%)",
                  background: "var(--mantine-color-body)",
                  padding: "2px 0",
                }}
              >
                <Tooltip label="Close paper (Esc)" position="left">
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    size="sm"
                    aria-label="Close paper"
                    onClick={closePanel}
                  >
                    <IconX size={15} />
                  </ActionIcon>
                </Tooltip>
              </Box>
              <Box
                onPointerDown={(e) => e.stopPropagation()}
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                  padding: 3,
                  background: "var(--mantine-color-body)",
                  border: "1px solid var(--pl-border)",
                  borderRadius: "var(--mantine-radius-xl)",
                }}
              >
                {(
                  [
                    ["read", <IconPanelLeft size={15} />, "Widen paper ( ] )"],
                    ["split", <IconPanelCenter size={15} />, "Split evenly ( \\ )"],
                    ["chat", <IconPanelRight size={15} />, "Widen chat ( [ )"],
                  ] as const
                ).map(([value, icon, label]) => {
                  const active = snap === value && dragRatio === null;
                  return (
                    <Tooltip key={value} label={label} position="left">
                      <ActionIcon
                        variant={active ? "light" : "subtle"}
                        color={active ? "accent" : "gray"}
                        size="sm"
                        radius="xl"
                        aria-label={label.split(" (")[0]}
                        onClick={() => snapTo(value)}
                      >
                        {icon}
                      </ActionIcon>
                    </Tooltip>
                  );
                })}
              </Box>
            </Box>
            <Box
              ref={paperColRef}
              style={{
                flexGrow: paperFrac,
                flexBasis: 0,
                minWidth: 0,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
              }}
            >
              <Group
                gap={6}
                wrap="nowrap"
                px="sm"
                py={7}
                style={{ flexShrink: 0, borderBottom: "1px solid var(--pl-border)" }}
              >
                <Group gap={2} wrap="nowrap" style={{ flexShrink: 0 }}>
                  <Tooltip label="Previous citation">
                    <ActionIcon
                      variant="subtle"
                      color="gray"
                      size="sm"
                      aria-label="Previous citation"
                      disabled={activeIdx <= 0}
                      onClick={() => stepCitation(-1)}
                    >
                      <IconChevron size={15} style={{ transform: "rotate(180deg)" }} />
                    </ActionIcon>
                  </Tooltip>
                  <Text size="xs" c="dimmed" className="tnum" ta="center" style={{ minWidth: 26 }}>
                    {activeIdx >= 0 ? `${activeIdx + 1}/${activeCited.length}` : "—"}
                  </Text>
                  <Tooltip label="Next citation">
                    <ActionIcon
                      variant="subtle"
                      color="gray"
                      size="sm"
                      aria-label="Next citation"
                      disabled={activeIdx < 0 || activeIdx >= activeCited.length - 1}
                      onClick={() => stepCitation(1)}
                    >
                      <IconChevron size={15} />
                    </ActionIcon>
                  </Tooltip>
                </Group>
                <Select
                  data={paperOptions}
                  value={openPaperId}
                  onChange={(pid) => pid && focusPaper(pid)}
                  size="xs"
                  variant="filled"
                  searchable
                  allowDeselect={false}
                  aria-label="Open a paper"
                  style={{ flex: 1, minWidth: 0 }}
                  // Mark papers this conversation's filter excludes: openable for reading,
                  // but not part of what the chat searches. Only when a filter is active.
                  renderOption={
                    scopeSet
                      ? ({ option }) => {
                          const outOfScope = !scopeSet.has(option.value);
                          return (
                            <Group
                              gap="xs"
                              justify="space-between"
                              wrap="nowrap"
                              style={{ flex: 1, minWidth: 0 }}
                            >
                              <Text size="xs" lineClamp={1} c={outOfScope ? "dimmed" : undefined}>
                                {option.label}
                              </Text>
                              {outOfScope && (
                                <Text size="10px" c="dimmed" style={{ flexShrink: 0 }}>
                                  not in this chat
                                </Text>
                              )}
                            </Group>
                          );
                        }
                      : undefined
                  }
                />
                {offCitation && (
                  <Tooltip label="Back to the cited passage">
                    <Button
                      size="compact-xs"
                      variant="light"
                      color="accent"
                      leftSection={
                        <IconChevron size={12} style={{ transform: "rotate(180deg)" }} />
                      }
                      onClick={backToCitation}
                      className="tnum"
                      style={{ flexShrink: 0 }}
                    >
                      {activeIdx + 1}/{activeCited.length}
                    </Button>
                  </Tooltip>
                )}
                {hasNewerCitations && (
                  <Tooltip label="Jump to the latest answer's citations">
                    <Button
                      size="compact-xs"
                      variant="light"
                      color="accent"
                      onClick={() =>
                        openCitation(
                          {
                            paperId: lastCited[0].paper_id,
                            ref: lastCited[0].ref,
                            snippet: lastCited[0].snippet,
                            section: lastCited[0].section_title,
                          },
                          lastIdx,
                        )
                      }
                      style={{ flexShrink: 0 }}
                    >
                      {lastCited.length} new
                    </Button>
                  </Tooltip>
                )}
              </Group>
              <Box style={{ flex: 1, minHeight: 0 }}>
                <PaperPanel
                  key={openPaperId}
                  paperId={openPaperId}
                  variant="panel"
                  highlight={panelHighlight}
                  onNoteSaved={rememberReadWidth}
                />
              </Box>
            </Box>
          </>
        )}
      </Group>
    </div>
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
