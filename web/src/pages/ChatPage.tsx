import { useEffect, useRef, useState } from "react";
import {
  ActionIcon,
  Alert,
  Box,
  Button,
  Flex,
  Group,
  Loader,
  MultiSelect,
  Paper,
  Stack,
  Text,
  Textarea,
  Tooltip,
} from "@mantine/core";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  chat,
  createChat,
  deleteChat,
  getChat,
  getPapers,
  getTags,
  listChats,
  type ChatMessage,
  type ChatSummary,
  type Citation,
  type TagCount,
  type TraceEntry,
} from "../api";
import Answer from "../components/Answer";
import ChatSidebar from "../components/ChatSidebar";
import TraceBox from "../components/TraceBox";

interface Turn {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  trace?: TraceEntry[];
  streaming?: boolean;
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
  const [busy, setBusy] = useState(false);
  const [empty, setEmpty] = useState(false);
  const loadedId = useRef<string | null>(null); // which chat's turns are in state
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshSessions = () => listChats().then(setSessions);

  useEffect(() => {
    getTags().then((t: TagCount[]) => setTagOptions(t.map((x) => x.tag)));
    getPapers().then((p) => setEmpty(p.length === 0));
    refreshSessions();
  }, []);

  // Load the session named in the URL (restores conversation after navigation).
  useEffect(() => {
    if (!chatId) {
      setTurns([]);
      loadedId.current = null;
      return;
    }
    if (loadedId.current === chatId) return; // already have it (e.g. just created)
    getChat(chatId)
      .then((s) => {
        setTurns(
          s.messages.map((m, i) => ({
            role: m.role,
            content: m.content,
            citations: s.citations?.[i] ?? undefined,
            trace: s.traces?.[i] ?? undefined,
          }))
        );
        loadedId.current = chatId;
      })
      .catch(() => setTurns([]));
  }, [chatId]);

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [turns]);

  const patchLast = (fn: (t: Turn) => Turn) =>
    setTurns((prev) => {
      const next = [...prev];
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");

    let id = chatId ?? null;
    if (!id) {
      const c = await createChat();
      id = c.id;
      loadedId.current = id; // prevent the load effect from clobbering our turns
      navigate(`/c/${id}`, { replace: true });
    }

    const history: ChatMessage[] = [
      ...turns.map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: q },
    ];
    setTurns((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);
    try {
      await chat(history, tags, null, id, {
        onToken: (tok) => patchLast((t) => ({ ...t, content: t.content + tok })),
        onCitations: (c) => patchLast((t) => ({ ...t, citations: c })),
        onTrace: (e) => patchLast((t) => ({ ...t, trace: [...(t.trace ?? []), e] })),
        onMeta: () => refreshSessions(),
        onError: (e) => patchLast((t) => ({ ...t, content: t.content + `\n\n_Error: ${e}_` })),
        onDone: () => patchLast((t) => ({ ...t, streaming: false })),
      });
    } finally {
      setBusy(false);
      refreshSessions();
    }
  }

  async function onDelete(id: string) {
    await deleteChat(id);
    await refreshSessions();
    if (id === chatId) navigate("/");
  }

  return (
    <Flex gap="md" align="stretch">
      {sidebarOpen && (
        <ChatSidebar
          sessions={sessions}
          activeId={chatId}
          onNew={() => navigate("/")}
          onSelect={(id) => navigate(`/c/${id}`)}
          onDelete={onDelete}
        />
      )}
      <Box style={{ flex: 1, minWidth: 0 }}>
        <Stack>
          <Group gap="xs">
            <Tooltip label={sidebarOpen ? "Hide chats" : "Show chats"}>
              <ActionIcon variant="subtle" onClick={() => setSidebarOpen((o) => !o)}>
                ☰
              </ActionIcon>
            </Tooltip>
            <Button size="xs" variant="light" onClick={() => navigate("/")}>
              + New chat
            </Button>
          </Group>

          {empty && (
            <Alert color="yellow" title="The library is empty">
              No papers indexed yet — ingestion may still be running. Check the{" "}
              <Link to="/admin">Admin</Link> page.
            </Alert>
          )}

          <MultiSelect
            label="Restrict search to tags (optional)"
            data={tagOptions}
            value={tags}
            onChange={setTags}
            placeholder="All papers"
            searchable
            clearable
          />

          <Stack gap="lg">
            {turns.map((t, i) => (
              <Paper
                key={i}
                p="md"
                withBorder
                radius="md"
                bg={t.role === "user" ? "var(--mantine-color-blue-light)" : undefined}
              >
                <Text size="xs" c="dimmed" mb={4}>
                  {t.role === "user" ? "You" : "Assistant"}
                </Text>
                {t.role === "user" ? (
                  <Text style={{ whiteSpace: "pre-wrap" }}>{t.content}</Text>
                ) : (
                  <>
                    {t.trace && <TraceBox entries={t.trace} streaming={t.streaming} />}
                    {t.content ? (
                      <Answer text={t.content} citations={t.citations ?? []} />
                    ) : t.streaming ? (
                      <Loader size="sm" />
                    ) : null}
                  </>
                )}
              </Paper>
            ))}
            <div ref={bottomRef} />
          </Stack>

          <Group align="flex-end">
            <Textarea
              flex={1}
              autosize
              minRows={1}
              maxRows={6}
              value={input}
              placeholder="Ask about a paper or a concept…"
              onChange={(e) => setInput(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <Button onClick={send} loading={busy}>
              Send
            </Button>
          </Group>
        </Stack>
      </Box>
    </Flex>
  );
}
