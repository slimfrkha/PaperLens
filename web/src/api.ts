export interface Paper {
  paper_id: string;
  title: string;
  tags: string[];
  arxiv_id?: string;
  n_chunks?: number;
}

export interface TagCount {
  tag: string;
  count: number;
}

export interface Citation {
  ref: string;
  paper_id: string;
  title: string;
  breadcrumb: string;
  section_title: string;
  snippet: string;
}

export interface AdminStatus {
  db: { n_papers: number; n_chunks: number };
  tags: TagCount[];
  pending: string[];
  ingestion: {
    state: "idle" | "running" | "error";
    total: number;
    done: number;
    current: { name: string; stage: string; pct: number } | null;
    errors: { name: string; error: string }[];
  };
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatSummary {
  id: string;
  name: string;
  updated_at: string;
}

export interface ChatSession {
  id: string;
  name: string;
  messages: { role: "user" | "assistant"; content: string }[];
  citations: (Citation[] | null)[];
  traces?: (TraceEntry[] | null)[];
}

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const getPapers = () => fetch("/api/papers").then(j<Paper[]>);
export const getPaper = (id: string) =>
  fetch(`/api/papers/${encodeURIComponent(id)}`).then(
    j<{ paper_id: string; title: string; tags: string[]; arxiv_id?: string; markdown: string }>
  );
export const getTags = () => fetch("/api/tags").then(j<TagCount[]>);
export const getStatus = () => fetch("/api/admin/status").then(j<AdminStatus>);
export const rescan = () =>
  fetch("/api/admin/rescan", { method: "POST" }).then(j<{ started: boolean }>);

export const listChats = () => fetch("/api/chats").then(j<ChatSummary[]>);
export const getChat = (id: string) =>
  fetch(`/api/chats/${encodeURIComponent(id)}`).then(j<ChatSession>);
export const createChat = () => fetch("/api/chats", { method: "POST" }).then(j<ChatSession>);
export const deleteChat = (id: string) =>
  fetch(`/api/chats/${encodeURIComponent(id)}`, { method: "DELETE" }).then((r) => r.json());

export interface TraceEntry {
  type: "thought" | "action" | "observation";
  text?: string;
  query?: string;
  paper?: string | null;
}

export interface ChatHandlers {
  onToken: (t: string) => void;
  onCitations: (c: Citation[]) => void;
  onTrace?: (e: TraceEntry) => void;
  onMeta?: (m: { chat_id: string; name: string }) => void;
  onError?: (e: string) => void;
  onDone?: () => void;
}

/** POST /api/chat and parse the SSE stream (token / citations / meta / error / done). */
export async function chat(
  messages: ChatMessage[],
  tags: string[],
  papers: string[],
  chatId: string | null,
  h: ChatHandlers
): Promise<void> {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, tags, papers, chat_id: chatId }),
  });
  if (!resp.body) throw new Error("no response body");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  const dispatch = (event: string, data: string) => {
    if (event === "token") h.onToken(data);
    else if (event === "citations") h.onCitations(JSON.parse(data) as Citation[]);
    else if (event === "trace") h.onTrace?.(JSON.parse(data) as TraceEntry);
    else if (event === "meta") h.onMeta?.(JSON.parse(data));
    else if (event === "error") h.onError?.(data);
    else if (event === "done") h.onDone?.();
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // Normalize CRLF (sse-starlette frames with \r\n) so framing is consistent.
    buf = (buf + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";
    for (const frame of frames) {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      }
      dispatch(event, dataLines.join("\n"));
    }
  }
}
