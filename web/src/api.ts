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

export type FaithfulnessLabel = "entailment" | "neutral" | "contradiction";

export interface FaithfulnessClaim {
  sentence: string;
  label: FaithfulnessLabel;
  score: number;
}

export type RetrievalSource = "dense" | "sparse" | "both";

export interface Citation {
  ref: string;
  paper_id: string;
  title: string;
  arxiv_id?: string | null;
  breadcrumb: string;
  section_title: string;
  section_number?: string;
  source?: RetrievalSource;
  snippet: string;
  body?: string;
  faithfulness?: FaithfulnessClaim[];
}

export interface Annotation {
  id: string;
  snippet: string;
  section_title: string;
  section_slug: string;
  note: string;
  created_at: string;
  updated_at: string;
}

/** An Annotation joined with the paper it belongs to — the shape `/api/annotations`
 *  (every annotation across the whole library) returns, for the cross-paper Notes page. */
export interface LibraryAnnotation extends Annotation {
  paper_id: string;
  paper_title: string;
  arxiv_id?: string;
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

export interface Feedback {
  vote: "up" | "down" | null;
  note: string | null;
}

export interface UsageInfo {
  input_tokens: number | null;
  output_tokens: number | null;
  latency_ms: number;
}

/** One paper's own dedicated search+answer sub-run within a Compare turn — the
 *  carousel's drill-down data. `text`/`citations` render through the same `Answer`
 *  component a normal turn's answer does, so its own citation links and faithfulness
 *  flags work exactly like a normal answer's. */
export interface CompareRow {
  paper_id: string;
  title: string;
  arxiv_id?: string | null;
  text: string;
  citations: Citation[];
  trace: TraceEntry[];
}

export interface ChatSession {
  id: string;
  name: string;
  messages: { role: "user" | "assistant"; content: string }[];
  citations: (Citation[] | null)[];
  traces?: (TraceEntry[] | null)[];
  feedback?: (Feedback | null)[];
  usage?: (UsageInfo | null)[];
  per_paper?: (boolean | null)[];
  compare?: (boolean | null)[];
  compare_results?: (CompareRow[] | null)[];
}

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

/** Like j<T>, but on failure surfaces the backend's `{"error": "..."}` body (e.g. the
 *  admin add/remove-paper routes' "already curated as X" / "not found") instead of a
 *  bare status line, and tolerates a bodyless 204 on success. */
async function jOrError<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = (await r.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `${r.status} ${r.statusText}`);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}

export const getPapers = () => fetch("/api/papers").then(j<Paper[]>);
export const getPaper = (id: string) =>
  fetch(`/api/papers/${encodeURIComponent(id)}`).then(
    j<{ paper_id: string; title: string; tags: string[]; arxiv_id?: string; markdown: string }>,
  );
export const getAnnotations = (paperId: string) =>
  fetch(`/api/papers/${encodeURIComponent(paperId)}/annotations`).then(j<Annotation[]>);
export const getAllAnnotations = () => fetch("/api/annotations").then(j<LibraryAnnotation[]>);
export const createAnnotation = (
  paperId: string,
  snippet: string,
  sectionTitle: string,
  sectionSlug: string,
  note: string,
) =>
  fetch(`/api/papers/${encodeURIComponent(paperId)}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      snippet,
      section_title: sectionTitle,
      section_slug: sectionSlug,
      note,
    }),
  }).then(j<Annotation>);
export const updateAnnotation = (paperId: string, annotationId: string, note: string) =>
  fetch(
    `/api/papers/${encodeURIComponent(paperId)}/annotations/${encodeURIComponent(annotationId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note }),
    },
  ).then(j<Annotation>);
export const deleteAnnotation = (paperId: string, annotationId: string) =>
  fetch(
    `/api/papers/${encodeURIComponent(paperId)}/annotations/${encodeURIComponent(annotationId)}`,
    { method: "DELETE" },
  ).then(j<{ ok: boolean }>);

export const getTags = () => fetch("/api/tags").then(j<TagCount[]>);
export const getStatus = () => fetch("/api/admin/status").then(j<AdminStatus>);
export const rescan = () =>
  fetch("/api/admin/rescan", { method: "POST" }).then(j<{ started: boolean }>);
export interface AddPaperResult {
  input: string;
  status: "queued" | "duplicate" | "invalid" | "error";
  name?: string;
  existing_name?: string;
  detail?: string; // present for "error" — the failed-write message
}
export const addPapers = (lines: string[]) =>
  fetch("/api/admin/papers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arxiv_ids_or_urls: lines }),
  }).then(jOrError<{ results: AddPaperResult[] }>);
export const removePaper = (paperId: string) =>
  fetch(`/api/admin/papers/${encodeURIComponent(paperId)}`, { method: "DELETE" }).then(
    jOrError<void>,
  );

export const listChats = () => fetch("/api/chats").then(j<ChatSummary[]>);
export const getChat = (id: string) =>
  fetch(`/api/chats/${encodeURIComponent(id)}`).then(j<ChatSession>);
export const createChat = () => fetch("/api/chats", { method: "POST" }).then(j<ChatSession>);
export const deleteChat = (id: string) =>
  fetch(`/api/chats/${encodeURIComponent(id)}`, { method: "DELETE" }).then((r) => r.json());
export const setFeedback = (
  chatId: string,
  index: number,
  vote: "up" | "down" | null,
  note: string | null,
) =>
  fetch(`/api/chats/${encodeURIComponent(chatId)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index, vote, note }),
  }).then(j<ChatSession>);

export interface TraceEntry {
  type: "thought" | "action" | "observation";
  text?: string;
  query?: string;
  paper?: string | null;
  per_paper?: boolean;
}

export interface ChatHandlers {
  onToken: (t: string) => void;
  onCitations: (c: Citation[]) => void;
  onTrace?: (e: TraceEntry) => void;
  onMeta?: (m: { chat_id: string; name: string }) => void;
  onUsage?: (u: UsageInfo) => void;
  onCompareRow?: (r: CompareRow) => void;
  onError?: (e: string) => void;
  onDone?: () => void;
}

const isAbortError = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

/** POST /api/chat and parse the SSE stream (token / citations / trace / usage /
 *  compare_row / meta / error / done). `compare_row` only fires on a Compare turn, once
 *  per completed paper — the carousel's drill-down data filling in progressively.
 *  `editIndex` truncates the stored chat back to that user turn before resuming from it.
 *  `signal`, if given, aborts the request (e.g. the chat page's Stop button) — an abort
 *  ends the stream silently (no onError/onDone) rather than throwing, since the caller
 *  already knows it stopped things itself. */
export async function chat(
  messages: ChatMessage[],
  tags: string[],
  papers: string[],
  perPaper: boolean,
  compare: boolean,
  chatId: string | null,
  h: ChatHandlers,
  editIndex?: number,
  signal?: AbortSignal,
): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        tags,
        papers,
        per_paper: perPaper,
        compare,
        chat_id: chatId,
        edit_index: editIndex ?? null,
      }),
      signal,
    });
  } catch (e) {
    if (isAbortError(e)) return;
    throw e;
  }
  if (resp.status === 409) {
    const body = await resp.json().catch(() => ({}));
    h.onError?.(body.error ?? "a turn is already in progress for this chat");
    h.onDone?.();
    return;
  }
  if (!resp.body) throw new Error("no response body");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  const dispatch = (event: string, data: string) => {
    if (event === "token") h.onToken(data);
    else if (event === "citations") h.onCitations(JSON.parse(data) as Citation[]);
    else if (event === "trace") h.onTrace?.(JSON.parse(data) as TraceEntry);
    else if (event === "usage") h.onUsage?.(JSON.parse(data) as UsageInfo);
    else if (event === "compare_row") h.onCompareRow?.(JSON.parse(data) as CompareRow);
    else if (event === "meta") h.onMeta?.(JSON.parse(data));
    else if (event === "error") h.onError?.(data);
    else if (event === "done") h.onDone?.();
  };

  while (true) {
    let done: boolean;
    let value: Uint8Array | undefined;
    try {
      ({ done, value } = await reader.read());
    } catch (e) {
      if (isAbortError(e)) return;
      throw e;
    }
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

/** POST /api/chats/{chatId}/stop — signal the in-flight turn (if any) to stop generating
 *  at its next checkpoint. `stopped: false` just means nothing was running; not an error. */
export const stopChat = (chatId: string) =>
  fetch(`/api/chats/${encodeURIComponent(chatId)}/stop`, { method: "POST" }).then(
    j<{ stopped: boolean }>,
  );
