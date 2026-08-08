import { afterEach, describe, expect, it, vi } from "vitest";
import { addPaper, chat, removePaper, setFeedback } from "./api";

function mockStreamResponse(sse: string): Response {
  let sent = false;
  return {
    body: {
      getReader() {
        return {
          async read() {
            if (!sent) {
              sent = true;
              return { done: false, value: new TextEncoder().encode(sse) };
            }
            return { done: true, value: undefined };
          },
        };
      },
    },
  } as unknown as Response;
}

describe("chat", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses token, citations, and done SSE events out of the stream", async () => {
    const sse =
      "event: token\ndata: Hello\n\n" +
      'event: citations\ndata: [{"ref":"1","paper_id":"p1","title":"T","breadcrumb":"b","section_title":"s","snippet":"sn"}]\n\n' +
      "event: done\ndata: \n\n";
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockStreamResponse(sse)));

    const onToken = vi.fn();
    const onCitations = vi.fn();
    const onDone = vi.fn();

    await chat([], [], [], false, null, { onToken, onCitations, onDone });

    expect(onToken).toHaveBeenCalledWith("Hello");
    expect(onCitations).toHaveBeenCalledWith([
      expect.objectContaining({ ref: "1", paper_id: "p1" }),
    ]);
    expect(onDone).toHaveBeenCalledOnce();
  });

  it("parses the usage SSE event", async () => {
    const sse = 'event: usage\ndata: {"input_tokens":100,"output_tokens":20,"latency_ms":1500}\n\n';
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockStreamResponse(sse)));

    const onUsage = vi.fn();
    await chat([], [], [], false, null, { onToken: vi.fn(), onCitations: vi.fn(), onUsage });

    expect(onUsage).toHaveBeenCalledWith({
      input_tokens: 100,
      output_tokens: 20,
      latency_ms: 1500,
    });
  });

  it("normalizes CRLF frame separators from the sse-starlette server", async () => {
    const sse = "event: token\r\ndata: Hi\r\n\r\n";
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockStreamResponse(sse)));

    const onToken = vi.fn();
    await chat([], [], [], false, null, { onToken, onCitations: vi.fn() });

    expect(onToken).toHaveBeenCalledWith("Hi");
  });

  it("sends edit_index in the request body when editing a prior turn", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], false, "c1", { onToken: vi.fn(), onCitations: vi.fn() }, 2);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: false,
          chat_id: "c1",
          edit_index: 2,
        }),
      }),
    );
  });

  it("sends edit_index: null for a normal (non-edit) send", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], false, "c1", { onToken: vi.fn(), onCitations: vi.fn() });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: false,
          chat_id: "c1",
          edit_index: null,
        }),
      }),
    );
  });

  it("sends per_paper: true in the request body when passed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], true, "c1", { onToken: vi.fn(), onCitations: vi.fn() });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: true,
          chat_id: "c1",
          edit_index: null,
        }),
      }),
    );
  });

  it("surfaces a 409 (turn already in progress) via onError + onDone without reading a body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        status: 409,
        json: () => Promise.resolve({ error: "a turn is already in progress for this chat" }),
      } as Response),
    );

    const onError = vi.fn();
    const onDone = vi.fn();
    await chat([], [], [], false, "c1", {
      onToken: vi.fn(),
      onCitations: vi.fn(),
      onError,
      onDone,
    });

    expect(onError).toHaveBeenCalledWith("a turn is already in progress for this chat");
    expect(onDone).toHaveBeenCalledOnce();
  });
});

describe("setFeedback", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs index/vote/note to the chat's feedback route", async () => {
    const session = { id: "c1", name: "T", messages: [], citations: [] };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(session),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const result = await setFeedback("c1", 1, "up", "great citation");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chats/c1/feedback",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: 1, vote: "up", note: "great citation" }),
      }),
    );
    expect(result).toEqual(session);
  });
});

describe("addPaper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs the raw id/URL and returns the queued name", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ queued: true, name: "2412.19437" }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const result = await addPaper("https://arxiv.org/abs/2412.19437");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/papers",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id_or_url: "https://arxiv.org/abs/2412.19437" }),
      }),
    );
    expect(result).toEqual({ queued: true, name: "2412.19437" });
  });

  it("throws the backend's error message on a 409 duplicate", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 409,
        statusText: "Conflict",
        json: () => Promise.resolve({ error: "already curated as deepseek-v3" }),
      } as Response),
    );

    await expect(addPaper("2412.19437")).rejects.toThrow("already curated as deepseek-v3");
  });
});

describe("removePaper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("DELETEs the paper and tolerates a bodyless 204", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 204 } as Response);
    vi.stubGlobal("fetch", fetchMock);

    await expect(removePaper("paper-a")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/papers/paper-a", { method: "DELETE" });
  });

  it("throws the backend's error message on a 404", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: () => Promise.resolve({ error: "not found" }),
      } as Response),
    );

    await expect(removePaper("missing")).rejects.toThrow("not found");
  });
});
