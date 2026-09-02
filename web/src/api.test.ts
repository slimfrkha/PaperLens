import { afterEach, describe, expect, it, vi } from "vitest";
import { addPapers, chat, classifyMode, removePaper, setFeedback, stopChat } from "./api";

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

    await chat([], [], [], false, false, null, { onToken, onCitations, onDone });

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
    await chat([], [], [], false, false, null, {
      onToken: vi.fn(),
      onCitations: vi.fn(),
      onUsage,
    });

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
    await chat([], [], [], false, false, null, { onToken, onCitations: vi.fn() });

    expect(onToken).toHaveBeenCalledWith("Hi");
  });

  it("parses the compare_row SSE event", async () => {
    const row = {
      paper_id: "p1",
      title: "Paper 1",
      arxiv_id: null,
      text: "answer",
      citations: [],
      trace: [],
    };
    const sse = `event: compare_row\ndata: ${JSON.stringify(row)}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockStreamResponse(sse)));

    const onCompareRow = vi.fn();
    await chat([], [], [], false, true, null, {
      onToken: vi.fn(),
      onCitations: vi.fn(),
      onCompareRow,
    });

    expect(onCompareRow).toHaveBeenCalledWith(row);
  });

  it("sends edit_turn in the request body when editing a prior turn", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], false, false, "c1", { onToken: vi.fn(), onCitations: vi.fn() }, 2);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: false,
          compare: false,
          auto: false,
          chat_id: "c1",
          edit_turn: 2,
        }),
      }),
    );
  });

  it("sends edit_turn: null for a normal (non-edit) send", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], false, false, "c1", { onToken: vi.fn(), onCitations: vi.fn() });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: false,
          compare: false,
          auto: false,
          chat_id: "c1",
          edit_turn: null,
        }),
      }),
    );
  });

  it("sends per_paper: true in the request body when passed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], true, false, "c1", { onToken: vi.fn(), onCitations: vi.fn() });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: true,
          compare: false,
          auto: false,
          chat_id: "c1",
          edit_turn: null,
        }),
      }),
    );
  });

  it("sends compare: true in the request body when passed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat([], [], [], false, true, "c1", { onToken: vi.fn(), onCitations: vi.fn() });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: false,
          compare: true,
          auto: false,
          chat_id: "c1",
          edit_turn: null,
        }),
      }),
    );
  });

  it("sends auto: true in the request body when passed", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await chat(
      [],
      [],
      [],
      false,
      true,
      "c1",
      { onToken: vi.fn(), onCitations: vi.fn() },
      undefined,
      undefined,
      true,
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        body: JSON.stringify({
          messages: [],
          tags: [],
          papers: [],
          per_paper: false,
          compare: true,
          auto: true,
          chat_id: "c1",
          edit_turn: null,
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
    await chat([], [], [], false, false, "c1", {
      onToken: vi.fn(),
      onCitations: vi.fn(),
      onError,
      onDone,
    });

    expect(onError).toHaveBeenCalledWith("a turn is already in progress for this chat");
    expect(onDone).toHaveBeenCalledOnce();
  });

  it("passes the given signal to fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockStreamResponse("event: done\ndata: \n\n"));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await chat(
      [],
      [],
      [],
      false,
      false,
      "c1",
      { onToken: vi.fn(), onCitations: vi.fn() },
      0,
      controller.signal,
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it("resolves silently (no onError/onDone) when fetch itself is aborted", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new DOMException("aborted", "AbortError")));

    const onError = vi.fn();
    const onDone = vi.fn();
    await chat([], [], [], false, false, "c1", {
      onToken: vi.fn(),
      onCitations: vi.fn(),
      onError,
      onDone,
    });

    expect(onError).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
  });

  it("resolves silently (no onError/onDone) when the reader is aborted mid-stream", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        body: {
          getReader() {
            return {
              read: () => Promise.reject(new DOMException("aborted", "AbortError")),
            };
          },
        },
      } as unknown as Response),
    );

    const onError = vi.fn();
    const onDone = vi.fn();
    await chat([], [], [], false, false, "c1", {
      onToken: vi.fn(),
      onCitations: vi.fn(),
      onError,
      onDone,
    });

    expect(onError).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
  });
});

describe("classifyMode", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts messages/tags/papers to /api/chat/classify", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ mode: "compare", scope_size: 5 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await classifyMode([{ role: "user", content: "compare them" }], ["moe"], ["paper-a"]);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat/classify",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          messages: [{ role: "user", content: "compare them" }],
          tags: ["moe"],
          papers: ["paper-a"],
        }),
      }),
    );
  });

  it("parses the {mode, scope_size} response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ mode: "ask", scope_size: 1 }),
      }),
    );

    const result = await classifyMode([], [], []);

    expect(result).toEqual({ mode: "ask", scope_size: 1 });
  });
});

describe("stopChat", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to /api/chats/{id}/stop and returns the parsed body", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ stopped: true }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const result = await stopChat("c1");

    expect(fetchMock).toHaveBeenCalledWith("/api/chats/c1/stop", { method: "POST" });
    expect(result).toEqual({ stopped: true });
  });
});

describe("setFeedback", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs turn_index/vote/note to the chat's feedback route", async () => {
    const session = { id: "c1", name: "T", turns: [] };
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
        body: JSON.stringify({ turn_index: 1, vote: "up", note: "great citation" }),
      }),
    );
    expect(result).toEqual(session);
  });
});

describe("addPapers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs the raw ids/URLs and returns the per-line results", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () =>
        Promise.resolve({
          results: [
            { input: "https://arxiv.org/abs/2412.19437", status: "queued", name: "2412.19437" },
          ],
        }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const result = await addPapers(["https://arxiv.org/abs/2412.19437"]);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/papers",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_ids_or_urls: ["https://arxiv.org/abs/2412.19437"] }),
      }),
    );
    expect(result).toEqual({
      results: [
        { input: "https://arxiv.org/abs/2412.19437", status: "queued", name: "2412.19437" },
      ],
    });
  });

  it("throws the backend's error message on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: () => Promise.resolve({ error: "boom" }),
      } as Response),
    );

    await expect(addPapers(["2412.19437"])).rejects.toThrow("boom");
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
