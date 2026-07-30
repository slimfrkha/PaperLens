import { afterEach, describe, expect, it, vi } from "vitest";
import { chat, setFeedback } from "./api";

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

    await chat([], [], [], null, { onToken, onCitations, onDone });

    expect(onToken).toHaveBeenCalledWith("Hello");
    expect(onCitations).toHaveBeenCalledWith([
      expect.objectContaining({ ref: "1", paper_id: "p1" }),
    ]);
    expect(onDone).toHaveBeenCalledOnce();
  });

  it("normalizes CRLF frame separators from the sse-starlette server", async () => {
    const sse = "event: token\r\ndata: Hi\r\n\r\n";
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockStreamResponse(sse)));

    const onToken = vi.fn();
    await chat([], [], [], null, { onToken, onCitations: vi.fn() });

    expect(onToken).toHaveBeenCalledWith("Hi");
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
