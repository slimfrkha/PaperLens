import { afterEach, describe, expect, it, vi } from "vitest";
import { chat } from "./api";

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
