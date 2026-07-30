import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ChatPage from "./ChatPage";

const chatSession = {
  id: "test-id",
  name: "Test",
  messages: [{ role: "assistant", content: "regression marker text" }],
  citations: [null],
  traces: [null],
};

const twoTurnSession = {
  id: "test-id",
  name: "Test",
  messages: [
    { role: "user", content: "first question" },
    { role: "assistant", content: "first answer" },
    { role: "user", content: "second question" },
    { role: "assistant", content: "second answer" },
  ],
  citations: [null, [], null, []],
  traces: [null, [], null, []],
};

function mockStreamResponse(sse: string): Response {
  let sent = false;
  return {
    status: 200,
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

function mockFetch() {
  return vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    const body =
      url === "/api/chats/test-id"
        ? chatSession
        : url.startsWith("/api/tags") ||
            url.startsWith("/api/papers") ||
            url.startsWith("/api/chats")
          ? []
          : {};
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
  });
}

describe("ChatPage scroll-to-bottom effect", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch());
    // Reproduce the Chrome condition that caused the crash: a smooth-scroll polyfill
    // / browser extension makes scrollIntoView return a Promise. A concise-body effect
    // would leak that Promise as its cleanup → "destroy is not a function" on unmount.
    Element.prototype.scrollIntoView = vi.fn(() =>
      Promise.resolve(),
    ) as unknown as typeof Element.prototype.scrollIntoView;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not leak scrollIntoView's return value as an effect cleanup", async () => {
    const { unmount } = render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );
    // Wait until the loaded turn renders, so the scroll effect has run against a real ref.
    await screen.findByText(/regression marker text/i);
    // Without the block-body fix, the stored cleanup is the Promise → unmount calls it → throws.
    expect(() => unmount()).not.toThrow();
  });
});

describe("ChatPage usage metadata", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders token count and latency for a loaded turn", async () => {
    const sessionWithUsage = {
      ...chatSession,
      usage: [{ input_tokens: 100, output_tokens: 20, latency_ms: 1500 }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        const body =
          url === "/api/chats/test-id"
            ? sessionWithUsage
            : url.startsWith("/api/tags") ||
                url.startsWith("/api/papers") ||
                url.startsWith("/api/chats")
              ? []
              : {};
        return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
      }),
    );

    render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );

    await screen.findByText(/regression marker text/i);
    expect(await screen.findByText("120 tokens · 1.5s")).toBeInTheDocument();
  });
});

describe("ChatPage feedback control", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders on a completed assistant turn and posts a vote on click", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (init?.method === "POST" && url === "/api/chats/test-id/feedback") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...chatSession, feedback: [{ vote: "up", note: null }] }),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url.startsWith("/api/tags") ||
              url.startsWith("/api/papers") ||
              url.startsWith("/api/chats")
            ? []
            : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );

    await screen.findByText(/regression marker text/i);
    fireEvent.click(await screen.findByLabelText("Good answer"));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/chats/test-id/feedback",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ index: 0, vote: "up", note: null }),
        }),
      ),
    );
  });

  it("does not scroll the page when feedback is set", async () => {
    // Regression: the scroll-to-bottom effect used to key off the whole `turns`
    // array, so patching feedback on ANY turn (even one far above the fold) re-ran
    // it and yanked the view down to the bottom.
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (init?.method === "POST" && url === "/api/chats/test-id/feedback") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...chatSession, feedback: [{ vote: "up", note: null }] }),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url.startsWith("/api/tags") ||
              url.startsWith("/api/papers") ||
              url.startsWith("/api/chats")
            ? []
            : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);
    const scrollSpy = vi.fn();
    Element.prototype.scrollIntoView = scrollSpy;

    render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );
    await screen.findByText(/regression marker text/i);
    const callsAfterLoad = scrollSpy.mock.calls.length;

    fireEvent.click(await screen.findByLabelText("Good answer"));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("/api/chats/test-id/feedback", expect.anything()),
    );

    expect(scrollSpy.mock.calls.length).toBe(callsAfterLoad);
  });
});

describe("ChatPage edit-and-resume", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function renderTwoTurnChat(fetchMock: ReturnType<typeof vi.fn>) {
    vi.stubGlobal("fetch", fetchMock);
    return render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );
  }

  function baseFetchMock(extra?: (url: string, init?: RequestInit) => Response | undefined) {
    return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const handled = extra?.(url, init);
      if (handled) return Promise.resolve(handled);
      const body =
        url === "/api/chats/test-id"
          ? twoTurnSession
          : url.startsWith("/api/tags") ||
              url.startsWith("/api/papers") ||
              url.startsWith("/api/chats")
            ? []
            : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
  }

  it("shows an edit affordance on user turns only", async () => {
    renderTwoTurnChat(baseFetchMock());
    await screen.findByText("first question");

    expect(screen.getAllByLabelText("Edit message")).toHaveLength(2); // one per user turn
  });

  it("cancel restores the original content without calling /api/chat", async () => {
    const fetchMock = baseFetchMock();
    renderTwoTurnChat(fetchMock);
    await screen.findByText("first question");

    fireEvent.click(screen.getAllByLabelText("Edit message")[0]);
    const textarea = await screen.findByDisplayValue("first question");
    fireEvent.change(textarea, { target: { value: "edited but cancelled" } });
    fireEvent.click(screen.getByLabelText("Cancel edit"));

    expect(await screen.findByText("first question")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith("/api/chat", expect.anything());
  });

  it("editing the last turn (no later exchanges) saves without a confirm prompt", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const fetchMock = baseFetchMock((url) =>
      url === "/api/chat" ? mockStreamResponse(sse) : undefined,
    );
    renderTwoTurnChat(fetchMock);
    await screen.findByText("second question");

    fireEvent.click(screen.getAllByLabelText("Edit message")[1]);
    const textarea = await screen.findByDisplayValue("second question");
    fireEvent.change(textarea, { target: { value: "second question, edited" } });
    fireEvent.click(screen.getByLabelText("Save and resend"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    expect(confirmSpy).not.toHaveBeenCalled();

    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.edit_index).toBe(2);
    expect(sent.messages).toEqual([
      { role: "user", content: "first question" },
      { role: "assistant", content: "first answer" },
      { role: "user", content: "second question, edited" },
    ]);
    // The now-stale second exchange is gone from view; only the edited turn + its
    // (streaming, then resolved) reply remain after it.
    expect(screen.queryByText("second answer")).not.toBeInTheDocument();
    expect(await screen.findByText("second question, edited")).toBeInTheDocument();
  });

  it("editing an earlier turn prompts for confirmation before discarding later exchanges", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const fetchMock = baseFetchMock();
    renderTwoTurnChat(fetchMock);
    await screen.findByText("first question");

    fireEvent.click(screen.getAllByLabelText("Edit message")[0]);
    const textarea = await screen.findByDisplayValue("first question");
    fireEvent.change(textarea, { target: { value: "first question, edited" } });
    fireEvent.click(screen.getByLabelText("Save and resend"));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalledOnce());
    // Declining the confirm must not touch the server or the conversation.
    expect(fetchMock).not.toHaveBeenCalledWith("/api/chat", expect.anything());
    expect(await screen.findByText("second question")).toBeInTheDocument();
  });

  it("proceeds with the edit once the confirmation is accepted", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const fetchMock = baseFetchMock((url) =>
      url === "/api/chat" ? mockStreamResponse(sse) : undefined,
    );
    renderTwoTurnChat(fetchMock);
    await screen.findByText("first question");

    fireEvent.click(screen.getAllByLabelText("Edit message")[0]);
    const textarea = await screen.findByDisplayValue("first question");
    fireEvent.change(textarea, { target: { value: "first question, edited" } });
    fireEvent.click(screen.getByLabelText("Save and resend"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.edit_index).toBe(0);
    expect(sent.messages).toEqual([{ role: "user", content: "first question, edited" }]);
    // Everything from the original second exchange onward is gone.
    expect(screen.queryByText("second question")).not.toBeInTheDocument();
    expect(screen.queryByText("second answer")).not.toBeInTheDocument();
  });

  it("disables the Save button when the edited content is empty", async () => {
    renderTwoTurnChat(baseFetchMock());
    await screen.findByText("first question");

    fireEvent.click(screen.getAllByLabelText("Edit message")[0]);
    const textarea = await screen.findByDisplayValue("first question");
    fireEvent.change(textarea, { target: { value: "   " } });

    expect(screen.getByLabelText("Save and resend")).toBeDisabled();
  });

  it("closes an open edit box when an unrelated normal message is sent", async () => {
    // Regression: leaving an edit box open and then sending a different message left
    // Save clickable but silently no-op'd once `busy` flipped true for the new turn.
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const fetchMock = baseFetchMock((url) =>
      url === "/api/chat" ? mockStreamResponse(sse) : undefined,
    );
    renderTwoTurnChat(fetchMock);
    await screen.findByText("first question");

    fireEvent.click(screen.getAllByLabelText("Edit message")[0]);
    await screen.findByDisplayValue("first question");

    const composer = screen.getByPlaceholderText("Ask about a paper or a concept…");
    fireEvent.change(composer, { target: { value: "a brand new message" } });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() =>
      expect(screen.queryByDisplayValue("first question")).not.toBeInTheDocument(),
    );
    // The original turn's plain text is back (not swallowed by the edit box closing).
    expect(screen.getByText("first question")).toBeInTheDocument();
  });
});
