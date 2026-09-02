import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ChatPage from "./ChatPage";

const chatSession = {
  id: "test-id",
  name: "Test",
  turns: [storedTurn("question", "regression marker text")],
};

const twoTurnSession = {
  id: "test-id",
  name: "Test",
  turns: [
    storedTurn("first question", "first answer"),
    storedTurn("second question", "second answer"),
  ],
};

function storedTurn(question: string, answer: string, overrides = {}) {
  return {
    question,
    answer,
    citations: [],
    trace: [],
    usage: null,
    feedback: null,
    per_paper: false,
    compare: false,
    compare_results: null,
    auto: false,
    ...overrides,
  };
}

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
      turns: [
        storedTurn("question", "regression marker text", {
          usage: { input_tokens: 100, output_tokens: 20, latency_ms: 1500 },
        }),
      ],
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

describe("ChatPage streamed-content separator", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("inserts a break between token text on either side of a tool-call trace event", async () => {
    // onToken concatenates raw text with no boundary of its own — a trace event (the
    // tool call) sits between two batches of prose from two different model rounds,
    // so without a separator "...handling." and "DeepSeek-V3 reports..." land glued
    // into one sentence.
    const sse =
      "event: token\ndata: KV cache handling.\n\n" +
      'event: trace\ndata: {"type":"action","query":"MLA"}\n\n' +
      "event: token\ndata: DeepSeek-V3 reports a 93% reduction.\n\n" +
      "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/chat") return Promise.resolve(mockStreamResponse(sse));
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

    fireEvent.change(screen.getByPlaceholderText("Ask about a paper or a concept…"), {
      target: { value: "How does MLA reduce KV cache?" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await screen.findByText(/DeepSeek-V3 reports/i);
    // The two fragments render as distinct text, not fused into "handling.DeepSeek-V3".
    expect(screen.queryByText(/handling\.DeepSeek-V3/)).not.toBeInTheDocument();
    expect(screen.getByText(/KV cache handling\./)).toBeInTheDocument();
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
          json: () =>
            Promise.resolve({
              ...chatSession,
              turns: [
                storedTurn("question", "regression marker text", {
                  feedback: { vote: "up", note: null },
                }),
              ],
            }),
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
          body: JSON.stringify({ turn_index: 0, vote: "up", note: null }),
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

describe("ChatPage Sources section", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("only shows citations actually cited in the answer text, not every retrieved passage", async () => {
    // Regression: a turn's stored `citations` array can include passages a search call
    // retrieved but the model never actually cited inline (registry entries aren't
    // filtered to what the final text uses) — SourceCards used to render all of them,
    // making the "Sources" list disagree with the answer's own faithfulness summary
    // (which only counts actually-cited refs). Most visible on a Compare turn's large
    // union citations list, but the bug — and the fix — apply to any turn.
    const cite = (ref: string, paperId: string, title: string) => ({
      ref,
      paper_id: paperId,
      title,
      breadcrumb: "",
      section_title: "Method",
      snippet: "some passage",
    });
    const session = {
      id: "test-id",
      name: "Test",
      turns: [
        storedTurn("question", "The cited claim [r1].", {
          citations: [cite("r1", "p1", "Cited Paper"), cite("r2", "p2", "Uncited Paper")],
        }),
      ],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        const body =
          url === "/api/chats/test-id"
            ? session
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

    await screen.findByText(/The cited claim/i);
    expect(screen.getByText("Cited Paper")).toBeInTheDocument();
    expect(screen.queryByText("Uncited Paper")).not.toBeInTheDocument();
  });
});

describe("ChatPage secondary 'Broaden recall per paper' knob (Ask mode)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders off by default", async () => {
    vi.stubGlobal("fetch", mockFetch());
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
    expect(screen.getByLabelText("Broaden recall per paper")).not.toBeChecked();
  });

  it("toggling on and sending includes per_paper: true in the request body", async () => {
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    // `init` is unused but kept in the signature so fetchMock.mock.calls types as a 2-tuple
    // below (call[1] / [, init]) instead of narrowing to just [input].
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "/api/chat") return Promise.resolve(mockStreamResponse(sse));
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

    fireEvent.click(screen.getByLabelText("Broaden recall per paper"));
    fireEvent.change(screen.getByPlaceholderText("Ask about a paper or a concept…"), {
      target: { value: "a question" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.per_paper).toBe(true);
  });

  it("stays on for a second message without re-toggling", async () => {
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    // `init` is unused but kept in the signature so fetchMock.mock.calls types as a 2-tuple
    // below (call[1] / [, init]) instead of narrowing to just [input].
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "/api/chat") return Promise.resolve(mockStreamResponse(sse));
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
    const composer = screen.getByPlaceholderText("Ask about a paper or a concept…");

    fireEvent.click(screen.getByLabelText("Broaden recall per paper"));
    fireEvent.change(composer, { target: { value: "first" } });
    fireEvent.click(screen.getByLabelText("Send"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));

    fireEvent.change(composer, { target: { value: "second" } });
    fireEvent.click(screen.getByLabelText("Send"));
    await waitFor(() =>
      expect(fetchMock.mock.calls.filter(([u]) => String(u) === "/api/chat")).toHaveLength(2),
    );

    const bodies = fetchMock.mock.calls
      .filter(([u]) => String(u) === "/api/chat")
      .map(([, init]) => JSON.parse((init as RequestInit).body as string));
    expect(bodies.map((b) => b.per_paper)).toEqual([true, true]);
  });

  it("resets to off on New chat", async () => {
    vi.stubGlobal("fetch", mockFetch());
    render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );
    await screen.findByText(/regression marker text/i);
    fireEvent.click(screen.getByLabelText("Broaden recall per paper"));
    expect(screen.getByLabelText("Broaden recall per paper")).toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /New chat/i }));

    // New chat defaults to Auto, which hides the Ask-only knob entirely — not just
    // unchecked, gone from the document (same "hide, don't disable" treatment Compare gets).
    await waitFor(() =>
      expect(screen.queryByLabelText("Broaden recall per paper")).not.toBeInTheDocument(),
    );
    // Switching explicitly to Ask reveals it, reset to off — not carried over from before.
    fireEvent.click(screen.getByRole("radio", { name: "Ask" }));
    expect(screen.getByLabelText("Broaden recall per paper")).not.toBeChecked();
  });

  it("restores the target conversation's own per_paper state when switching (not a blind reset)", async () => {
    // otherSession's last message used per_paper: true — switching to it must show the
    // toggle checked, not reset to off, proving this reflects the target conversation's
    // own history rather than always defaulting.
    const otherSession = {
      id: "other-id",
      name: "Other",
      turns: [storedTurn("other question", "other answer", { per_paper: true })],
    };
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/chats") {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve([
              { id: "test-id", name: "Test", updated_at: "" },
              { id: "other-id", name: "Other", updated_at: "" },
            ]),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/chats/other-id"
            ? otherSession
            : url.startsWith("/api/tags") || url.startsWith("/api/papers")
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
    // test-id has no per_paper history (chatSession predates the field) -> off by default.
    expect(screen.getByLabelText("Broaden recall per paper")).not.toBeChecked();

    fireEvent.click(await screen.findByText("Other"));

    // other-id's last message used per_paper: true -> restored, not reset to off.
    expect(await screen.findByLabelText("Broaden recall per paper")).toBeChecked();
  });

  it("restores true from an earlier turn but false from the latest turn (uses the latest, not the first)", async () => {
    const mixedSession = {
      id: "test-id",
      name: "Test",
      turns: [
        storedTurn("first question", "first answer", { per_paper: true }),
        storedTurn("second question", "second answer"),
      ],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        const body =
          url === "/api/chats/test-id"
            ? mixedSession
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

    expect(await screen.findByLabelText("Broaden recall per paper")).not.toBeChecked();
  });
});

describe("ChatPage Ask/Compare mode", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const twoPapers = [
    { paper_id: "p1", title: "Paper One", tags: [] },
    { paper_id: "p2", title: "Paper Two", tags: [] },
  ];

  function twoPapersFetch(extra?: (url: string) => Response | null) {
    return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      const overridden = extra?.(url);
      if (overridden) return Promise.resolve(overridden);
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/papers"
            ? twoPapers
            : url.startsWith("/api/tags") || url.startsWith("/api/chats")
              ? []
              : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
  }

  it("defaults to Auto when starting a fresh conversation", async () => {
    vi.stubGlobal("fetch", mockFetch());
    render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/"]}>
          <Routes>
            <Route path="/" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );

    expect(screen.getByRole("radio", { name: "Auto" })).toBeChecked();
  });

  it("Ask is selected by default and Compare is disabled below 2 resolved papers", async () => {
    vi.stubGlobal("fetch", mockFetch()); // default mock: 0 papers
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
    expect(screen.getByRole("radio", { name: "Ask" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Compare" })).toBeDisabled();
  });

  it("Compare is enabled once 2+ papers resolve into scope", async () => {
    vi.stubGlobal("fetch", twoPapersFetch());
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
    await waitFor(() => expect(screen.getByRole("radio", { name: "Compare" })).not.toBeDisabled());
  });

  it("switching to Compare hides the secondary 'Broaden recall' knob; switching back restores it", async () => {
    vi.stubGlobal("fetch", twoPapersFetch());
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
    await waitFor(() => expect(screen.getByRole("radio", { name: "Compare" })).not.toBeDisabled());
    expect(screen.getByLabelText("Broaden recall per paper")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "Compare" }));
    expect(screen.queryByLabelText("Broaden recall per paper")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "Ask" }));
    expect(screen.getByLabelText("Broaden recall per paper")).toBeInTheDocument();
  });

  it("sending in Compare mode includes compare: true and per_paper: false in the request body", async () => {
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const fetchMock = twoPapersFetch((url) =>
      url === "/api/chat" ? mockStreamResponse(sse) : null,
    );
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
    await waitFor(() => expect(screen.getByRole("radio", { name: "Compare" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("radio", { name: "Compare" }));
    fireEvent.change(screen.getByPlaceholderText(/Ask one thing to compare/i), {
      target: { value: "compare them" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.compare).toBe(true);
    expect(sent.per_paper).toBe(false);
  });

  it("resets to Auto on New chat", async () => {
    vi.stubGlobal("fetch", twoPapersFetch());
    render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );
    await screen.findByText(/regression marker text/i);
    await waitFor(() => expect(screen.getByRole("radio", { name: "Compare" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("radio", { name: "Compare" }));
    expect(screen.getByRole("radio", { name: "Compare" })).toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /New chat/i }));

    expect(await screen.findByRole("radio", { name: "Auto" })).toBeChecked();
  });

  it("restores Compare when switching to a conversation whose latest turn used it (not a blind reset)", async () => {
    const otherSession = {
      id: "other-id",
      name: "Other",
      turns: [storedTurn("other question", "other answer", { compare: true })],
    };
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/chats") {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve([
              { id: "test-id", name: "Test", updated_at: "" },
              { id: "other-id", name: "Other", updated_at: "" },
            ]),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/chats/other-id"
            ? otherSession
            : url === "/api/papers"
              ? twoPapers
              : url.startsWith("/api/tags")
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
    expect(screen.getByRole("radio", { name: "Ask" })).toBeChecked();

    fireEvent.click(await screen.findByText("Other"));

    expect(await screen.findByRole("radio", { name: "Compare" })).toBeChecked();
  });

  it("confirms before sending Compare over more than 12 resolved papers, and aborts on cancel", async () => {
    const manyPapers = Array.from({ length: 13 }, (_, i) => ({
      paper_id: `p${i}`,
      title: `Paper ${i}`,
      tags: [],
    }));
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/papers"
            ? manyPapers
            : url.startsWith("/api/tags") || url.startsWith("/api/chats")
              ? []
              : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);

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
    await waitFor(() => expect(screen.getByRole("radio", { name: "Compare" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("radio", { name: "Compare" }));
    fireEvent.change(screen.getByPlaceholderText(/Ask one thing to compare/i), {
      target: { value: "compare them" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    expect(confirmSpy).toHaveBeenCalled();
    expect(fetchMock.mock.calls.some(([u]) => String(u) === "/api/chat")).toBe(false);
    confirmSpy.mockRestore();
  });

  it("Broaden recall stays hidden under Auto, same as under Compare", async () => {
    vi.stubGlobal("fetch", twoPapersFetch());
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
    fireEvent.click(screen.getByRole("radio", { name: "Ask" }));
    expect(screen.getByLabelText("Broaden recall per paper")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));
    expect(screen.queryByLabelText("Broaden recall per paper")).not.toBeInTheDocument();
  });

  it("Auto with a small resolved scope sends without a classify-confirm dialog, and shows the Auto badge", async () => {
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const classifyMock = vi.fn();
    const fetchMock = twoPapersFetch((url) => {
      if (url === "/api/chat") return mockStreamResponse(sse);
      if (url === "/api/chat/classify") {
        classifyMock();
        return {
          ok: true,
          json: () => Promise.resolve({ mode: "ask", scope_size: 2 }),
        } as Response;
      }
      return null;
    });
    vi.stubGlobal("fetch", fetchMock);
    const confirmSpy = vi.spyOn(window, "confirm");

    const { container } = render(
      <MantineProvider>
        <MemoryRouter initialEntries={["/c/test-id"]}>
          <Routes>
            <Route path="/c/:chatId" element={<ChatPage />} />
          </Routes>
        </MemoryRouter>
      </MantineProvider>,
    );
    await screen.findByText(/regression marker text/i);
    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));

    fireEvent.change(screen.getByPlaceholderText(/Auto will pick how to search/i), {
      target: { value: "a question" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    expect(classifyMock).toHaveBeenCalled();
    expect(confirmSpy).not.toHaveBeenCalled();
    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.compare).toBe(false);
    expect(sent.auto).toBe(true);

    const badge = container.querySelector(".mantine-Badge-root");
    expect(badge).toHaveTextContent("Auto");
    confirmSpy.mockRestore();
  });

  it("greys out the composer while Auto's classify call is in flight", async () => {
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    let resolveClassify!: (value: Response) => void;
    const classifyPromise = new Promise<Response>((resolve) => {
      resolveClassify = resolve;
    });
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/chat/classify") return classifyPromise;
      if (url === "/api/chat") return Promise.resolve(mockStreamResponse(sse));
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/papers"
            ? twoPapers
            : url.startsWith("/api/tags") || url.startsWith("/api/chats")
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
    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));

    const textarea = screen.getByPlaceholderText(/Auto will pick how to search/i);
    fireEvent.change(textarea, { target: { value: "a question" } });
    expect(textarea).not.toBeDisabled();
    fireEvent.click(screen.getByLabelText("Send"));

    // Still captured in the request that's about to be classified — further typing here
    // wouldn't do anything, so the composer greys out for the duration of the pre-flight call.
    await waitFor(() => expect(textarea).toBeDisabled());

    resolveClassify({
      ok: true,
      json: () => Promise.resolve({ mode: "ask", scope_size: 2 }),
    } as Response);

    await waitFor(() => expect(textarea).not.toBeDisabled());
    // Let the real turn settle too, so no request from this test is still in flight
    // when the next test's fetch mock takes over.
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
  });

  it("Auto resolved to Compare over a large scope shows the confirm dialog before sending", async () => {
    const manyPapers = Array.from({ length: 13 }, (_, i) => ({
      paper_id: `p${i}`,
      title: `Paper ${i}`,
      tags: [],
    }));
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    // `init` is unused but kept in the signature so fetchMock.mock.calls types as a 2-tuple
    // below (call[1]) instead of narrowing to just [input].
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "/api/chat") return Promise.resolve(mockStreamResponse(sse));
      if (url === "/api/chat/classify") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ mode: "compare", scope_size: 13 }),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/papers"
            ? manyPapers
            : url.startsWith("/api/tags") || url.startsWith("/api/chats")
              ? []
              : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);

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
    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));

    fireEvent.change(screen.getByPlaceholderText(/Auto will pick how to search/i), {
      target: { value: "compare them" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.compare).toBe(true);
    expect(sent.auto).toBe(true);
    confirmSpy.mockRestore();
  });

  it("canceling the Auto confirm dialog sends nothing", async () => {
    const manyPapers = Array.from({ length: 13 }, (_, i) => ({
      paper_id: `p${i}`,
      title: `Paper ${i}`,
      tags: [],
    }));
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/chat/classify") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ mode: "compare", scope_size: 13 }),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/papers"
            ? manyPapers
            : url.startsWith("/api/tags") || url.startsWith("/api/chats")
              ? []
              : {};
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);

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
    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));

    fireEvent.change(screen.getByPlaceholderText(/Auto will pick how to search/i), {
      target: { value: "compare them" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(fetchMock.mock.calls.some(([u]) => String(u) === "/api/chat")).toBe(false);
    confirmSpy.mockRestore();
  });

  it("a classifyMode failure falls back to sending as Ask", async () => {
    const sse = "event: citations\ndata: []\n\nevent: done\ndata: \n\n";
    const fetchMock = twoPapersFetch((url) => {
      if (url === "/api/chat") return mockStreamResponse(sse);
      if (url === "/api/chat/classify") {
        return { ok: false, status: 500, statusText: "Internal Server Error" } as Response;
      }
      return null;
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
    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));

    fireEvent.change(screen.getByPlaceholderText(/Auto will pick how to search/i), {
      target: { value: "a question" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const call = fetchMock.mock.calls.find(([u]) => String(u) === "/api/chat")!;
    const sent = JSON.parse((call[1] as RequestInit).body as string);
    expect(sent.compare).toBe(false);
    expect(sent.auto).toBe(true);
  });

  it("restores Auto (not the resolved mode) when switching to a conversation whose latest turn was auto-decided", async () => {
    const otherSession = {
      id: "other-id",
      name: "Other",
      turns: [storedTurn("other question", "other answer", { compare: true, auto: true })],
    };
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/chats") {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve([
              { id: "test-id", name: "Test", updated_at: "" },
              { id: "other-id", name: "Other", updated_at: "" },
            ]),
        } as Response);
      }
      const body =
        url === "/api/chats/test-id"
          ? chatSession
          : url === "/api/chats/other-id"
            ? otherSession
            : url === "/api/papers"
              ? twoPapers
              : url.startsWith("/api/tags")
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
    expect(screen.getByRole("radio", { name: "Ask" })).toBeChecked();

    fireEvent.click(await screen.findByText("Other"));

    expect(await screen.findByRole("radio", { name: "Auto" })).toBeChecked();
  });
});

describe("ChatPage stop generating", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("clicking Stop unlocks the composer immediately without waiting for the stream", async () => {
    // The stream never resolves on its own (reader.read() hangs forever) — Stop must
    // still flip the UI back, proving it doesn't wait on the backend.
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/chat") {
        return Promise.resolve({
          status: 200,
          body: { getReader: () => ({ read: () => new Promise(() => {}) }) },
        } as unknown as Response);
      }
      if (url === "/api/chats/test-id/stop" && init?.method === "POST") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ stopped: true }),
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

    fireEvent.change(screen.getByPlaceholderText("Ask about a paper or a concept…"), {
      target: { value: "a question" },
    });
    fireEvent.click(screen.getByLabelText("Send"));

    const stopButton = await screen.findByLabelText("Stop generating");
    expect(screen.queryByLabelText("Send")).not.toBeInTheDocument();
    // Editing the just-sent user turn is locked out while a turn is in flight.
    const editsWhileStreaming = screen.getAllByLabelText("Edit message");
    expect(editsWhileStreaming[editsWhileStreaming.length - 1]).toBeDisabled();

    fireEvent.click(stopButton);

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/chats/test-id/stop",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    // Composer is unlocked right away — no waiting on the (still-hanging) stream.
    expect(await screen.findByLabelText("Send")).toBeInTheDocument();
    const editsAfterStop = screen.getAllByLabelText("Edit message");
    expect(editsAfterStop[editsAfterStop.length - 1]).not.toBeDisabled();
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
    expect(sent.edit_turn).toBe(1);
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

  it("under Auto mode, declining the discard-exchanges confirm never calls classifyMode", async () => {
    // vi.spyOn reuses window.confirm's existing spy (if the preceding test left one behind
    // without restoring it) rather than resetting its call history — mockClear() makes this
    // test's own call count assertion robust to that regardless of test order.
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    confirmSpy.mockClear();
    const classifyMock = vi.fn();
    const fetchMock = baseFetchMock((url) => {
      if (url === "/api/chat/classify") {
        classifyMock();
        return {
          ok: true,
          json: () => Promise.resolve({ mode: "ask", scope_size: 0 }),
        } as Response;
      }
      return undefined;
    });
    renderTwoTurnChat(fetchMock);
    await screen.findByText("first question");
    fireEvent.click(screen.getByRole("radio", { name: "Auto" }));

    fireEvent.click(screen.getAllByLabelText("Edit message")[0]);
    const textarea = await screen.findByDisplayValue("first question");
    fireEvent.change(textarea, { target: { value: "first question, edited" } });
    fireEvent.click(screen.getByLabelText("Save and resend"));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalledOnce());
    // The cheap, synchronous discard confirm is checked before the async classifyMode
    // round trip — declining it means classifyMode is never called at all.
    expect(classifyMock).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalledWith("/api/chat", expect.anything());
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
    expect(sent.edit_turn).toBe(0);
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
