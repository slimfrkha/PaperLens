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
