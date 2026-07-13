import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
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
