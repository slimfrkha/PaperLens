import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import PapersPage from "./PapersPage";

const papers = [{ paper_id: "paper-a", title: "Paper A", tags: [], n_chunks: 3 }];

function stubFetch(deleteResponse: Response = { ok: true, status: 204 } as Response) {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/admin/papers/paper-a" && init?.method === "DELETE") {
      return Promise.resolve(deleteResponse);
    }
    if (url === "/api/papers") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(papers) } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

function renderPapersPage() {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <PapersPage />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("PapersPage remove paper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("asks for confirmation, DELETEs on confirm, and refetches the list", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const fetchMock = stubFetch();
    vi.stubGlobal("fetch", fetchMock);

    renderPapersPage();
    await screen.findByText("Paper A");

    fireEvent.click(screen.getByRole("button", { name: /remove paper/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("/api/admin/papers/paper-a", { method: "DELETE" }),
    );
    // Refetched the list after a successful removal (initial load + post-delete reload).
    await waitFor(() =>
      expect(fetchMock.mock.calls.filter(([u]) => String(u) === "/api/papers").length).toBe(2),
    );
  });

  it("does not delete when the confirm dialog is dismissed", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const fetchMock = stubFetch();
    vi.stubGlobal("fetch", fetchMock);

    renderPapersPage();
    await screen.findByText("Paper A");

    fireEvent.click(screen.getByRole("button", { name: /remove paper/i }));

    await new Promise((r) => setTimeout(r, 0));
    expect(fetchMock).not.toHaveBeenCalledWith("/api/admin/papers/paper-a", { method: "DELETE" });
  });

  it("surfaces the backend's error via alert and does not refetch on a failed delete", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const alertMock = vi.spyOn(window, "alert").mockImplementation(() => {});
    const fetchMock = stubFetch({
      ok: false,
      status: 404,
      statusText: "Not Found",
      json: () => Promise.resolve({ error: "not found" }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    renderPapersPage();
    await screen.findByText("Paper A");

    fireEvent.click(screen.getByRole("button", { name: /remove paper/i }));

    await waitFor(() => expect(alertMock).toHaveBeenCalledWith("not found"));
    // No post-failure refetch: only the initial page-load GET happened.
    expect(fetchMock.mock.calls.filter(([u]) => String(u) === "/api/papers").length).toBe(1);
    // The card is still there — a failed remove doesn't silently vanish it.
    expect(screen.getByText("Paper A")).toBeInTheDocument();
  });
});
