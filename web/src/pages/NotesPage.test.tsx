import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import NotesPage from "./NotesPage";

const papers = [
  { paper_id: "paper-a", title: "Paper A", tags: [] },
  { paper_id: "paper-b", title: "Paper B", tags: [] },
];

const annotations = [
  {
    id: "1",
    snippet: "attention shrinks the cache",
    section_title: "Method",
    section_slug: "method",
    note: "check this",
    created_at: "2026-01-01T00:00:00",
    updated_at: "2026-01-01T00:00:00",
    paper_id: "paper-a",
    paper_title: "Paper A",
    arxiv_id: "1111.1111",
  },
  {
    id: "2",
    snippet: "a different passage",
    section_title: "Results",
    section_slug: "results",
    note: "",
    created_at: "2026-01-02T00:00:00",
    updated_at: "2026-01-02T00:00:00",
    paper_id: "paper-b",
    paper_title: "Paper B",
  },
];

function stubFetch(
  data: typeof annotations = annotations,
  deleteResponse: Response = { ok: true, json: () => Promise.resolve({ ok: true }) } as Response,
) {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/annotations") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(data) } as Response);
    }
    if (url === "/api/papers") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(papers) } as Response);
    }
    if (url === "/api/papers/paper-a/annotations/1" && init?.method === "DELETE") {
      return Promise.resolve(deleteResponse);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

const mockNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => mockNavigate };
});

function renderNotesPage() {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <NotesPage />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("NotesPage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    mockNavigate.mockClear();
  });

  it("groups notes by paper and renders each one", async () => {
    vi.stubGlobal("fetch", stubFetch());
    renderNotesPage();

    await screen.findByRole("link", { name: "Paper A" });
    expect(screen.getByRole("link", { name: "Paper B" })).toBeInTheDocument();
    expect(screen.getByText(/attention shrinks the cache/)).toBeInTheDocument();
    expect(screen.getByText(/a different passage/)).toBeInTheDocument();
  });

  it("filters by text across snippet and note", async () => {
    vi.stubGlobal("fetch", stubFetch());
    renderNotesPage();
    await screen.findByRole("link", { name: "Paper A" });

    fireEvent.change(screen.getByLabelText("Search notes"), { target: { value: "check this" } });

    await waitFor(() => expect(screen.queryByText(/a different passage/)).not.toBeInTheDocument());
    expect(screen.getByText(/attention shrinks the cache/)).toBeInTheDocument();
  });

  it("navigates to the paper with highlight state on click", async () => {
    vi.stubGlobal("fetch", stubFetch());
    renderNotesPage();
    const snippet = await screen.findByText(/attention shrinks the cache/);

    fireEvent.click(snippet);

    expect(mockNavigate).toHaveBeenCalledWith("/papers/paper-a", {
      state: { highlight: "attention shrinks the cache", section: "Method" },
    });
  });

  it("deletes a note on confirm and removes it from the list", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const fetchMock = stubFetch();
    vi.stubGlobal("fetch", fetchMock);
    renderNotesPage();
    await screen.findByText(/attention shrinks the cache/);

    // Two notes are on the page (one per paper); the paper-a one is first.
    fireEvent.click(screen.getAllByRole("button", { name: "Delete note" })[0]);

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("/api/papers/paper-a/annotations/1", {
        method: "DELETE",
      }),
    );
    await waitFor(() =>
      expect(screen.queryByText(/attention shrinks the cache/)).not.toBeInTheDocument(),
    );
    // Deleting doesn't trigger the card's own click-through navigation.
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("surfaces the backend's error via alert and keeps the note on a failed delete", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const alertMock = vi.spyOn(window, "alert").mockImplementation(() => {});
    const fetchMock = stubFetch(annotations, {
      ok: false,
      status: 404,
      statusText: "Not Found",
      json: () => Promise.resolve({ error: "not found" }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);
    renderNotesPage();
    await screen.findByText(/attention shrinks the cache/);

    fireEvent.click(screen.getAllByRole("button", { name: "Delete note" })[0]);

    await waitFor(() => expect(alertMock).toHaveBeenCalled());
    // The note is still there — a failed delete doesn't silently vanish it.
    expect(screen.getByText(/attention shrinks the cache/)).toBeInTheDocument();
  });

  it("does not delete when the confirm dialog is dismissed", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const fetchMock = stubFetch();
    vi.stubGlobal("fetch", fetchMock);
    renderNotesPage();
    await screen.findByText(/attention shrinks the cache/);

    // Two notes are on the page (one per paper); the paper-a one is first.
    fireEvent.click(screen.getAllByRole("button", { name: "Delete note" })[0]);

    await new Promise((r) => setTimeout(r, 0));
    expect(fetchMock).not.toHaveBeenCalledWith("/api/papers/paper-a/annotations/1", {
      method: "DELETE",
    });
    expect(screen.getByText(/attention shrinks the cache/)).toBeInTheDocument();
  });

  it("shows the true-empty state when the library has no notes", async () => {
    vi.stubGlobal("fetch", stubFetch([]));
    renderNotesPage();

    await screen.findByText(/No notes yet/);
  });

  it("shows the filtered-empty state distinctly from the true-empty state", async () => {
    vi.stubGlobal("fetch", stubFetch());
    renderNotesPage();
    await screen.findByRole("link", { name: "Paper A" });

    fireEvent.change(screen.getByLabelText("Search notes"), {
      target: { value: "nothing matches this" },
    });

    await screen.findByText("No notes match your filters.");
    expect(screen.queryByText(/No notes yet/)).not.toBeInTheDocument();
  });
});
