import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import AdminPage from "./AdminPage";

const status = {
  db: { n_papers: 1, n_chunks: 5 },
  tags: [],
  pending: [],
  ingestion: { state: "idle", total: 0, done: 0, current: null, errors: [] },
};

function renderAdminPage() {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <AdminPage />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("AdminPage add paper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs the entered arXiv id and clears the input on success", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/admin/status") {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(status) } as Response);
      }
      if (url === "/api/admin/papers" && init?.method === "POST") {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ queued: true, name: "2412.19437" }),
        } as Response);
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);

    renderAdminPage();
    await screen.findByPlaceholderText(/arXiv id or URL/i);

    const input = screen.getByPlaceholderText(/arXiv id or URL/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "2412.19437" } });
    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/admin/papers",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ arxiv_id_or_url: "2412.19437" }),
        }),
      ),
    );
    await waitFor(() => expect(input.value).toBe(""));
  });

  it("shows the backend's error message on a 409 duplicate", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/admin/status") {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(status) } as Response);
      }
      if (url === "/api/admin/papers" && init?.method === "POST") {
        return Promise.resolve({
          ok: false,
          status: 409,
          statusText: "Conflict",
          json: () => Promise.resolve({ error: "already curated as deepseek-v3" }),
        } as Response);
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
    });
    vi.stubGlobal("fetch", fetchMock);

    renderAdminPage();
    await screen.findByPlaceholderText(/arXiv id or URL/i);

    fireEvent.change(screen.getByPlaceholderText(/arXiv id or URL/i), {
      target: { value: "2412.19437" },
    });
    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await screen.findByText("already curated as deepseek-v3");
  });
});
