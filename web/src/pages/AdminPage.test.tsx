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

function stubFetch(addResponse: { ok: boolean; status?: number; body: unknown }) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/admin/status") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(status) } as Response);
    }
    if (url === "/api/admin/papers" && init?.method === "POST") {
      return Promise.resolve({
        ok: addResponse.ok,
        status: addResponse.status ?? (addResponse.ok ? 200 : 500),
        statusText: "",
        json: () => Promise.resolve(addResponse.body),
      } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function getTagsInput() {
  return (await screen.findByPlaceholderText(/paste or type arXiv/i)) as HTMLInputElement;
}

describe("AdminPage add paper", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("Enter turns typed text into a pill, and Add sends every pill", async () => {
    const fetchMock = stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.change(input, { target: { value: "2412.19437" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await screen.findByText("2412.19437"); // pill rendered

    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/admin/papers",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ arxiv_ids_or_urls: ["2412.19437"] }),
        }),
      ),
    );
    // Pills clear on success.
    await waitFor(() => expect(screen.queryByText("2412.19437")).not.toBeInTheDocument());
  });

  it("space and Tab also commit the current text as a pill", async () => {
    stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.change(input, { target: { value: "2401.00001" } });
    fireEvent.keyDown(input, { key: " " });
    await screen.findByText("2401.00001");

    fireEvent.change(input, { target: { value: "2401.00002" } });
    fireEvent.keyDown(input, { key: "Tab" });
    await screen.findByText("2401.00002");
  });

  it("pasting a multi-line list creates one pill per line, deduped", async () => {
    stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.paste(input, {
      clipboardData: { getData: () => "2401.00001\n2401.00002\n2401.00001\n" },
    });

    await screen.findByText("2401.00001");
    await screen.findByText("2401.00002");
    // Deduped: only one pill for the repeated id.
    expect(screen.getAllByText("2401.00001")).toHaveLength(1);
  });

  it("removing a pill excludes it from the submitted request", async () => {
    const fetchMock = stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.change(input, { target: { value: "2401.00001" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.change(input, { target: { value: "2401.00002" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await screen.findByText("2401.00002");

    // The pill's remove ("x") button is aria-hidden (Mantine's Pill marks it
    // decorative), so it's found by DOM structure, not an accessible role/name.
    const pillLabel = screen.getByText("2401.00001");
    const removeButton = pillLabel.parentElement?.querySelector("button");
    expect(removeButton).toBeTruthy();
    fireEvent.click(removeButton!);
    await waitFor(() => expect(screen.queryByText("2401.00001")).not.toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/admin/papers",
        expect.objectContaining({
          body: JSON.stringify({ arxiv_ids_or_urls: ["2401.00002"] }),
        }),
      ),
    );
  });

  it("renders a per-line status row for each result", async () => {
    stubFetch({
      ok: true,
      body: {
        results: [
          { input: "2401.00001", status: "queued", name: "2401.00001" },
          { input: "2412.19437", status: "duplicate", existing_name: "deepseek-v3" },
          { input: "not-an-id", status: "invalid" },
        ],
      },
    });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.paste(input, {
      clipboardData: { getData: () => "2401.00001\n2412.19437\nnot-an-id" },
    });
    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await screen.findByText(/already curated as deepseek-v3/i);
    await screen.findByText(/not a recognizable arXiv id or URL/i);
  });

  it("shows a whole-request failure without crashing", async () => {
    stubFetch({ ok: false, status: 500, body: { error: "boom" } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.change(input, { target: { value: "2401.00001" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await screen.findByText("boom");
  });

  it("disables the Add button until at least one pill exists", async () => {
    stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    expect(screen.getByRole("button", { name: /add paper/i })).toBeDisabled();

    fireEvent.change(input, { target: { value: "2401.00001" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(screen.getByRole("button", { name: /add paper/i })).toBeEnabled());
  });

  it("submits typed text that was never turned into a pill", async () => {
    // Regression: typing an id then clicking Add directly (the most natural path,
    // skipping Enter/Tab/Space/comma) must not be a silent no-op.
    const fetchMock = stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.change(input, { target: { value: "2401.00001" } });
    await waitFor(() => expect(screen.getByRole("button", { name: /add paper/i })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/admin/papers",
        expect.objectContaining({
          body: JSON.stringify({ arxiv_ids_or_urls: ["2401.00001"] }),
        }),
      ),
    );
  });

  it("submits pending text alongside already-committed pills", async () => {
    const fetchMock = stubFetch({ ok: true, body: { results: [] } });
    renderAdminPage();
    const input = await getTagsInput();

    fireEvent.change(input, { target: { value: "2401.00001" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.change(input, { target: { value: "2401.00002" } }); // typed, never committed
    fireEvent.click(screen.getByRole("button", { name: /add paper/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/admin/papers",
        expect.objectContaining({
          body: JSON.stringify({ arxiv_ids_or_urls: ["2401.00001", "2401.00002"] }),
        }),
      ),
    );
  });
});
