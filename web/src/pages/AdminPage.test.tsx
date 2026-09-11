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

interface Suggested {
  arxiv_id: string;
  cited_by: number;
  label: string;
}

function stubFetchWithSuggested(suggested: Suggested[], addResults: unknown[] = []) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/admin/status") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(status) } as Response);
    }
    if (url === "/api/admin/suggested") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(suggested) } as Response);
    }
    if (url === "/api/admin/papers" && init?.method === "POST") {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ results: addResults }),
      } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function stubSuggestedWithStatus(suggested: Suggested[], getStatus: () => unknown) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/admin/status") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(getStatus()) } as Response);
    }
    if (url === "/api/admin/suggested") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(suggested) } as Response);
    }
    if (url === "/api/admin/papers" && init?.method === "POST") {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ results: [] }),
      } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("AdminPage suggested papers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("adds only the checked suggestion ids", async () => {
    const fetchMock = stubFetchWithSuggested([
      { arxiv_id: "3333.33333", cited_by: 2, label: "Foo et al. Title one." },
      { arxiv_id: "4444.44444", cited_by: 1, label: "Bar et al. Title two." },
    ]);
    renderAdminPage();

    await screen.findByText(/suggested from your pool/i);
    expect(screen.getByRole("button", { name: /add 0 selected/i })).toBeDisabled();

    fireEvent.click(screen.getByLabelText("Select 3333.33333"));
    fireEvent.click(await screen.findByRole("button", { name: /add 1 selected/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/admin/papers",
        expect.objectContaining({
          body: JSON.stringify({ arxiv_ids_or_urls: ["3333.33333"] }),
        }),
      ),
    );
  });

  it("greys a just-added suggestion immediately, before pending catches up", async () => {
    // Anti-flicker: right after the click, `pending` is still empty (the poll hasn't
    // landed), but the row must not read as selectable. `justAdded` covers the gap.
    stubSuggestedWithStatus(
      [{ arxiv_id: "3333.33333", cited_by: 2, label: "Foo et al." }],
      () => status, // pending stays [] throughout
    );
    renderAdminPage();
    await screen.findByText(/suggested from your pool/i);

    const checkbox = () => screen.getByLabelText("Select 3333.33333");
    expect(checkbox()).not.toBeDisabled();

    fireEvent.click(checkbox());
    fireEvent.click(await screen.findByRole("button", { name: /add 1 selected/i }));

    await waitFor(() => expect(checkbox()).toBeDisabled()); // greyed via justAdded, not pending
  });

  it("hides the section when nothing is suggested", async () => {
    stubFetchWithSuggested([]);
    renderAdminPage();

    await screen.findByText(/^Add paper$/);
    expect(screen.queryByText(/suggested from your pool/i)).not.toBeInTheDocument();
  });

  it("greys out and blocks a suggestion still queued for ingestion", async () => {
    // `pending` lists an added-but-not-yet-ingested paper by name (= its arxiv_id).
    const pendingStatus = { ...status, pending: ["3333.33333"] };
    stubSuggestedWithStatus(
      [{ arxiv_id: "3333.33333", cited_by: 2, label: "Foo et al. Title." }],
      () => pendingStatus,
    );
    renderAdminPage();

    await screen.findByText(/suggested from your pool/i);
    expect(screen.getByLabelText("Select 3333.33333")).toBeDisabled();
    await screen.findByText(/^queued$/i); // the row shows a "queued" badge, not the id
  });

  it("refetches suggestions and scrolls to top when an ingestion run completes", async () => {
    vi.useFakeTimers();
    // jsdom doesn't implement Element.scrollTo — provide a spy so the reset is observable.
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      value: scrollTo,
      writable: true,
      configurable: true,
    });
    let state = "running";
    const fetchMock = stubSuggestedWithStatus(
      [{ arxiv_id: "3333.33333", cited_by: 1, label: "Foo et al." }],
      () => ({ ...status, ingestion: { ...status.ingestion, state } }),
    );
    try {
      renderAdminPage();
      await vi.runOnlyPendingTimersAsync(); // mount: status(running) + suggested
      const suggestedCalls = () =>
        fetchMock.mock.calls.filter(([u]) => String(u) === "/api/admin/suggested").length;
      expect(suggestedCalls()).toBe(1);
      expect(scrollTo).not.toHaveBeenCalled(); // not on mount

      state = "idle"; // run finished; the next poll observes running -> idle
      await vi.advanceTimersByTimeAsync(1600);
      expect(suggestedCalls()).toBe(2); // suggestions were refetched on completion
      expect(scrollTo).toHaveBeenCalledWith({ top: 0 }); // scroll box reset to top
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not refetch suggestions on the 1.5s status poll", async () => {
    vi.useFakeTimers();
    const fetchMock = stubFetchWithSuggested([
      { arxiv_id: "3333.33333", cited_by: 1, label: "Foo et al." },
    ]);
    try {
      renderAdminPage();
      await vi.runOnlyPendingTimersAsync(); // flush the mount fetches
      const suggestedCalls = () =>
        fetchMock.mock.calls.filter(([u]) => String(u) === "/api/admin/suggested").length;
      const statusCalls = () =>
        fetchMock.mock.calls.filter(([u]) => String(u) === "/api/admin/status").length;

      expect(suggestedCalls()).toBe(1);
      const statusBefore = statusCalls();
      await vi.advanceTimersByTimeAsync(5000); // several poll ticks

      expect(statusCalls()).toBeGreaterThan(statusBefore); // status kept polling
      expect(suggestedCalls()).toBe(1); // suggestions did not
    } finally {
      vi.useRealTimers();
    }
  });
});
