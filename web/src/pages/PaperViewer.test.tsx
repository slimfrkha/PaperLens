import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { getAnnotationRange } from "../highlight";
import PaperViewer from "./PaperViewer";

// jsdom doesn't implement the CSS Custom Highlight API — install a minimal polyfill so
// PaperViewer's highlight-registration effects don't silently no-op.
class FakeHighlight {
  ranges = new Set<Range>();
  constructor(...ranges: Range[]) {
    ranges.forEach((r) => this.ranges.add(r));
  }
  add(range: Range) {
    this.ranges.add(range);
    return this;
  }
  delete(range: Range) {
    return this.ranges.delete(range);
  }
  clear() {
    this.ranges.clear();
  }
}

beforeAll(() => {
  (window as unknown as { Highlight: typeof FakeHighlight }).Highlight = FakeHighlight;
  const registry = new Map<string, FakeHighlight>();
  (CSS as unknown as { highlights: unknown }).highlights = {
    set(name: string, hl: FakeHighlight) {
      registry.set(name, hl);
      return this;
    },
    delete: (name: string) => registry.delete(name),
    get: (name: string) => registry.get(name),
  };
});

const paper = {
  paper_id: "paper1",
  title: "Test Paper",
  tags: [],
  arxiv_id: undefined,
  markdown: "## Section One\n\nThis is a long test passage worth annotating right here.",
};

const paper2 = {
  paper_id: "paper2",
  title: "Second Paper",
  tags: [],
  arxiv_id: undefined,
  markdown: "## Intro\n\nA completely different paper with its own unrelated content.",
};

function mockFetch(
  annotations: unknown[] = [],
  extra?: (url: string, init?: RequestInit) => unknown,
) {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const handled = extra?.(url, init);
    if (handled !== undefined)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(handled) } as Response);
    if (url === "/api/papers/paper1") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(paper) } as Response);
    }
    if (url === "/api/papers/paper1/annotations") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(annotations) } as Response);
    }
    if (url === "/api/papers/paper2") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(paper2) } as Response);
    }
    if (url === "/api/papers/paper2/annotations") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

function renderPaperViewer() {
  return render(
    <MantineProvider>
      <MemoryRouter initialEntries={["/papers/paper1"]}>
        <Routes>
          <Route path="/papers/:id" element={<PaperViewer />} />
        </Routes>
      </MemoryRouter>
    </MantineProvider>,
  );
}

// Navigates in-app (same component instance, same as clicking a paper link elsewhere in
// the SPA) rather than remounting — the only way to exercise a paper-to-paper `id` change
// against module-level highlight state instead of trivially passing via a fresh mount.
function NavigateTo({ to }: { to: string }) {
  const navigate = useNavigate();
  return <button onClick={() => navigate(to)}>go</button>;
}

function renderPaperViewerWithNav() {
  return render(
    <MantineProvider>
      <MemoryRouter initialEntries={["/papers/paper1"]}>
        <NavigateTo to="/papers/paper2" />
        <Routes>
          <Route path="/papers/:id" element={<PaperViewer />} />
        </Routes>
      </MemoryRouter>
    </MantineProvider>,
  );
}

// Surfaces location.state.highlight as text, so a test can assert PaperViewer's
// citation-highlight effect actually clears it after consuming it (not just that the
// highlight got applied) — the state itself isn't otherwise observable from outside.
function LocationStateProbe() {
  const location = useLocation() as { state?: { highlight?: string } };
  return <div data-testid="loc-state">{location.state?.highlight ?? "none"}</div>;
}

function renderPaperViewerWithHighlightState(snippet: string) {
  return render(
    <MantineProvider>
      <MemoryRouter
        initialEntries={[
          { pathname: "/papers/paper1", state: { highlight: snippet, section: "Section One" } },
        ]}
      >
        <LocationStateProbe />
        <Routes>
          <Route path="/papers/:id" element={<PaperViewer />} />
        </Routes>
      </MemoryRouter>
    </MantineProvider>,
  );
}

// Selects the reading pane's paragraph text — jsdom doesn't synthesize selection from
// mouse events, so the Selection API is driven directly, matching how a real drag-select
// ends up. jsdom dispatches `selectionchange` itself (asynchronously, via a timer) once
// `addRange` runs; PaperViewer's listener picks that up without any manual event needed —
// firing one manually here as well double-fires it and races the popover's `seq`/key.
function selectParagraphText() {
  const paragraph = document.querySelector(".reading p")!;
  const range = document.createRange();
  range.selectNodeContents(paragraph);
  const sel = window.getSelection()!;
  sel.removeAllRanges();
  sel.addRange(range);
}

describe("PaperViewer annotations", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    // jsdom doesn't implement layout, and doesn't even stub Range.getBoundingClientRect —
    // it's simply missing. Real browsers implement it fully; only the coordinates matter
    // here, and this test never asserts on them.
    Range.prototype.getBoundingClientRect = () =>
      ({ x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }) as DOMRect;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    window.getSelection()?.removeAllRanges();
  });

  it("fetches the paper's annotations alongside the paper itself", async () => {
    const fetchMock = mockFetch([]);
    vi.stubGlobal("fetch", fetchMock);
    renderPaperViewer();

    await screen.findByText("Test Paper");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/papers/paper1/annotations"));
  });

  it("shows a floating hint to select text when the paper has no annotations yet", async () => {
    // Deliberately not asserting anything about scroll position or fixed placement —
    // jsdom has no layout — just that the hint (rendered via Affix, a portal) exists and
    // survives independent of the reading pane's DOM, unlike an inline hint would.
    vi.stubGlobal("fetch", mockFetch([]));
    renderPaperViewer();

    await screen.findByText("Test Paper");
    expect(
      await screen.findByText(/select any passage in the paper to highlight it or add a note/i),
    ).toBeInTheDocument();
  });

  it("hides the hint and shows a count badge once the paper has an annotation", async () => {
    const annotation = {
      id: "a1",
      snippet: "This is a long test passage worth annotating right here.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "",
      created_at: "",
      updated_at: "",
    };
    vi.stubGlobal("fetch", mockFetch([annotation]));
    renderPaperViewer();

    await screen.findByText("Test Paper");
    expect(screen.queryByText(/select any passage in the paper/i)).not.toBeInTheDocument();
    expect(await screen.findByText("1")).toBeInTheDocument(); // the Notes icon's count badge
  });

  it("Notes rail shows an empty state with no annotations", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    renderPaperViewer();

    await screen.findByText("Test Paper");
    fireEvent.click(screen.getByLabelText("Notes"));

    expect(await screen.findByText(/no annotations yet/i)).toBeInTheDocument();
  });

  it("Notes rail lists a fetched annotation's note", async () => {
    const annotation = {
      id: "a1",
      snippet: "This is a long test passage worth annotating right here.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "remember this for later",
      created_at: "",
      updated_at: "",
    };
    vi.stubGlobal("fetch", mockFetch([annotation]));
    renderPaperViewer();

    await screen.findByText("Test Paper");
    fireEvent.click(screen.getByLabelText("Notes"));

    expect(await screen.findByText("remember this for later")).toBeInTheDocument();
  });

  it("selecting text opens the popover; Highlight posts an empty-note annotation", async () => {
    const fetchMock = mockFetch([], (url, init) =>
      url === "/api/papers/paper1/annotations" && init?.method === "POST"
        ? { id: "new1", ...JSON.parse(init.body as string), created_at: "", updated_at: "" }
        : undefined,
    );
    vi.stubGlobal("fetch", fetchMock);
    renderPaperViewer();
    await screen.findByText("Test Paper");

    selectParagraphText();
    fireEvent.click(await screen.findByText("Highlight"));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/paper1/annotations",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            snippet: "This is a long test passage worth annotating right here.",
            section_title: "Section One",
            section_slug: "section-one",
            note: "",
          }),
        }),
      ),
    );
  });

  it("selecting text, Add note, then Save posts the typed note", async () => {
    const fetchMock = mockFetch([], (url, init) =>
      url === "/api/papers/paper1/annotations" && init?.method === "POST"
        ? { id: "new1", ...JSON.parse(init.body as string), created_at: "", updated_at: "" }
        : undefined,
    );
    vi.stubGlobal("fetch", fetchMock);
    renderPaperViewer();
    await screen.findByText("Test Paper");

    selectParagraphText();
    fireEvent.click(await screen.findByText("Add note"));
    const textarea = await screen.findByPlaceholderText("Add a note...");
    fireEvent.change(textarea, { target: { value: "a note on this passage" } });
    fireEvent.click(screen.getByText("Save"));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/paper1/annotations",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            snippet: "This is a long test passage worth annotating right here.",
            section_title: "Section One",
            section_slug: "section-one",
            note: "a note on this passage",
          }),
        }),
      ),
    );
  });

  it("deleting an annotation from its popover removes it from the Notes rail", async () => {
    const annotation = {
      id: "a1",
      snippet: "This is a long test passage worth annotating right here.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "a note to delete",
      created_at: "",
      updated_at: "",
    };
    const fetchMock = mockFetch([annotation], (url, init) =>
      url === "/api/papers/paper1/annotations/a1" && init?.method === "DELETE"
        ? { ok: true }
        : undefined,
    );
    vi.stubGlobal("fetch", fetchMock);
    renderPaperViewer();
    await screen.findByText("Test Paper");

    // Wait for the annotation to be fetched (and its highlight registered) before
    // simulating a click on it.
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/papers/paper1/annotations"));

    // Click the annotated passage (jsdom's caretRangeFromPoint isn't implemented; stub it
    // to land inside the registered range, the same seam highlight.test.ts exercises).
    const paragraph = document.querySelector(".reading p")!;
    document.caretRangeFromPoint = () => {
      const r = document.createRange();
      r.setStart(paragraph.firstChild!, 5);
      r.setEnd(paragraph.firstChild!, 5);
      return r;
    };
    fireEvent.click(paragraph, { clientX: 1, clientY: 1 });

    fireEvent.click(await screen.findByText("Delete"));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/paper1/annotations/a1",
        expect.objectContaining({ method: "DELETE" }),
      ),
    );

    fireEvent.click(screen.getByLabelText("Notes"));
    expect(await screen.findByText(/no annotations yet/i)).toBeInTheDocument();
  });

  it("consumes the deep-link highlight from location.state so a refresh can't re-apply it", async () => {
    // Regression: the browser preserves history.state (this navigation's {highlight,
    // section}) across a native page refresh, unlike component state — without clearing
    // it once applied, refreshing the page re-triggers the same highlight indefinitely,
    // even reconstructing it for a passage whose annotation has since been deleted.
    const snippet = "This is a long test passage worth annotating right here.";
    vi.stubGlobal("fetch", mockFetch([]));
    renderPaperViewerWithHighlightState(snippet);

    await screen.findByText("Test Paper");
    expect(screen.getByTestId("loc-state")).toHaveTextContent(snippet);
    await waitFor(() => expect(screen.getByTestId("loc-state")).toHaveTextContent("none"));
  });

  it("deleting an annotation also clears any active citation highlight on the same passage", async () => {
    // Regression: navigating here via a Notes-page/citation click-through registers a
    // separate transient "citation" highlight (independent of the annotation's own
    // persistent one) — deleting the annotation used to leave that highlight showing.
    const annotation = {
      id: "a1",
      snippet: "This is a long test passage worth annotating right here.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "",
      created_at: "",
      updated_at: "",
    };
    const fetchMock = mockFetch([annotation], (url, init) =>
      url === "/api/papers/paper1/annotations/a1" && init?.method === "DELETE"
        ? { ok: true }
        : undefined,
    );
    vi.stubGlobal("fetch", fetchMock);
    renderPaperViewerWithHighlightState(annotation.snippet);
    await screen.findByText("Test Paper");

    await waitFor(() =>
      expect(
        (CSS as unknown as { highlights: { get: (n: string) => unknown } }).highlights.get(
          "citation",
        ),
      ).not.toBeUndefined(),
    );

    const paragraph = document.querySelector(".reading p")!;
    document.caretRangeFromPoint = () => {
      const r = document.createRange();
      r.setStart(paragraph.firstChild!, 5);
      r.setEnd(paragraph.firstChild!, 5);
      return r;
    };
    fireEvent.click(paragraph, { clientX: 1, clientY: 1 });
    fireEvent.click(await screen.findByText("Delete"));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/paper1/annotations/a1",
        expect.objectContaining({ method: "DELETE" }),
      ),
    );
    expect(
      (CSS as unknown as { highlights: { get: (n: string) => unknown } }).highlights.get(
        "citation",
      ),
    ).toBeUndefined();
  });

  it("navigating to a different paper clears the previous paper's highlight registrations", async () => {
    // Regression: highlight.ts's registry is module-level, not component state — React
    // Router reuses this same PaperViewer instance across an in-app navigation (it never
    // unmounts), so without an explicit clear, paper1's stale Range would linger forever.
    const annotation = {
      id: "a1",
      snippet: "This is a long test passage worth annotating right here.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "",
      created_at: "",
      updated_at: "",
    };
    vi.stubGlobal("fetch", mockFetch([annotation]));
    renderPaperViewerWithNav();

    await screen.findByText("Test Paper");
    await waitFor(() => expect(getAnnotationRange("a1")).toBeDefined());

    fireEvent.click(screen.getByText("go"));

    await screen.findByText("Second Paper");
    expect(getAnnotationRange("a1")).toBeUndefined();
  });

  it("shows an unmatched annotation as not found, reactively, without a coincidental re-render", async () => {
    const staleAnnotation = {
      id: "a1",
      snippet: "This exact phrase was never present in the rendered markdown at all.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "a note on stale text",
      created_at: "",
      updated_at: "",
    };
    vi.stubGlobal("fetch", mockFetch([staleAnnotation]));
    renderPaperViewer();

    await screen.findByText("Test Paper");
    fireEvent.click(screen.getByLabelText("Notes"));

    expect(await screen.findByText(/not found in current text/i)).toBeInTheDocument();
  });
});

describe("PaperViewer outline (Contents rail)", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // The shared `paper` fixture's markdown has only one heading, which the production
  // code treats as the paper's own title and strips from the outline (see
  // PaperViewer.tsx's outline-building effect) — so these tests need a second heading to
  // survive that strip and actually appear in the Contents drawer.
  function mockFetchWithHeadings() {
    return mockFetch([], (url) =>
      url === "/api/papers/paper1"
        ? { ...paper, markdown: "## Paper Title\n\nAbstract text.\n\n## Section One\n\nBody text." }
        : undefined,
    );
  }

  it("Contents toggle opens a drawer listing the paper's rendered headings", async () => {
    vi.stubGlobal("fetch", mockFetchWithHeadings());
    renderPaperViewer();
    await screen.findByText("Test Paper");

    fireEvent.click(screen.getByLabelText("Contents"));
    // Scoped by role, not text: the heading itself is still rendered (unscoped) in the
    // reading pane behind the drawer, so a plain text query would match both.
    expect(await screen.findByRole("button", { name: "Section One" })).toBeInTheDocument();
  });

  it("clicking a heading scrolls to it", async () => {
    vi.stubGlobal("fetch", mockFetchWithHeadings());
    renderPaperViewer();
    await screen.findByText("Test Paper");

    fireEvent.click(screen.getByLabelText("Contents"));
    fireEvent.click(await screen.findByRole("button", { name: "Section One" }));

    expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "start",
    });
  });

  it("opening the Contents rail hides the Notes toggle", async () => {
    vi.stubGlobal("fetch", mockFetchWithHeadings());
    renderPaperViewer();
    await screen.findByText("Test Paper");

    fireEvent.click(screen.getByLabelText("Contents"));
    expect(screen.queryByLabelText("Notes")).not.toBeInTheDocument();
  });

  it("opening the Notes rail hides the Contents toggle", async () => {
    vi.stubGlobal("fetch", mockFetchWithHeadings());
    renderPaperViewer();
    await screen.findByText("Test Paper");

    fireEvent.click(screen.getByLabelText("Notes"));
    expect(screen.queryByLabelText("Contents")).not.toBeInTheDocument();
  });
});
