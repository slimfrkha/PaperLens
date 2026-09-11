import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import PaperPanel, { type HighlightTarget } from "./PaperPanel";

// jsdom lacks the CSS Custom Highlight API — minimal polyfill so the highlight-registration
// effects don't silently no-op (mirrors PaperViewer.test.tsx).
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

function mockFetch(
  annotations: unknown[] = [],
  extra?: (url: string, init?: RequestInit) => unknown,
) {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const handled = extra?.(url, init);
    if (handled !== undefined)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(handled) } as Response);
    if (url === "/api/papers/paper1")
      return Promise.resolve({ ok: true, json: () => Promise.resolve(paper) } as Response);
    if (url === "/api/papers/paper1/annotations")
      return Promise.resolve({ ok: true, json: () => Promise.resolve(annotations) } as Response);
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

function renderPanel(highlight: HighlightTarget | null = null, onNoteSaved?: () => void) {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <div style={{ height: 600 }}>
          <PaperPanel
            paperId="paper1"
            variant="panel"
            highlight={highlight}
            onNoteSaved={onNoteSaved}
          />
        </div>
      </MemoryRouter>
    </MantineProvider>,
  );
}

function selectParagraphText() {
  const paragraph = document.querySelector(".reading p")!;
  const range = document.createRange();
  range.selectNodeContents(paragraph);
  const sel = window.getSelection()!;
  sel.removeAllRanges();
  sel.addRange(range);
}

describe("PaperPanel (panel variant)", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    Range.prototype.getBoundingClientRect = () =>
      ({ x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }) as DOMRect;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    window.getSelection()?.removeAllRanges();
  });

  it("renders the paper and fetches its annotations", async () => {
    const fetchMock = mockFetch([]);
    vi.stubGlobal("fetch", fetchMock);
    renderPanel();

    await screen.findByText("Test Paper");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/papers/paper1/annotations"));
  });

  it("has no Contents control (the panel deliberately omits section nav)", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    renderPanel();

    await screen.findByText("Test Paper");
    expect(screen.queryByLabelText("Contents")).not.toBeInTheDocument();
  });

  it("Notes toggle opens an in-panel card listing a fetched note", async () => {
    const annotation = {
      id: "a1",
      snippet: "This is a long test passage worth annotating right here.",
      section_title: "Section One",
      section_slug: "section-one",
      note: "keep this one",
      created_at: "",
      updated_at: "",
    };
    vi.stubGlobal("fetch", mockFetch([annotation]));
    renderPanel();

    await screen.findByText("Test Paper");
    fireEvent.click(screen.getByLabelText("Notes"));
    expect(await screen.findByText("keep this one")).toBeInTheDocument();
  });

  it("applies the citation highlight from the `highlight` prop", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    renderPanel({ snippet: "This is a long test passage worth annotating right here." });

    await screen.findByText("Test Paper");
    await waitFor(() =>
      expect(
        (CSS as unknown as { highlights: { get: (n: string) => unknown } }).highlights.get(
          "citation",
        ),
      ).not.toBeUndefined(),
    );
  });

  it("selecting text and clicking Highlight posts the annotation and reports the save", async () => {
    const onNoteSaved = vi.fn();
    const fetchMock = mockFetch([], (url, init) =>
      url === "/api/papers/paper1/annotations" && init?.method === "POST"
        ? { id: "new1", ...JSON.parse(init.body as string), created_at: "", updated_at: "" }
        : undefined,
    );
    vi.stubGlobal("fetch", fetchMock);
    renderPanel(null, onNoteSaved);
    await screen.findByText("Test Paper");

    selectParagraphText();
    fireEvent.click(await screen.findByText("Highlight"));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/paper1/annotations",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    await waitFor(() => expect(onNoteSaved).toHaveBeenCalled());
  });
});
