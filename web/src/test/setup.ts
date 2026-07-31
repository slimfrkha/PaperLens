import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(cleanup);

// jsdom lacks matchMedia; Mantine's color-scheme logic reads it. (Mantine's
// documented test shim.)
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }),
});

// jsdom lacks ResizeObserver; Mantine's ScrollArea reads it. (Mantine's documented test shim.)
window.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// jsdom implements no Clipboard API at all; Mantine's useClipboard (CopyButton) guards
// with `"clipboard" in navigator`, then chains `.then()` on the write — writeText must
// resolve, not just exist, or that chain throws on a bare mock's undefined return.
Object.defineProperty(navigator, "clipboard", {
  value: { writeText: vi.fn(() => Promise.resolve()) },
  configurable: true,
});
