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
