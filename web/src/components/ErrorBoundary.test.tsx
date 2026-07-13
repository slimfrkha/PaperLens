import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import ErrorBoundary from "./ErrorBoundary";

function Boom(): never {
  throw new Error("kaboom");
}

describe("ErrorBoundary", () => {
  afterEach(cleanup);

  it("renders children when they don't throw", () => {
    render(
      <ErrorBoundary>
        <div>all good</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText("all good")).toBeTruthy();
  });

  it("renders the error message and a reload button when a child throws", () => {
    vi.spyOn(console, "error").mockImplementation(() => {}); // silence React's error log
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/kaboom/)).toBeTruthy();
    expect(screen.getByRole("button", { name: /reload/i })).toBeTruthy();
  });
});
