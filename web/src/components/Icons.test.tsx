import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { IconSidebar } from "./Icons";

describe("IconSidebar", () => {
  it("renders an svg sized by the size prop", () => {
    const { container } = render(<IconSidebar size={24} />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("width", "24");
    expect(svg).toHaveAttribute("height", "24");
  });

  it("defaults to size 18 when no size prop is given", () => {
    const { container } = render(<IconSidebar />);
    expect(container.querySelector("svg")).toHaveAttribute("width", "18");
  });
});
