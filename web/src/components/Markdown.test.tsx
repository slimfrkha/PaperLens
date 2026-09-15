import { MantineProvider } from "@mantine/core";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { normalizeLatexMath } from "../markdownMath";
import Answer from "./Answer";
import Markdown from "./Markdown";

function renderAnswer(source: string) {
  return render(
    <MantineProvider>
      <MemoryRouter>
        <Answer text={source} citations={[]} />
      </MemoryRouter>
    </MantineProvider>,
  );
}

describe("Markdown math rendering", () => {
  it("renders parenthesized inline math and bracketed display math", () => {
    const source = String.raw`Inline: \(\mathbf{x} \in \mathbb{R}^d\).

1. Rotation matrix

   \[
   R(\theta) = \begin{bmatrix}
   \cos \theta & -\sin \theta\\
   \sin \theta & \cos \theta
   \end{bmatrix}
   \]`;

    const { container } = renderAnswer(source);

    expect(container.querySelectorAll(".katex")).toHaveLength(2);
    expect(container.querySelectorAll(".katex-display")).toHaveLength(1);
  });

  it("renders blockquoted and same-line display delimiters", () => {
    const source = String.raw`> Quoted formula:
>
> \[
> q^2
> \]

Before \[x^2 + y^2\] after.`;

    const { container } = renderAnswer(source);

    expect(container.querySelectorAll(".katex-display")).toHaveLength(2);
    expect(container.querySelector(".katex-error")).not.toBeInTheDocument();
  });

  it("keeps existing dollar-delimited math working", () => {
    const source = String.raw`Inline: $x^2$.

$$
y^2
$$`;

    const { container } = render(<Markdown>{source}</Markdown>);

    expect(normalizeLatexMath(source)).toBe(source);
    expect(container.querySelectorAll(".katex")).toHaveLength(2);
    expect(container.querySelectorAll(".katex-display")).toHaveLength(1);
  });

  it("does not interpret delimiter-like text inside code", () => {
    const inlineExample = String.raw`\(inline_example\)`;
    const blockExample = String.raw`\[
block_example
\]`;
    const source = `Use \`${inlineExample}\` in prose.

~~~text
${blockExample}
~~~`;

    const { container } = render(<Markdown>{source}</Markdown>);

    expect(normalizeLatexMath(source)).toBe(source);
    expect(container.querySelector(".katex")).not.toBeInTheDocument();
    expect(container.querySelector("p code")?.textContent).toBe(inlineExample);
    expect(container.querySelector("pre")?.textContent).toContain(blockExample);
  });

  it("preserves indented code and fences nested in Markdown containers", () => {
    const source = String.raw`    \(indented_inline\)
    \[
    indented_display
    \]

> ~~~text
> \(quoted_fence\)
> ~~~

10. Example

    ~~~text
    \(list_fence\)
    ~~~`;

    expect(normalizeLatexMath(source)).toBe(source);
  });

  it("does not pair display delimiters across code", () => {
    const source = String.raw`\[
~~~text
\]
~~~
\]`;

    expect(normalizeLatexMath(source)).toBe(source);
  });

  it("leaves unmatched and escaped delimiters unchanged", () => {
    const source = String.raw`Streaming \(x + 1 and literal \\(y\\).`;

    expect(normalizeLatexMath(source)).toBe(source);
  });

  it("leaves MathJax delimiters alone in the shared paper renderer", () => {
    const source = String.raw`Paper example: \(literal\)`;
    const { container } = render(<Markdown>{source}</Markdown>);

    expect(container.querySelector(".katex")).not.toBeInTheDocument();
  });
});
