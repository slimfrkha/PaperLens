import { unified } from "unified";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";

type PositionedNode = {
  type: string;
  position?: { start: { offset?: number }; end: { offset?: number } };
  children?: PositionedNode[];
};

type Replacement = { end: number; value: string };

const markdownParser = unified().use(remarkParse).use(remarkMath).freeze();

function protectedSyntax(markdown: string): Uint8Array {
  const protectedCharacters = new Uint8Array(markdown.length);
  const tree = markdownParser.parse(markdown) as PositionedNode;

  function visit(node: PositionedNode) {
    if (["code", "inlineCode", "math", "inlineMath"].includes(node.type)) {
      const start = node.position?.start.offset;
      const end = node.position?.end.offset;
      if (start !== undefined && end !== undefined) protectedCharacters.fill(1, start, end);
      return;
    }
    node.children?.forEach(visit);
  }

  visit(tree);
  return protectedCharacters;
}

function isUnescapedBackslash(markdown: string, index: number): boolean {
  let preceding = 0;
  for (let cursor = index - 1; cursor >= 0 && markdown[cursor] === "\\"; cursor -= 1) {
    preceding += 1;
  }
  return preceding % 2 === 0;
}

function containsProtected(mask: Uint8Array, start: number, end: number): boolean {
  for (let index = start; index < end; index += 1) {
    if (mask[index]) return true;
  }
  return false;
}

function findDelimiter(
  markdown: string,
  delimiter: "(" | ")" | "[" | "]",
  start: number,
  end: number,
  protectedCharacters: Uint8Array,
): number {
  for (let index = start; index < end - 1; index += 1) {
    if (
      !protectedCharacters[index] &&
      markdown[index] === "\\" &&
      markdown[index + 1] === delimiter &&
      isUnescapedBackslash(markdown, index)
    ) {
      return index;
    }
  }
  return -1;
}

function inlineDisplayReplacement(formula: string, prefix: string): string {
  if (!prefix) return `\n\n$$\n${formula}\n$$\n\n`;

  const blankPrefix = prefix.trimEnd();
  return `\n${blankPrefix}\n${prefix}$$\n${prefix}${formula}\n${prefix}$$\n${blankPrefix}\n${prefix}`;
}

/** Normalize complete MathJax-style delimiters in model-generated Markdown to
 * remark-math's dollar syntax. A CommonMark parse supplies protected code/math
 * ranges; this function never attempts to recognize Markdown constructs itself. */
export function normalizeLatexMath(markdown: string): string {
  if (!markdown.includes("\\(") && !markdown.includes("\\[")) return markdown;

  const protectedCharacters = protectedSyntax(markdown);
  const displayMath = new Uint8Array(markdown.length);
  const replacements = new Map<number, Replacement>();
  let blockOpen: { index: number; start: number; prefix: string } | undefined;

  // Convert display delimiters on their own lines, including blockquote and list
  // continuation prefixes. A protected code range cancels any pending pair.
  for (let start = 0; start < markdown.length;) {
    const newline = markdown.indexOf("\n", start);
    const end = newline === -1 ? markdown.length : newline + 1;
    if (containsProtected(protectedCharacters, start, end)) {
      blockOpen = undefined;
      start = end;
      continue;
    }

    const rawLine = markdown.slice(start, newline === -1 ? end : newline).replace(/\r$/, "");
    const match = rawLine.match(/^((?:(?: {0,3}>[ \t]?)+)?[ \t]*)(\\\[|\\\])[ \t]*$/);
    if (match) {
      const index = start + match[1].length;
      if (match[2] === "\\[") {
        blockOpen = { index, start, prefix: match[1] };
      } else if (blockOpen?.prefix === match[1]) {
        replacements.set(blockOpen.index, { end: blockOpen.index + 2, value: "$$" });
        replacements.set(index, { end: index + 2, value: "$$" });
        displayMath.fill(1, blockOpen.start, end);
        blockOpen = undefined;
      } else {
        blockOpen = undefined;
      }
    }

    start = end;
  }

  // A same-line display pair needs real flow boundaries; replacing it with
  // `$$...$$` on the same line would make remark-math treat it as inline math.
  for (let lineStart = 0; lineStart < markdown.length;) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline === -1 ? markdown.length : newline;
    let cursor = lineStart;

    while (cursor < lineEnd) {
      const open = findDelimiter(markdown, "[", cursor, lineEnd, protectedCharacters);
      if (open === -1) break;
      const close = findDelimiter(markdown, "]", open + 2, lineEnd, protectedCharacters);
      if (close === -1) break;
      if (!containsProtected(protectedCharacters, open, close + 2) && !displayMath[open]) {
        const before = markdown.slice(lineStart, open);
        const prefix = before.match(/^((?:(?: {0,3}>[ \t]?)+)?[ \t]*)/)?.[1] ?? "";
        replacements.set(open, {
          end: close + 2,
          value: inlineDisplayReplacement(markdown.slice(open + 2, close), prefix),
        });
        displayMath.fill(1, open, close + 2);
      }
      cursor = close + 2;
    }

    lineStart = newline === -1 ? markdown.length : newline + 1;
  }

  // Inline delimiters may cross line endings, but never protected code, existing
  // dollar math, or display math.
  for (let start = 0; start < markdown.length;) {
    const open = findDelimiter(markdown, "(", start, markdown.length, protectedCharacters);
    if (open === -1) break;
    const close = findDelimiter(markdown, ")", open + 2, markdown.length, protectedCharacters);
    if (close === -1) break;

    if (
      !containsProtected(protectedCharacters, open, close + 2) &&
      !containsProtected(displayMath, open, close + 2)
    ) {
      replacements.set(open, { end: open + 2, value: "$" });
      replacements.set(close, { end: close + 2, value: "$" });
    }
    start = close + 2;
  }

  if (replacements.size === 0) return markdown;

  let normalized = "";
  for (let index = 0; index < markdown.length;) {
    const replacement = replacements.get(index);
    if (replacement) {
      normalized += replacement.value;
      index = replacement.end;
    } else {
      normalized += markdown[index];
      index += 1;
    }
  }
  return normalized;
}
