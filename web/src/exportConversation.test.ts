import { describe, expect, it } from "vitest";
import type { Citation, CompareRow, StoredTurn, TraceEntry } from "./api";
import { conversationToMarkdown } from "./exportConversation";

function citation(overrides: Partial<Citation> & { ref: string; paper_id: string }): Citation {
  return {
    title: "A Paper",
    breadcrumb: "b",
    section_title: "Method",
    snippet: "sn",
    ...overrides,
  };
}

function turn(overrides: Partial<StoredTurn> & { question: string }): StoredTurn {
  return {
    answer: "",
    citations: [],
    trace: [],
    usage: null,
    feedback: null,
    per_paper: false,
    compare: false,
    compare_results: null,
    auto: false,
    ...overrides,
  };
}

describe("conversationToMarkdown", () => {
  it("emits a title, a section per turn, the question, and one References block", () => {
    const turns = [
      turn({
        question: "What is X?",
        answer: "X is a thing [r1].",
        citations: [citation({ ref: "r1", paper_id: "p1", arxiv_id: "2401.00001" })],
      }),
      turn({ question: "And Y?", answer: "Y is small talk." }),
    ];
    const md = conversationToMarkdown(turns, "My Chat");

    expect(md).toContain("# My Chat");
    expect(md).toContain("## Turn 1");
    expect(md).toContain("## Turn 2");
    expect(md).toContain("**Q:** What is X?");
    expect(md).toContain("**Q:** And Y?");
    // Turns are separated by a thematic break.
    expect(md).toContain("\n\n---\n\n");
    // Exactly one consolidated References section, at the end.
    expect(md.match(/## References/g)).toHaveLength(1);
    expect(md).toContain("[^t1-1]: **A Paper** — Method, https://arxiv.org/abs/2401.00001");
  });

  it("namespaces footnotes per turn so the same ref number can't collide", () => {
    const turns = [
      turn({
        question: "Q1",
        answer: "First [r1].",
        citations: [citation({ ref: "r1", paper_id: "p1", title: "Paper One" })],
      }),
      turn({
        question: "Q2",
        answer: "Second [r1].",
        citations: [citation({ ref: "r1", paper_id: "p2", title: "Paper Two" })],
      }),
    ];
    const md = conversationToMarkdown(turns, "Chat");

    // Distinct markers, no bare duplicated [^1].
    expect(md).toContain("First [^t1-1].");
    expect(md).toContain("Second [^t2-1].");
    expect(md).not.toContain("[^1]");
    // Both definitions present in the single References block, blank-line separated.
    expect(md).toContain("[^t1-1]: **Paper One**");
    expect(md).toContain("[^t2-1]: **Paper Two**");
    expect(md).toMatch(/\[\^t1-1\]:[^\n]*\n\n\[\^t2-1\]:/);
  });

  it("renders the whole trace as one code block inside the collapsible", () => {
    const trace: TraceEntry[] = [
      { type: "thought", text: "I should search." },
      { type: "action", query: "attention", paper: "p1" },
      { type: "observation", text: "Found it." },
    ];
    const md = conversationToMarkdown([turn({ question: "Q", answer: "A", trace })], "Chat");

    expect(md).toContain("<details>\n<summary>Reasoning</summary>\n\n```");
    expect(md).toContain(
      'Thought: I should search.\n\nAction: search "attention" (p1)\n\nObservation:\nFound it.',
    );
  });

  it("shows both paper and per-paper scope on an action", () => {
    const trace: TraceEntry[] = [{ type: "action", query: "q", paper: "p1", per_paper: true }];
    const md = conversationToMarkdown([turn({ question: "Q", answer: "A", trace })], "Chat");
    expect(md).toContain('Action: search "q" (p1) (per-paper)');
  });

  it("keeps raw observation text (HTML-like, math, [rN]) verbatim in the code block", () => {
    const trace: TraceEntry[] = [
      { type: "observation", text: "sees <problem, response> and $$x$$ and [r9]" },
    ];
    const md = conversationToMarkdown([turn({ question: "Q", answer: "A", trace })], "Chat");
    expect(md).toContain("Observation:\nsees <problem, response> and $$x$$ and [r9]");
  });

  it("lengthens the fence past any backtick run in the trace text", () => {
    const trace: TraceEntry[] = [{ type: "observation", text: "code ```py\nx\n```" }];
    const md = conversationToMarkdown([turn({ question: "Q", answer: "A", trace })], "Chat");
    expect(md).toContain("````\nObservation:\ncode ```py\nx\n```\n````");
  });

  it("keeps a literal </details> in trace text inside the code block, unable to break out", () => {
    const trace: TraceEntry[] = [{ type: "thought", text: "closing </details> now" }];
    const md = conversationToMarkdown([turn({ question: "Q", answer: "A", trace })], "Chat");
    // Verbatim inside the reasoning fence — a code block escapes it on render, so it can't
    // close the surrounding <details>.
    const openFence = md.indexOf("```", md.indexOf("<summary>Reasoning</summary>"));
    const fenceBody = md.slice(openFence, md.indexOf("```", openFence + 3));
    expect(fenceBody).toContain("Thought: closing </details> now");
  });

  it("orders the References numerically, not by inline-citation order", () => {
    const turns = [
      turn({
        question: "Q",
        answer: "See [r3] then [r1].",
        citations: [
          citation({ ref: "r3", paper_id: "p3", title: "Three" }),
          citation({ ref: "r1", paper_id: "p1", title: "One" }),
        ],
      }),
    ];
    const refs = conversationToMarkdown(turns, "Chat").split("## References")[1];
    expect(refs.indexOf("[^t1-1]:")).toBeLessThan(refs.indexOf("[^t1-3]:"));
  });

  it("omits the reasoning block when a turn has no trace", () => {
    const md = conversationToMarkdown([turn({ question: "Q", answer: "A" })], "Chat");
    expect(md).not.toContain("<details>");
  });

  it("handles bunched, unresolved, and repeated markers within a turn", () => {
    const turns = [
      turn({
        question: "Q",
        answer: "Both [r1, r2] agree, but [r9] is unknown, and again [r1].",
        citations: [
          citation({ ref: "r1", paper_id: "p1", title: "One" }),
          citation({ ref: "r2", paper_id: "p2", title: "Two" }),
        ],
      }),
    ];
    const md = conversationToMarkdown(turns, "Chat");
    expect(md).toContain("Both [^t1-1][^t1-2] agree, but [r9] is unknown, and again [^t1-1].");
    // One definition per distinct cited ref, and the unresolved ref never gets one.
    expect(md.match(/\[\^t1-1\]:/g)).toHaveLength(1);
    expect(md).toContain("[^t1-1]: **One**");
    expect(md).toContain("[^t1-2]: **Two**");
    expect(md).not.toContain("[^t1-9]");
  });

  it("rewrites fullwidth CJK 【rN】 markers the same as ASCII ones", () => {
    const turns = [
      turn({
        question: "Q",
        answer: "Value【r1】here.",
        citations: [citation({ ref: "r1", paper_id: "p1", title: "One" })],
      }),
    ];
    const md = conversationToMarkdown(turns, "Chat");
    expect(md).toContain("Value[^t1-1]here.");
    expect(md).toContain("[^t1-1]: **One**");
  });

  it("omits References when no turn cites anything", () => {
    const md = conversationToMarkdown([turn({ question: "Q", answer: "Just prose." })], "Chat");
    expect(md).not.toContain("## References");
  });

  it("keeps the synthesis answer visible and collapses each compared paper", () => {
    const rows: CompareRow[] = [
      {
        paper_id: "p1",
        title: "Paper One",
        arxiv_id: "2401.00001",
        text: "One says [r1].",
        citations: [
          citation({
            ref: "r1",
            paper_id: "p1",
            title: "Paper One",
            section_title: "S1",
            snippet: "snip1",
          }),
        ],
        trace: [{ type: "thought", text: "row one thinking" }],
      },
      {
        paper_id: "p2",
        title: "Paper Two",
        arxiv_id: null,
        text: "Two says [r1].",
        citations: [citation({ ref: "r1", paper_id: "p2", title: "Paper Two" })],
        trace: [],
      },
    ];
    const md = conversationToMarkdown(
      [
        turn({
          question: "Compare them",
          answer: "Synthesis across both.",
          compare: true,
          compare_results: rows,
        }),
      ],
      "Chat",
    );

    // Synthesis answer stays as the turn's visible content.
    expect(md).toContain("Synthesis across both.");
    // Each paper is a collapsed section, not a rendered heading.
    expect(md).toContain("<summary>Paper One</summary>");
    expect(md).toContain("<summary>Paper Two</summary>");
    expect(md).not.toContain("### Paper One");
    // The per-paper answer is kept verbatim inside the fence (raw [rN], not footnoted).
    expect(md).toContain("One says [r1].");
    expect(md).not.toContain("[^t1-1-1]");
    // Row one folds in its own reasoning and sources; row two (empty trace) has no Reasoning.
    expect(md).toContain("Reasoning:\nThought: row one thinking");
    expect(md).toContain("Sources:\nPaper One");
  });

  it("adds a collapsed Sources block with cited passages grouped by paper", () => {
    const turns = [
      turn({
        question: "Q",
        answer: "Answer [r1] and [r2].",
        citations: [
          citation({
            ref: "r1",
            paper_id: "p1",
            title: "Paper One",
            arxiv_id: "2401.00001",
            section_title: "Intro",
            snippet: "first passage",
          }),
          citation({
            ref: "r2",
            paper_id: "p1",
            title: "Paper One",
            section_title: "Method",
            snippet: "second passage",
          }),
        ],
      }),
    ];
    const md = conversationToMarkdown(turns, "Chat");
    expect(md).toContain("<details>\n<summary>Sources</summary>\n\n```");
    // Both refs group under one paper heading with its arXiv link, as plain code text.
    expect(md).toContain("Paper One — https://arxiv.org/abs/2401.00001");
    expect(md).toContain("[1] Intro\nfirst passage");
    expect(md).toContain("[2] Method\nsecond passage");
  });

  it("flags a source whose faithfulness check isn't a clean support", () => {
    const turns = [
      turn({
        question: "Q",
        answer: "Claim [r1].",
        citations: [
          citation({
            ref: "r1",
            paper_id: "p1",
            section_title: "S",
            snippet: "x",
            faithfulness: [{ sentence: "Claim [r1].", label: "neutral", score: 0.1 }],
          }),
        ],
      }),
    ];
    expect(conversationToMarkdown(turns, "Chat")).toContain(
      "⚠️ doesn't clearly support this claim",
    );
  });

  it("escapes a line-leading code fence in a question so it can't open a code block", () => {
    const md = conversationToMarkdown(
      [turn({ question: "explain\n```\ncode", answer: "The answer." })],
      "Chat",
    );
    expect(md).toContain("\\```"); // fence escaped → no code block opens
    expect(md).toContain("The answer."); // answer not swallowed
  });

  it("shows only latency when token counts are missing, and a thumbs-down", () => {
    const md = conversationToMarkdown(
      [
        turn({
          question: "Q",
          answer: "A",
          usage: { input_tokens: null, output_tokens: null, latency_ms: 3400 },
          feedback: { vote: "down", note: null },
        }),
      ],
      "Chat",
    );
    expect(md).toContain("_3.4s_");
    expect(md).toContain("**Feedback:** 👎");
  });

  it("includes the usage line and the reader's feedback when present", () => {
    const turns = [
      turn({
        question: "Q",
        answer: "A",
        usage: { input_tokens: 100, output_tokens: 50, latency_ms: 2300 },
        feedback: { vote: "up", note: "helpful" },
      }),
    ];
    const md = conversationToMarkdown(turns, "Chat");
    expect(md).toContain("_150 tokens · 2.3s_");
    expect(md).toContain("**Feedback:** 👍 — helpful");
  });

  it("returns just the title for an empty conversation (no dangling References)", () => {
    expect(conversationToMarkdown([], "Empty")).toBe("# Empty");
  });
});
