import type {
  Citation,
  CompareRow,
  Feedback,
  PaperCitation,
  StoredTurn,
  TraceEntry,
  UsageInfo,
} from "./api";
import { isPaperCitation, isWebCitation } from "./api";
import { citedCitations, REF_ID, REF_MARKER, referenceLine, refNumber } from "./exportAnswer";
import { faithfulnessMessage, worstLabel } from "./faithfulness";

// A cited citation tagged with the footnote label of the turn (or compare row) it
// belongs to. Collected across the whole conversation so a single `## References`
// section at the document end can define each one — GFM hoists all footnote
// *definitions* to the bottom regardless of where they appear in source, so per-turn
// reference blocks would render as empty headings. The label is namespaced per turn
// (`t2-1`) so the same on-screen ref number in two different turns can't collide on a
// single `[^1]`.
interface RefEntry {
  label: string;
  citation: Citation;
}

/** Rewrites an answer's `[rN]` markers to namespaced `[^{prefix}{n}]` footnote markers,
 *  recording each resolved citation under its label in `out`. Mirrors the marker rewrite
 *  in `answerToMarkdown` — bunched `[r1, r2]` and fullwidth `【rN】` handled the same way,
 *  unresolved brackets left unchanged — but namespaces the label so footnotes from
 *  different turns don't clash in one document. */
function rewriteRefs(text: string, cited: Citation[], prefix: string, out: RefEntry[]): string {
  const byRef = new Map(cited.map((c) => [c.ref, c]));
  return text.replace(REF_MARKER, (m, group: string) => {
    const notes = [...group.matchAll(REF_ID)]
      .map((refMatch) => byRef.get(refMatch[0]))
      .filter((c): c is Citation => !!c)
      .map((c) => {
        const label = `${prefix}${refNumber(c)}`;
        if (!out.some((e) => e.label === label)) out.push({ label, citation: c });
        return `[^${label}]`;
      });
    return notes.length > 0 ? notes.join("") : m;
  });
}

/** Wraps `text` in a fenced code block, using a backtick run one longer than the longest
 *  run inside `text` so the content can never close the fence early. */
function codeFence(text: string): string {
  const runs = text.match(/`+/g);
  const longest = runs ? Math.max(...runs.map((r) => r.length)) : 0;
  const fence = "`".repeat(Math.max(3, longest + 1));
  return `${fence}\n${text}\n${fence}`;
}

// Neutralizes angle brackets so a plain-text string rendered as Markdown can't be parsed as
// HTML. Entities render back as `<`/`>`.
function escapeAngles(s: string): string {
  return s.replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Makes free text (a question, a feedback note) safe to drop into the Markdown flow as
 *  prose — the app shows these as plain text, so nothing pasted into them should alter the
 *  document. Escapes HTML angles, and backslash-escapes a leading block trigger on every line
 *  (heading, blockquote, list, thematic break, and — the dangerous one — a code fence, whose
 *  unbalanced `` ``` `` would otherwise swallow the rest of the document). Natural prose has
 *  no line-leading trigger, so it is left untouched. */
function inlineText(s: string): string {
  return escapeAngles(s)
    .split("\n")
    .map((line) =>
      line
        .replace(/^(\s{0,3})([#>`~*+\-=_])/, "$1\\$2")
        .replace(/^(\s{0,3}\d+)([.)])(\s)/, "$1\\$2$3"),
    )
    .join("\n");
}

/** The trace as plain text — one entry per block, blank-line separated. Mirrors what
 *  `TraceEntries.tsx` shows: thought → the text, action → the query plus its paper/per-paper
 *  scope, observation → the text. Emitted without Markdown styling because the caller wraps
 *  the whole thing in a code fence (keeping this secondary material visually quiet and the
 *  answer prose prominent); a code block also renders every `<...>`, `[rN]`, and `$math$` in
 *  the raw search output verbatim instead of trying to parse it. */
function traceToText(entries: TraceEntry[]): string {
  return entries
    .map((e) => {
      if (e.type === "thought") return `Thought: ${e.text ?? ""}`;
      if (e.type === "action") {
        const scope = [e.paper ? `(${e.paper})` : "", e.per_paper ? "(per-paper)" : ""]
          .filter(Boolean)
          .join(" ");
        return `Action: search "${e.query ?? ""}"${scope ? ` ${scope}` : ""}`;
      }
      return `Observation:\n${e.text ?? ""}`;
    })
    .join("\n\n");
}

/** A collapsed-by-default reasoning trace, its body a single code block so it reads as quiet
 *  monospace next to the answer prose. Empty trace → no block. */
function reasoningBlock(trace: TraceEntry[]): string {
  if (trace.length === 0) return "";
  return `<details>\n<summary>Reasoning</summary>\n\n${codeFence(traceToText(trace))}\n\n</details>`;
}

/** The cited passages as plain text, grouped by paper (first-seen order): a paper title +
 *  arXiv link, then each ref's number, section, a faithfulness flag when the automated check
 *  isn't a clean support, and the passage snippet. Wrapped in a code fence by the caller. */
function sourcesToText(cited: Citation[]): string {
  // Web citations carry no paper_id/section/arXiv link — mirror the "From the web" split in
  // SourceCards rather than grouping them under an empty paper_id with undefined sections.
  const webCited = cited.filter(isWebCitation);
  const paperCited = cited.filter(isPaperCitation);

  const byPaper = new Map<
    string,
    { title: string; arxivId?: string | null; refs: PaperCitation[] }
  >();
  for (const c of paperCited) {
    const g = byPaper.get(c.paper_id);
    if (g) g.refs.push(c);
    else byPaper.set(c.paper_id, { title: c.title, arxivId: c.arxiv_id, refs: [c] });
  }
  const papers = [...byPaper.values()].map(({ title, arxivId, refs }) => {
    const link = arxivId ? ` — https://arxiv.org/abs/${arxivId}` : "";
    const items = refs.map((c) => {
      const label = worstLabel(c.faithfulness);
      const flag = label && label !== "entailment" ? `  ⚠️ ${faithfulnessMessage(label)}` : "";
      return `[${refNumber(c)}] ${c.section_title}${flag}\n${c.snippet}`;
    });
    return `${title}${link}\n\n${items.join("\n\n")}`;
  });

  // Each web ref as: number, title — URL, then its snippet. Web refs aren't
  // faithfulness-checked, so no flag.
  const web = webCited.map((c) => `[${refNumber(c)}] ${c.title} — ${c.url}\n${c.snippet}`);
  const webBlock = web.length > 0 ? [`From the web:\n\n${web.join("\n\n")}`] : [];
  return [...papers, ...webBlock].join("\n\n");
}

/** A collapsed-by-default "Sources" block mirroring the `SourceCards` panel under each
 *  answer, its body a single code block (quiet monospace). Empty → no block. */
function sourcesBlock(cited: Citation[]): string {
  if (cited.length === 0) return "";
  return `<details>\n<summary>Sources</summary>\n\n${codeFence(sourcesToText(cited))}\n\n</details>`;
}

/** One compared paper's whole drill-down (compare mode), collapsed under the paper title:
 *  its own answer, then that answer's reasoning and sources, all as plain text in a single
 *  code block. The synthesis answer stays the turn's visible content; each paper's answer is
 *  secondary, so it lives here rather than as a rendered section. */
function perPaperBlock(row: CompareRow): string {
  const cited = citedCitations(row.text, row.citations);
  const parts = [row.text.trim()];
  if (row.trace.length > 0) parts.push(`Reasoning:\n${traceToText(row.trace)}`);
  if (cited.length > 0) parts.push(`Sources:\n${sourcesToText(cited)}`);
  return `<details>\n<summary>${escapeAngles(row.title)}</summary>\n\n${codeFence(parts.join("\n\n"))}\n\n</details>`;
}

/** The dimmed token/latency line shown under an answer (mirrors ChatPage's `formatUsage`). */
function usageLine(u: UsageInfo | null): string {
  if (!u) return "";
  const parts: string[] = [];
  if (u.input_tokens != null && u.output_tokens != null) {
    const total = u.input_tokens + u.output_tokens;
    parts.push(`${total.toLocaleString()} token${total === 1 ? "" : "s"}`);
  }
  parts.push(`${(u.latency_ms / 1000).toFixed(1)}s`);
  return `_${parts.join(" · ")}_`;
}

/** The reader's own thumbs-up/down and note on the answer, when they left one. */
function feedbackLine(f: Feedback | null): string {
  if (!f || !f.vote) return "";
  const icon = f.vote === "up" ? "👍" : "👎";
  return `**Feedback:** ${icon}${f.note ? ` — ${inlineText(f.note)}` : ""}`;
}

/** The whole conversation as one Markdown document, built so a reader skims question →
 *  answer → question → answer: a `# title`, then one `## Turn N` section per turn whose
 *  visible content is just the question and the answer (citations rewritten to namespaced
 *  footnotes). Everything secondary is collapsed below it and rendered as a plain code block
 *  when expanded — the reasoning trace, the Sources panel, and, in compare mode, each
 *  compared paper's own answer. Cited synthesis sources are also consolidated into one
 *  `## References` section at the end. Empty conversation → just the title. */
export function conversationToMarkdown(turns: StoredTurn[], title: string): string {
  const refs: RefEntry[] = [];

  const sections = turns.map((turn, i) => {
    const prefix = `t${i + 1}-`;
    const blocks: string[] = [`## Turn ${i + 1}`, `**Q:** ${inlineText(turn.question)}`];

    // The answer (the synthesis answer for a compare turn) is the visible content, right
    // under the question. Everything else is collapsed below it.
    const cited = turn.answer ? citedCitations(turn.answer, turn.citations) : [];
    if (turn.answer) blocks.push(rewriteRefs(turn.answer, cited, prefix, refs));

    const reasoning = reasoningBlock(turn.trace);
    if (reasoning) blocks.push(reasoning);

    const sources = sourcesBlock(cited);
    if (sources) blocks.push(sources);

    if (turn.compare && turn.compare_results) {
      for (const row of turn.compare_results) blocks.push(perPaperBlock(row));
    }

    const usage = usageLine(turn.usage);
    if (usage) blocks.push(usage);
    const feedback = feedbackLine(turn.feedback);
    if (feedback) blocks.push(feedback);

    return blocks.join("\n\n");
  });

  const body = sections.length > 0 ? `# ${title}\n\n${sections.join("\n\n---\n\n")}` : `# ${title}`;

  if (refs.length === 0) return body;

  // Order definitions by the numeric segments of their label (`t2-11` → [2,11]) so the
  // References list reads turn-by-turn and numerically within each turn, rather than in the
  // answers' inline-citation order.
  const key = (label: string) => label.replace(/^t/, "").split("-").map(Number);
  const ordered = [...refs].sort((a, b) => {
    const ka = key(a.label);
    const kb = key(b.label);
    for (let i = 0; i < Math.max(ka.length, kb.length); i++) {
      const d = (ka[i] ?? -1) - (kb[i] ?? -1);
      if (d) return d;
    }
    return 0;
  });

  const defs = ordered.map(({ label, citation }) => referenceLine(label, citation)).join("\n\n");
  return `${body}\n\n## References\n\n${defs}`;
}
