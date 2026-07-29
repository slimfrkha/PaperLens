import type { Citation, FaithfulnessClaim, FaithfulnessLabel } from "./api";

// Mirrors `_SENT_SPLIT`/`_REF_BRACKET` in src/rag/faithfulness.py — must stay
// byte-compatible with those regexes. A drift here only loses a badge (matching
// below requires an exact string match against the backend's `claim.sentence`),
// never mislabels one, but nothing else will notice if the two sides diverge.
const SENT_SPLIT = /(?<=[.!?])\s+/;

const SEVERITY: Record<FaithfulnessLabel, number> = {
  contradiction: 3,
  neutral: 2,
  entailment: 1,
};

export interface SentenceSpan {
  text: string;
  start: number;
  end: number;
}

/** Split `text` into sentences with character offsets into the original string
 *  (offsets `String.split` would otherwise discard). */
export function splitSentencesWithOffsets(text: string): SentenceSpan[] {
  const chunks = text
    .trim()
    .split(SENT_SPLIT)
    .map((s) => s.trim())
    .filter(Boolean);
  const spans: SentenceSpan[] = [];
  let cursor = 0;
  for (const chunk of chunks) {
    const start = text.indexOf(chunk, cursor);
    if (start === -1) continue;
    const end = start + chunk.length;
    spans.push({ text: chunk, start, end });
    cursor = end;
  }
  return spans;
}

export function sentenceAt(spans: SentenceSpan[], offset: number): SentenceSpan | undefined {
  return spans.find((s) => offset >= s.start && offset < s.end);
}

/** Resolves each `[rN]` marker occurrence in an answer to the specific
 *  faithfulness claim for the sentence it appears in. A ref cited in several
 *  sentences gets one claim per sentence (FIFO, in text order); a ref cited
 *  twice within the SAME sentence resolves both occurrences to the same claim,
 *  mirroring the backend's per-sentence dedup (one registry entry per distinct
 *  citing sentence, not per literal marker glyph). */
export function createClaimResolver(citations: Citation[]) {
  const byRef = new Map(citations.map((c) => [c.ref, c]));
  const queuesByRef = new Map<string, Map<string, FaithfulnessClaim[]>>();
  // Not a perf memo — this IS the same-sentence dedup mechanism: the second
  // marker for a (ref, sentence) pair must reuse the first's resolved claim
  // instead of shifting a new one off the queue.
  const resolvedAt = new Map<string, FaithfulnessClaim | undefined>();

  function queueFor(ref: string): Map<string, FaithfulnessClaim[]> {
    let q = queuesByRef.get(ref);
    if (!q) {
      q = new Map();
      for (const claim of byRef.get(ref)?.faithfulness ?? []) {
        const list = q.get(claim.sentence);
        if (list) list.push(claim);
        else q.set(claim.sentence, [claim]);
      }
      queuesByRef.set(ref, q);
    }
    return q;
  }

  return function resolve(
    ref: string,
    sentence: SentenceSpan | undefined,
  ): FaithfulnessClaim | undefined {
    if (!sentence) return undefined;
    const key = `${ref}@${sentence.start}`;
    if (resolvedAt.has(key)) return resolvedAt.get(key);
    const claim = queueFor(ref).get(sentence.text)?.shift();
    resolvedAt.set(key, claim);
    return claim;
  };
}

/** SourceCards renders one glyph per ref (not per occurrence), so it can only
 *  ever show a worst-of-per-ref verdict. `undefined` means "unchecked" — never
 *  conflate it with `"entailment"`. */
export function worstLabel(claims: FaithfulnessClaim[] | undefined): FaithfulnessLabel | undefined {
  if (!claims || claims.length === 0) return undefined;
  return claims.reduce<FaithfulnessLabel>(
    (worst, c) => (SEVERITY[c.label] > SEVERITY[worst] ? c.label : worst),
    claims[0].label,
  );
}

export interface FaithfulnessSummary {
  total: number;
  counts: Record<FaithfulnessLabel, number>;
  worst: FaithfulnessLabel;
}

/** Aggregates every citation's claims for a turn-level summary. `total` counts
 *  citation markers (one per claim), not distinct refs. Returns `undefined` when
 *  there's no faithfulness data at all (feature disabled, or nothing cited). */
export function summarizeFaithfulness(citations: Citation[]): FaithfulnessSummary | undefined {
  const counts: Record<FaithfulnessLabel, number> = { entailment: 0, neutral: 0, contradiction: 0 };
  let total = 0;
  let worst: FaithfulnessLabel | undefined;
  for (const c of citations) {
    for (const claim of c.faithfulness ?? []) {
      total++;
      counts[claim.label]++;
      if (!worst || SEVERITY[claim.label] > SEVERITY[worst]) worst = claim.label;
    }
  }
  return total === 0 || !worst ? undefined : { total, counts, worst };
}

/** A plain-language sentence for a single citation's verdict — the raw label
 *  ("neutral", "contradiction") means nothing to someone who hasn't read
 *  src/rag/faithfulness.py, so every render site should show this instead. */
export function faithfulnessMessage(label: FaithfulnessLabel): string {
  switch (label) {
    case "contradiction":
      return "may contradict this claim";
    case "neutral":
      return "doesn't clearly support this claim";
    case "entailment":
      return "supports this claim";
  }
}

/** Only `neutral`/`contradiction` are ever actually rendered with a color — the
 *  UI stays silent on `entailment` rather than certifying it, since the
 *  thresholds behind these labels are a starting calibration, not a validated
 *  guarantee. */
export function faithfulnessColor(label: FaithfulnessLabel): string {
  switch (label) {
    case "contradiction":
      return "red";
    case "neutral":
      return "orange";
    case "entailment":
      return "gray";
  }
}
