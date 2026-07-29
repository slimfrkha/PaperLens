import { describe, expect, it } from "vitest";
import type { Citation, FaithfulnessClaim } from "./api";
import {
  createClaimResolver,
  sentenceAt,
  splitSentencesWithOffsets,
  summarizeFaithfulness,
  worstLabel,
} from "./faithfulness";

// Matches the marker regex Answer.tsx uses to find [rN] occurrences in the text.
const MARKER = /\[(r\d+)\]/g;

function citation(ref: string, faithfulness?: FaithfulnessClaim[]): Citation {
  return {
    ref,
    paper_id: "p1",
    title: "T",
    breadcrumb: "b",
    section_title: "s",
    snippet: "sn",
    faithfulness,
  };
}

describe("splitSentencesWithOffsets", () => {
  it("recovers offsets that round-trip via text.slice", () => {
    const text = "  First sentence. Second sentence! Third?";
    const spans = splitSentencesWithOffsets(text);
    expect(spans.map((s) => s.text)).toEqual(["First sentence.", "Second sentence!", "Third?"]);
    for (const s of spans) {
      expect(text.slice(s.start, s.end)).toBe(s.text);
    }
  });

  it("agrees with the backend's naive _SENT_SPLIT on abbreviation over-splitting (tripwire for regex drift)", () => {
    // Mirrors src/rag/faithfulness.py's _SENT_SPLIT: no abbreviation handling, so
    // "Fig." incorrectly ends a sentence too. If either regex is ever "improved"
    // to special-case abbreviations without updating the other, this breaks loudly.
    const text = "Fig. 2 shows results. RoPE performs well.";
    const spans = splitSentencesWithOffsets(text);
    expect(spans.map((s) => s.text)).toEqual(["Fig.", "2 shows results.", "RoPE performs well."]);
  });
});

describe("createClaimResolver", () => {
  it("resolves a ref cited once to its single claim", () => {
    const text = "RoPE improves extrapolation [r1]. It was introduced in 2021.";
    const claim: FaithfulnessClaim = {
      sentence: "RoPE improves extrapolation [r1].",
      label: "entailment",
      score: 0.9,
    };
    const spans = splitSentencesWithOffsets(text);
    const resolve = createClaimResolver([citation("r1", [claim])]);
    const offset = text.indexOf("[r1]");
    expect(resolve("r1", sentenceAt(spans, offset))).toBe(claim);
  });

  it("resolves a ref cited in 2 different sentences to distinct claims, in text order", () => {
    const text = "Claim one [r1]. Claim two [r1].";
    const claimA: FaithfulnessClaim = {
      sentence: "Claim one [r1].",
      label: "entailment",
      score: 0.9,
    };
    const claimB: FaithfulnessClaim = { sentence: "Claim two [r1].", label: "neutral", score: 0.2 };
    const spans = splitSentencesWithOffsets(text);
    const resolve = createClaimResolver([citation("r1", [claimA, claimB])]);
    const offsets = [...text.matchAll(MARKER)].map((m) => m.index);
    expect(offsets).toHaveLength(2);
    expect(resolve("r1", sentenceAt(spans, offsets[0]))).toBe(claimA);
    expect(resolve("r1", sentenceAt(spans, offsets[1]))).toBe(claimB);
  });

  it("resolves a ref cited twice within one sentence to the SAME claim (backend's per-sentence dedup)", () => {
    const text = "Uses RoPE [r1] with base 10000 [r1].";
    const claim: FaithfulnessClaim = {
      sentence: "Uses RoPE [r1] with base 10000 [r1].",
      label: "contradiction",
      score: 0.01,
    };
    const spans = splitSentencesWithOffsets(text);
    expect(spans).toHaveLength(1);
    const resolve = createClaimResolver([citation("r1", [claim])]);
    const offsets = [...text.matchAll(MARKER)].map((m) => m.index);
    expect(offsets).toHaveLength(2);
    const first = resolve("r1", sentenceAt(spans, offsets[0]));
    const second = resolve("r1", sentenceAt(spans, offsets[1]));
    expect(first).toBe(claim);
    expect(second).toBe(claim);
  });

  it("returns undefined, never throws, for a citation with no faithfulness key", () => {
    const text = "Something [r2].";
    const spans = splitSentencesWithOffsets(text);
    const resolve = createClaimResolver([citation("r2")]);
    const offset = text.indexOf("[r2]");
    expect(resolve("r2", sentenceAt(spans, offset))).toBeUndefined();
  });

  it("returns undefined, never throws, when a marker's offset matches no sentence span", () => {
    const resolve = createClaimResolver([
      citation("r1", [{ sentence: "x", label: "entailment", score: 0.9 }]),
    ]);
    expect(sentenceAt([], 5)).toBeUndefined();
    expect(resolve("r1", undefined)).toBeUndefined();
  });

  it("a ref cited only in a bunched bracket is never matched by the single-ref marker regex, and never throws", () => {
    const text = "See [r1, r2] for details.";
    // Bunched brackets aren't matched by Answer.tsx's marker regex — a pre-existing,
    // documented limitation. The claim then simply sits unconsumed in the resolver's
    // queue, harmless: no crash, no misattribution.
    expect([...text.matchAll(MARKER)]).toHaveLength(0);

    const claim: FaithfulnessClaim = {
      sentence: "See [r1, r2] for details.",
      label: "contradiction",
      score: 0.01,
    };
    const citations = [citation("r1", [claim])];
    expect(() => createClaimResolver(citations)).not.toThrow();
    // Still visible to consumers that read citation.faithfulness directly (e.g. SourceCards).
    expect(summarizeFaithfulness(citations)).toEqual({
      total: 1,
      counts: { entailment: 0, neutral: 0, contradiction: 1 },
      worst: "contradiction",
    });
  });
});

describe("worstLabel", () => {
  const e: FaithfulnessClaim = { sentence: "a", label: "entailment", score: 0.9 };
  const n: FaithfulnessClaim = { sentence: "b", label: "neutral", score: 0.5 };
  const c: FaithfulnessClaim = { sentence: "c", label: "contradiction", score: 0.01 };

  it("picks the most severe label regardless of input order", () => {
    expect(worstLabel([e, n, c])).toBe("contradiction");
    expect(worstLabel([c, e, n])).toBe("contradiction");
    expect(worstLabel([e, n])).toBe("neutral");
    expect(worstLabel([e])).toBe("entailment");
  });

  it("returns undefined (a distinct 'unchecked' state) for undefined/empty input", () => {
    expect(worstLabel(undefined)).toBeUndefined();
    expect(worstLabel([])).toBeUndefined();
  });
});

describe("summarizeFaithfulness", () => {
  it("returns undefined when there's nothing to show", () => {
    expect(summarizeFaithfulness([])).toBeUndefined();
    expect(summarizeFaithfulness([citation("r1")])).toBeUndefined();
  });

  it("counts markers, not refs, and reports the correct worst label when everything is entailment", () => {
    const summary = summarizeFaithfulness([
      citation("r1", [
        { sentence: "a", label: "entailment", score: 0.9 },
        { sentence: "b", label: "entailment", score: 0.8 },
      ]),
    ]);
    expect(summary).toEqual({
      total: 2,
      counts: { entailment: 2, neutral: 0, contradiction: 0 },
      worst: "entailment",
    });
    // Consumers (e.g. SourceCards) derive "nothing to flag" from this, not from
    // summarizeFaithfulness itself — it always reports the full aggregate.
    expect(summary!.total - summary!.counts.entailment).toBe(0);
  });

  it("aggregates mixed labels across multiple citations", () => {
    const summary = summarizeFaithfulness([
      citation("r1", [
        { sentence: "a", label: "entailment", score: 0.9 },
        { sentence: "b", label: "neutral", score: 0.2 },
      ]),
      citation("r2", [{ sentence: "c", label: "contradiction", score: 0.01 }]),
    ]);
    expect(summary).toEqual({
      total: 3,
      counts: { entailment: 1, neutral: 1, contradiction: 1 },
      worst: "contradiction",
    });
  });
});
