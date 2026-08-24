"""Post-generation faithfulness check: verify each ``[rN]``-cited sentence of the
agent's answer is supported by the passage it cites, via a local consistency-
scoring cross-encoder.

One backend per ``faithfulness.type`` variant (see ``FaithfulnessCfg`` in
``config.py``):

* ``hf`` — a local sentence-transformers cross-encoder (default
          ``vectara/hallucination_evaluation_model`` at revision
          ``hhem-1.0-open``), loaded the same lazy way as ``CrossEncoderReranker``
          (see ``reranker.py``) — no new dependency. Unlike a generic 3-way NLI
          classifier, this checkpoint outputs one ``[0, 1]`` consistency score per
          pair; labels are derived from two configured thresholds (see
          ``HFFaithfulnessCfg``).

Add a backend by registering a ``FaithfulnessCfg`` variant in ``config.py`` and
adding a match arm to ``build_faithfulness_checker`` below.

Scoring is sentence-vs-sentence (SummaC-style), not whole-passage-vs-sentence: a
generic sentence-pair checkpoint like this one is out of its training distribution
against a multi-hundred-token passage, and ``max_length`` truncation would
silently cut the passage before the supporting fact. ``best_support`` picks the
passage sentence giving a claim its strongest score.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import FaithfulnessCfg, HFFaithfulnessCfg

# One or more comma-separated refs per bracket: `[r1]` or `[r1, r2]`. A model
# bunching citations into one bracket is at least as likely as the system
# prompt's one-marker-per-citation examples; matching only `[rN]` would silently
# drop every ref in a bunched bracket, indistinguishable from "never cited".
#
# Also accepts fullwidth CJK brackets `【r1】` — observed in practice from a real
# local model that used them instead of ASCII brackets despite every prompt
# instructing `[rN]` explicitly (so a prompt reword alone isn't a reliable fix;
# this is a parsing-tolerance fix, not a change to what markers we ask for).
_REF_BRACKET = re.compile(r"[\[【](r\d+(?:\s*,\s*r\d+)*)[\]】]")
_REF_ID = re.compile(r"r\d+")

# Naive sentence boundary: `.`/`!`/`?` followed by whitespace. Doesn't special-
# case abbreviations ("e.g.", "Fig.") — worst case it over/under-splits a
# sentence around a citation marker, it never drops or misattributes a ref.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Verdict:
    label: str  # "entailment" | "neutral" | "contradiction"
    score: float  # the model's raw [0, 1] consistency score (higher = more supported)


class FaithfulnessChecker(ABC):
    """Scores whether each (premise, hypothesis) pair is entailed, neutral, or
    contradicted."""

    @abstractmethod
    def check_batch(self, pairs: list[tuple[str, str]]) -> list[Verdict]:
        """Return one Verdict per (premise, hypothesis) pair, same order."""


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences for citation attribution (not general NLP)."""
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def attribute_refs(text: str, refs: set[str]) -> dict[str, list[str]]:
    """Map each ref in `refs` to the sentence(s) of `text` that cite it via an
    `[rN]` (or bunched `[r1, r2]`) marker. A ref never cited in `text` is absent
    from the result."""
    out: dict[str, list[str]] = {}
    for sentence in split_sentences(text):
        brackets = (m.group(1) for m in _REF_BRACKET.finditer(sentence))
        cited = {m.group(0) for bracket in brackets for m in _REF_ID.finditer(bracket)}
        for ref in cited & refs:
            out.setdefault(ref, []).append(sentence)
    return out


def best_support(verdicts: list[Verdict]) -> Verdict:
    """SummaC-style max-pooling: the passage sentence giving this claim its
    strongest support wins — the verdict with the highest raw `score`."""
    return max(verdicts, key=lambda v: v.score)


def _to_verdict(score: float, contradiction_max: float, entailment_min: float) -> Verdict:
    """Pure, model-free score -> Verdict mapping, kept separate from
    `HFFaithfulnessChecker` so it's unit-testable without a real model."""
    if score <= contradiction_max:
        label = "contradiction"
    elif score >= entailment_min:
        label = "entailment"
    else:
        label = "neutral"
    return Verdict(label=label, score=score)


class HFFaithfulnessChecker(FaithfulnessChecker):
    """Local sentence-transformers consistency-scoring cross-encoder (lazy: loads
    on first `check_batch`), same shape as `CrossEncoderReranker`."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        contradiction_max: float = 0.05,
        entailment_min: float = 0.3,
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self.contradiction_max = contradiction_max
        self.entailment_min = entailment_min
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                revision=self.revision,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model

    def check_batch(self, pairs: list[tuple[str, str]]) -> list[Verdict]:
        if not pairs:
            return []
        # This checkpoint's own head already outputs calibrated [0, 1] consistency
        # scores (confirmed empirically) — no apply_softmax, unlike a generic
        # 3-way NLI cross-encoder.
        scores = self.model.predict(pairs)
        return [_to_verdict(float(s), self.contradiction_max, self.entailment_min) for s in scores]


def build_faithfulness_checker(
    cfg: FaithfulnessCfg, *, device: str | None = None
) -> FaithfulnessChecker:
    """Construct the checker described by `config.faithfulness` (its variant)."""
    match cfg:
        case HFFaithfulnessCfg():
            return HFFaithfulnessChecker(
                cfg.model,
                revision=cfg.revision,
                device=device,
                max_length=cfg.max_length,
                contradiction_max=cfg.contradiction_max,
                entailment_min=cfg.entailment_min,
            )
        case _:
            raise ValueError(f"Unknown faithfulness config: {type(cfg).__name__}")
