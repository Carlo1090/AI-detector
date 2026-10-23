"""Humanizer: rewrite the parts of a text that read as AI, keep the meaning, prove it with the detector.

Pipeline
1. Detect: score every sentence (same report the web app shows).
2. Rules: word- and phrase-level fixes for AI-typical wording (always applied as suggestions).
3. Rewrite: flagged sentences are grouped into short passages; the rewriter proposes several versions.
4. Select: each version must keep the meaning (embedding similarity), every name/number/quote, a sane
   length, and add no new AI phrases. Among those, the one the detector finds least AI-like wins.
   Passages still flagged get another round with a push toward a more personal voice.
5. Report: every change with its reason, before/after scores, and the composed text.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from ml.detector.detector import Detector
from ml.detector.lexicon import find_patterns
from ml.detector.text import Span, count_words, normalize, split_sentences

from .meaning import Embedder, missing_spans, new_informal_words, reading_grade
from .rewriter import LocalRewriter
from .rules import Edit, apply_edits, find_edits

STRENGTHS = {
    # rewrite: use the language model; min_prob: rewrite sentences at/above this (None = detector threshold)
    "light": {"rewrite": False, "min_prob": None, "rounds": 0, "candidates": 0},
    "balanced": {"rewrite": True, "min_prob": None, "rounds": 2, "candidates": 4},
    "strong": {"rewrite": True, "min_prob": 0.3, "rounds": 3, "candidates": 6},
}
MIN_SIMILARITY = 0.82  # whole passage, MiniLM cosine
MIN_COVERAGE = 0.62  # every original sentence must be matched at least this well
MAX_GRADE_DROP = 3.0  # Flesch-Kincaid grades a rewrite may lose (keeps an essay from turning into a kids' text)
_CJK = re.compile(r"[　-鿿가-힯]")


@dataclass
class Passage:
    start: int
    end: int
    sentences: list[int]
    source: str  # original text after rule edits
    before: str  # preceding text, for the language model's context
    best: str = ""
    best_prob: float = 1.0
    best_similarity: float = 1.0
    tried: int = 0
    rejected: dict = field(default_factory=dict)


def _passages(text: str, sentences: list[Span], targets: list[int], max_sentences: int = 2) -> list[list[int]]:
    """Group consecutive target sentences, never across a paragraph break."""
    groups: list[list[int]] = []
    for i in targets:
        if groups:
            last = groups[-1]
            gap = text[sentences[last[-1]].end:sentences[i].start]
            if last[-1] == i - 1 and "\n" not in gap and len(last) < max_sentences:
                last.append(i)
                continue
        groups.append([i])
    return groups


def _lexicon_weight(text: str) -> float:
    return sum(h.weight for h in find_patterns(text, split_sentences(text)))


class Humanizer:
    def __init__(self, detector: Detector, rewriter=None, embedder: Embedder | None = None):
        self.detector = detector
        self.rewriter = rewriter
        self.embedder = embedder

    @contextmanager
    def _rewriter_on_gpu(self):
        """The 4 GB GPU holds either the detector models or the local rewriter, so they take turns."""
        swap = isinstance(self.rewriter, LocalRewriter) and self.detector.scorer.device == "cuda"
        if swap:
            import torch

            self.detector.scorer.to("cpu")
            torch.cuda.empty_cache()
            self.rewriter.to("cuda")
        try:
            yield
        finally:
            if swap:
                self.rewriter.to("cpu")
                torch.cuda.empty_cache()
                self.detector.scorer.to("cuda")

    def _accept(self, p: Passage, candidate: str) -> tuple[bool, str, float]:
        """Quality gates a rewrite must pass before the detector gets a say."""
        if not candidate or _CJK.search(candidate):
            return False, "garbled", 0.0
        ratio = count_words(candidate) / max(1, count_words(p.source))
        if not 0.7 <= ratio <= 1.5:
            return False, "length", 0.0
        if missing_spans(p.source, candidate):
            return False, "dropped a name/number/quote", 0.0
        if new_informal_words(p.source, candidate):
            return False, "slang / register drop", 0.0
        if reading_grade(candidate) < reading_grade(p.source) - MAX_GRADE_DROP:
            return False, "dumbed down", 0.0
        if _lexicon_weight(candidate) > _lexicon_weight(p.source):
            return False, "added AI phrases", 0.0
        similarity = self.embedder.similarity(p.source, [candidate])[0]
        if similarity < MIN_SIMILARITY:
            return False, "meaning drifted", similarity
        src_sents = [p.source[s.start:s.end] for s in split_sentences(p.source)]
        new_sents = [candidate[s.start:s.end] for s in split_sentences(candidate)]
        if self.embedder.coverage(src_sents, new_sents) < MIN_COVERAGE:
            return False, "dropped an idea", similarity
        return True, "", similarity

    def humanize(self, text: str, strength: str = "balanced", tone: str = "natural") -> dict:
        t0 = time.perf_counter()
        cfg = STRENGTHS.get(strength, STRENGTHS["balanced"])
        text = normalize(text).text  # rewrite the visible text; hidden characters and look-alike letters go
        before = self.detector.analyze(text)
        sentences = split_sentences(text)
        probs = [s["probability"] or 0.0 for s in before["sentences"]]
        threshold = self.detector.calibration.window.threshold
        min_prob = cfg["min_prob"] if cfg["min_prob"] is not None else threshold
        edits = find_edits(text, contractions=(tone == "natural"), sentences=sentences)

        passages: list[Passage] = []
        if cfg["rewrite"] and self.rewriter is not None:
            targets = [i for i, p in enumerate(probs) if p >= min_prob]
            for group in _passages(text, sentences, targets):
                start, end = sentences[group[0]].start, sentences[group[-1]].end
                source = apply_edits(text, edits, start, end)
                passages.append(Passage(start, end, group, source, text[max(0, start - 800):start]))
            for p in passages:
                p.best, p.best_prob = p.source, self.detector.passage_probability(p.source, p.before)

            pending = list(passages)
            for round_no in range(cfg["rounds"]):
                if not pending:
                    break
                with self._rewriter_on_gpu():
                    candidates = self.rewriter.rewrite([p.source for p in pending], n=cfg["candidates"], tone=tone,
                                                       push=round_no > 0, temperature=0.85 + 0.1 * round_no)
                for p, cands in zip(pending, candidates):
                    for cand in dict.fromkeys(cands):
                        p.tried += 1
                        ok, why, sim = self._accept(p, cand)
                        if not ok:
                            p.rejected[why] = p.rejected.get(why, 0) + 1
                            continue
                        prob = self.detector.passage_probability(cand, p.before)
                        if prob < p.best_prob:
                            p.best, p.best_prob, p.best_similarity = cand, prob, sim
                pending = [p for p in pending if p.best_prob >= threshold]

        changes = self._changes(text, sentences, probs, edits, passages)
        final = self._compose(text, changes)
        after = self.detector.analyze(final)
        rewritten = [c for c in changes if c["kind"] == "rewrite"]
        return {
            "original": text,
            "text": final,
            "changes": changes,
            "before": _summary(before),
            "after": _summary(after),
            "similarity": round(sum(c["similarity"] for c in rewritten) / len(rewritten), 3) if rewritten else 1.0,
            "backend": getattr(self.rewriter, "name", None) if cfg["rewrite"] else "rules",
            "strength": strength, "tone": tone,
            "stats": {"passages": len(passages), "candidates_tried": sum(p.tried for p in passages),
                      "rejected": _merge(p.rejected for p in passages)},
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    def _changes(self, text, sentences, probs, edits: list[Edit], passages: list[Passage]) -> list[dict]:
        changes = []
        covered = []
        for p in passages:
            if p.best == p.source:  # no rewrite beat the rule-only version: fall back to individual rule edits
                continue
            covered.append((p.start, p.end))
            reasons = list(dict.fromkeys(e.reason for e in edits if p.start <= e.start < p.end))
            reasons.append("Rewrote for more natural rhythm and word choice; meaning checked.")
            changes.append({
                "kind": "rewrite", "start": p.start, "end": p.end, "before": text[p.start:p.end], "after": p.best,
                "reasons": reasons, "ai_before": round(max(probs[i] for i in p.sentences), 3),
                "ai_after": round(p.best_prob, 3), "similarity": round(p.best_similarity, 3),
            })
        for e in edits:
            if any(s <= e.start < t for s, t in covered):
                continue
            changes.append({"kind": "rule", "start": e.start, "end": e.end, "before": text[e.start:e.end],
                            "after": e.replacement, "reasons": [e.reason]})
        changes.sort(key=lambda c: c["start"])
        return changes

    @staticmethod
    def _compose(text: str, changes: list[dict]) -> str:
        """Apply accepted changes; rule deletions at a sentence start re-capitalise the next word."""
        edits = [Edit(c["start"], c["end"], c["after"], "") for c in changes]
        return apply_edits(text, edits)


def _summary(report: dict) -> dict:
    return {k: report[k] for k in ("verdict", "ai_share", "document_probability", "words")}


def _merge(dicts) -> dict:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out
