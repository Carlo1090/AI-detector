"""The detector: one pass of the language model, then a document score and a score for every sentence."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from .features import (document_features, pooled_hidden, token_features, tokens_by_sentence, window_features,
                       window_tokens)
from .lexicon import Hit, density, find_patterns
from .lm import DEFAULT_MODEL, DEFAULT_PERFORMER, TokenScorer, TokenStats
from .scoring import Calibration, calibration_path
from .text import Span, count_words, normalize, split_sentences

MIN_WORDS = 150  # below this, a document-level verdict is not reliable enough to show
MAX_CHARS = 30_000

SIGNALS = {
    "probe": ("Trained AI-style classifier",
              "A classifier trained on thousands of human and AI texts reads the language model's internal "
              "picture of your text and estimates how AI-like its style is."),
    "curvature": ("Word-choice typicality",
                  "Compares each word with what the model itself would pick at random. AI text scores well above zero; "
                  "human text sits near or below it (Fast-DetectGPT)."),
    "binoculars": ("Binoculars",
                   "Compares a base model with its chat-tuned twin. Text that both find unsurprising scores low, "
                   "which leans AI (Hans et al., 2024)."),
    "mean_logprob": ("Predictability",
                     "Perplexity: how surprised the model was by each next word. Lower means more predictable, "
                     "which leans AI."),
    "top10_frac": ("Top-10 word share",
                   "Share of words that were among the model's 10 top guesses."),
    "burst_ppl": ("Burstiness",
                  "How much predictability swings from sentence to sentence. People vary more than AI."),
    "burst_len": ("Sentence-length variety",
                  "How much sentence length varies. AI tends to keep sentences evenly sized."),
    "lexicon": ("AI phrases",
                "Weighted count of AI-typical words and phrases per 100 words."),
}

_REASONS = {
    "probe": ("Style matches AI-written training texts", "Style matches human-written training texts"),
    "curvature": ("Word choices are unusually typical", "Unexpected word choices"),
    "binoculars": ("Reads like chatbot output", "Doesn't read like chatbot output"),
    "mean_logprob": ("Very predictable wording", "Less predictable wording"),
    "top10_frac": ("Most words were the model's top guesses", "Many words the model didn't expect"),
}


@dataclass
class Extraction:
    text: str
    sentences: list[Span]
    stats: TokenStats
    sent_tokens: list[np.ndarray]
    hits: list[Hit]
    document: dict[str, float]
    windows: list[dict[str, float]]
    doc_vec: np.ndarray | None = None  # pooled hidden state, when the scorer records one
    window_vecs: list[np.ndarray | None] | None = None


def _display(key: str, value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    if key == "mean_logprob":
        return f"{math.exp(-value):.1f}"
    if key == "top10_frac":
        return f"{value * 100:.0f}%"
    if key == "lexicon":
        return f"{value:.1f} per 100 words"
    if key == "curvature":
        return f"{value:+.2f}"
    if key == "binoculars":
        return f"{value:.3f}"
    if key == "probe":
        return f"{100 / (1 + math.exp(-max(-40.0, min(40.0, value)))):.0f}% AI-like"
    return f"{value:.2f}"


class Detector:
    def __init__(self, scorer: TokenScorer, calibration: Calibration | None):
        self.scorer = scorer
        self.calibration = calibration

    @classmethod
    def load(cls, model_name: str | None = None, performer_name: str | None = None) -> "Detector":
        """Model choice comes from DETECT_MODEL / DETECT_PERFORMER (set DETECT_PERFORMER=none to skip Binoculars)."""
        model_name = model_name or os.environ.get("DETECT_MODEL") or DEFAULT_MODEL
        performer_name = performer_name or os.environ.get("DETECT_PERFORMER") or DEFAULT_PERFORMER
        performer = None if performer_name.lower() == "none" else performer_name
        path = calibration_path(f"{model_name}+{performer}" if performer else model_name)
        calibration = Calibration.load(path) if path.exists() else None
        hidden_layer = calibration.hidden_layer if calibration else None
        return cls(TokenScorer(model_name, performer, hidden_layer=hidden_layer), calibration)

    def extract(self, text: str) -> Extraction:
        sentences = split_sentences(text)
        stats = self.scorer.score(text)
        sent_tokens = tokens_by_sentence(stats, sentences, len(text))
        hits = find_patterns(text, sentences)
        doc = document_features(text, stats, sentences, sent_tokens, hits)
        windows = [window_features(text, stats, sentences, sent_tokens, hits, i) for i in range(len(sentences))]
        doc_vec = window_vecs = None
        if stats.hidden is not None:
            doc_vec = pooled_hidden(stats, np.flatnonzero(stats.ends > stats.starts))
            window_vecs = [pooled_hidden(stats, window_tokens(sent_tokens, i)) for i in range(len(sentences))]
            cal = self.calibration
            if cal and cal.doc_probe:
                doc["probe"] = cal.doc_probe.logit(doc_vec)
                for w, vec in zip(windows, window_vecs):
                    w["probe"] = cal.window_probe.logit(vec)
        return Extraction(text, sentences, stats, sent_tokens, hits, doc, windows, doc_vec, window_vecs)

    def passage_probability(self, passage: str, before: str = "") -> float:
        """AI probability of `passage` read after `before` (context for the language model only)."""
        text = before + passage
        stats = self.scorer.score(text)
        start = len(before)
        idx = np.flatnonzero((stats.ends > stats.starts) & (stats.starts >= start))
        feats = token_features(stats, idx)
        feats["lexicon"] = density(find_patterns(passage, split_sentences(passage)), passage)
        if self.calibration.window_probe is not None:
            feats["probe"] = self.calibration.window_probe.logit(pooled_hidden(stats, idx))
        return self.calibration.window.probability(feats)

    def analyze(self, text: str) -> dict:
        """Full report. Analysis runs on a cleaned copy (look-alike letters, hidden characters undone); every
        span in the report points into the text as given."""
        t0 = time.perf_counter()
        norm = normalize(text[:MAX_CHARS])
        report = self._analyze(norm.text)
        for s in report["sentences"] + report["patterns"]:
            s["start"], s["end"] = norm.span(s["start"], s["end"])
        for t in report["tokens"]:
            t["s"], t["e"] = norm.span(t["s"], t["e"])
        report["manipulation"] = {"homoglyphs": norm.homoglyphs, "invisible": norm.invisible}
        report["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
        return report

    def _analyze(self, text: str) -> dict:
        ex = self.extract(text)
        n_words = count_words(text)
        too_short = n_words < MIN_WORDS
        cal = self.calibration

        sentences = []
        flagged_words = 0
        for i, (s, feats) in enumerate(zip(ex.sentences, ex.windows)):
            hits_here = [h for h in ex.hits if s.start <= h.start < s.end]
            prob = cal.window.probability(feats) if cal else None
            flagged = bool(cal and not too_short and prob >= cal.window.threshold)
            if flagged:
                flagged_words += count_words(text[s.start:s.end])
            sentences.append({
                "start": s.start, "end": s.end,
                "probability": None if prob is None else round(prob, 4),
                "flagged": flagged,
                "reasons": self._reasons(feats, hits_here),
            })

        doc_prob = cal.document.probability(ex.document) if cal else None
        ai_share = flagged_words / n_words if n_words and cal and not too_short else None
        contributions = cal.document.contributions(ex.document) if cal else {}

        signals = [{
            "key": key, "label": label, "description": desc,
            "value": _display(key, ex.document.get(key)),
            "contribution": round(contributions.get(key, 0.0), 3),
        } for key, (label, desc) in SIGNALS.items()
            if not math.isnan(ex.document.get(key, math.nan))]  # e.g. no Binoculars without a performer model
        signals.sort(key=lambda s: -abs(s["contribution"]))

        return {
            "model": self.scorer.name,
            "calibrated": cal is not None,
            "words": n_words,
            "too_short": too_short,
            "min_words": MIN_WORDS,
            "verdict": self._verdict(doc_prob, ai_share, too_short),
            "ai_share": None if ai_share is None else round(ai_share, 4),
            "document_probability": None if doc_prob is None else round(doc_prob, 4),
            "signals": signals,
            "sentences": sentences,
            "tokens": self._tokens(ex.stats),
            "patterns": [{"start": h.start, "end": h.end, "label": h.label, "kind": h.kind,
                          "weight": h.weight, "why": h.why} for h in ex.hits],
        }

    def _verdict(self, doc_prob: float | None, ai_share: float | None, too_short: bool) -> str:
        if too_short:
            return "too_short"
        if doc_prob is None:
            return "uncalibrated"
        doc_ai = doc_prob >= self.calibration.document.threshold
        if doc_ai and ai_share >= 0.5:
            return "likely_ai"
        if doc_ai or ai_share >= 0.2:
            return "mixed"
        return "likely_human"

    def _reasons(self, feats: dict[str, float], hits: list[Hit]) -> list[str]:
        reasons = []
        if self.calibration:
            for key, push in sorted(self.calibration.window.contributions(feats).items(), key=lambda kv: -abs(kv[1])):
                if key in _REASONS and abs(push) >= 0.4:
                    reasons.append(_REASONS[key][0 if push > 0 else 1])
        labels = list(dict.fromkeys(h.label for h in hits))
        if labels:
            reasons.append("AI-typical: " + ", ".join(labels[:4]))
        return reasons

    def _tokens(self, stats: TokenStats) -> list[dict]:
        tokens = []
        for i in np.flatnonzero(stats.ends > stats.starts):
            tokens.append({
                "s": int(stats.starts[i]), "e": int(stats.ends[i]),
                "lp": round(float(stats.logprob[i]), 3), "r": int(stats.rank[i]),
                "alts": [[self.scorer.decode(int(t)), round(float(p), 4)]
                         for t, p in zip(stats.alt_ids[i], stats.alt_probs[i])],
            })
        return tokens
