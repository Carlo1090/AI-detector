"""Turn token statistics, sentences and pattern hits into the numbers the scoring models use."""

from __future__ import annotations

import math

import numpy as np

from .lexicon import Hit, density
from .lm import TokenStats
from .text import Span, count_words

DOC_FEATURES = ("probe", "curvature", "binoculars", "mean_logprob", "top10_frac", "burst_ppl", "burst_len", "lexicon")
WINDOW_FEATURES = ("probe", "curvature", "binoculars", "mean_logprob", "top10_frac", "lexicon")

_NAN = float("nan")


def token_features(stats: TokenStats, idx: np.ndarray) -> dict[str, float]:
    if len(idx) == 0:
        return {"tokens": 0, "mean_logprob": _NAN, "curvature": _NAN, "binoculars": _NAN, "top1_frac": _NAN,
                "top10_frac": _NAN, "mean_entropy": _NAN}
    lp, mu, var, rank = stats.logprob[idx], stats.expected[idx], stats.variance[idx], stats.rank[idx]
    binoculars = _NAN
    if stats.perf_logprob is not None:
        # Binoculars: performer perplexity relative to how surprised the performer is by the observer's guesses.
        binoculars = float(-stats.perf_logprob[idx].mean() / stats.cross_entropy[idx].mean())
    return {
        "tokens": int(len(idx)),
        "mean_logprob": float(lp.mean()),
        # Fast-DetectGPT: how far the observed log-likelihood sits above what the model's own samples would give.
        "curvature": float((lp - mu).sum() / math.sqrt(float(var.sum()))),
        "binoculars": binoculars,
        "top1_frac": float((rank == 1).mean()),
        "top10_frac": float((rank <= 10).mean()),
        "mean_entropy": float(-mu.mean()),
    }


def pooled_hidden(stats: TokenStats, idx: np.ndarray) -> np.ndarray | None:
    """Mean of unit-normalised token states (normalising stops a few huge 'attention sink' tokens dominating)."""
    if stats.hidden is None or len(idx) == 0:
        return None
    h = stats.hidden[idx].astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-6
    return h.mean(axis=0)


def tokens_by_sentence(stats: TokenStats, sentences: list[Span], text_len: int) -> list[np.ndarray]:
    owner = np.full(text_len + 1, -1, np.int64)
    for i, s in enumerate(sentences):
        owner[s.start:s.end] = i
    token_owner = np.where(stats.ends > stats.starts, owner[stats.starts], -1)
    return [np.flatnonzero(token_owner == i) for i in range(len(sentences))]


def document_features(text: str, stats: TokenStats, sentences: list[Span],
                      sent_tokens: list[np.ndarray], hits: list[Hit]) -> dict[str, float]:
    feats = token_features(stats, np.flatnonzero(stats.ends > stats.starts))

    # Burstiness: humans swing between plain and surprising sentences, and between short and long ones.
    sent_logprob = [float(stats.logprob[idx].mean()) for idx in sent_tokens if len(idx) >= 6]
    feats["burst_ppl"] = float(np.std(sent_logprob)) if len(sent_logprob) >= 3 else _NAN
    lengths = np.array([count_words(text[s.start:s.end]) for s in sentences], np.float64)
    feats["burst_len"] = float(lengths.std() / lengths.mean()) if len(lengths) >= 3 and lengths.mean() > 0 else _NAN

    feats["lexicon"] = density(hits, text)
    return feats


def window_tokens(sent_tokens: list[np.ndarray], i: int, radius: int = 1) -> np.ndarray:
    lo, hi = max(0, i - radius), min(len(sent_tokens), i + radius + 1)
    return np.concatenate(sent_tokens[lo:hi])


def window_features(text: str, stats: TokenStats, sentences: list[Span], sent_tokens: list[np.ndarray],
                    hits: list[Hit], i: int, radius: int = 1) -> dict[str, float]:
    """Features for sentence i, read together with its neighbours (one sentence alone is too short to judge)."""
    lo, hi = max(0, i - radius), min(len(sentences), i + radius + 1)
    feats = token_features(stats, window_tokens(sent_tokens, i, radius))
    feats["lexicon"] = density(hits, text, sentences[lo].start, sentences[hi - 1].end)
    return feats
