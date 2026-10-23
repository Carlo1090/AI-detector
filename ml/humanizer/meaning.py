"""Meaning-preservation checks: sentence-embedding similarity plus 'must keep' spans (names, numbers, quotes)."""

from __future__ import annotations

import re

import numpy as np

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

_NUMBER = re.compile(r"\b\d[\d,.]*%?")
_QUOTE = re.compile(r"[\"“][^\"”]{3,200}[\"”]")
_CITATION = re.compile(r"\([A-Z][A-Za-z\-]+(?: et al\.)?(?:,| and [A-Z][A-Za-z\-]+,)? \d{4}[a-z]?\)|\[\d+(?:[,–-]\d+)*\]")
_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z]+\b")
_COMMON = {"The", "This", "That", "These", "Those", "It", "They", "We", "You", "He", "She", "There", "Here"}


def must_keep(text: str) -> list[str]:
    """Spans a rewrite must reproduce exactly: numbers, quotations, citations, links and mid-sentence names."""
    keep = []
    for rx in (_NUMBER, _QUOTE, _CITATION, _URL):
        keep += [m.group(0).rstrip(".,") for m in rx.finditer(text)]
    for m in _CAPITALISED.finditer(text):
        before = text[:m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:\n\"“"
        if not sentence_start and m.group(0) not in _COMMON and len(m.group(0)) > 2:
            keep.append(m.group(0))
    return list(dict.fromkeys(k for k in keep if k))


def missing_spans(original: str, rewrite: str) -> list[str]:
    lowered = rewrite.lower()
    return [k for k in must_keep(original) if k.lower() not in lowered]


_INFORMAL = re.compile(
    r"\b(?:super|cool|stuff|kinda|sorta|gonna|wanna|gotta|buddy|buddies|awesome|grown-?ups?|kids?|folks|totally|"
    r"a bunch|huge|crazy|okay|ok|yeah|nope|guys|pretty much|tons of|loads of|basically|literally|chill)\b",
    re.IGNORECASE)


def new_informal_words(original: str, rewrite: str) -> list[str]:
    """Slang the rewrite introduces that the original didn't use (a register drop)."""
    before = {m.group(0).lower() for m in _INFORMAL.finditer(original)}
    return sorted({m.group(0).lower() for m in _INFORMAL.finditer(rewrite)} - before)


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups) - (1 if word.endswith("e") and len(groups) > 1 and not word.endswith("le") else 0)
    return max(1, n)


def reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level: roughly the school grade needed to read the text."""
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    if not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59


class Embedder:
    """Mean-pooled MiniLM sentence embeddings (runs comfortably on CPU)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        vecs = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy()

    def similarity(self, original: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        vecs = self.embed([original] + candidates)
        return (vecs[1:] @ vecs[0]).tolist()

    def coverage(self, original_sentences: list[str], rewrite_sentences: list[str]) -> float:
        """How well the worst-covered original sentence is matched by some rewritten sentence (or adjacent
        pair, since rewrites merge sentences). Low means an idea was dropped or changed."""
        if not original_sentences or not rewrite_sentences:
            return 0.0
        units = rewrite_sentences + [a + " " + b for a, b in zip(rewrite_sentences, rewrite_sentences[1:])]
        vecs = self.embed(original_sentences + units)
        orig, rew = vecs[:len(original_sentences)], vecs[len(original_sentences):]
        return float((orig @ rew.T).max(axis=1).min())
