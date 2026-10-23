"""Sentence splitting, word counting and detector-evasion cleanup, keeping character offsets into the original text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int


_WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
# A run of sentence-ending punctuation plus any closing quotes/brackets, followed by whitespace or the end.
_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
_DOTTED = re.compile(r"^(?:[a-z]\.)+[a-z]$")  # e.g, i.e, u.s

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "figs", "no", "nos",
    "vol", "vols", "approx", "inc", "ltd", "co", "corp", "dept", "est", "gen", "gov", "lt", "col",
    "sgt", "capt", "rev", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mt", "ave", "blvd", "rd", "al", "cf", "ch", "pp", "eq", "ed", "eds",
}


# Look-alike letters from other scripts, used to hide text from detectors and plagiarism checkers.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ԁ": "d", "ѕ": "s",
    "ԛ": "q", "ԝ": "w", "һ": "h", "ӏ": "l", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S", "Ԁ": "D",
    "α": "a", "ο": "o", "ν": "v", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X",
})
_INVISIBLE = {"​", "‌", "‍", "⁠", "﻿", "­", "᠎"}
_SPACES = {" ", " ", " ", " ", " ", "　"}


@dataclass
class Normalized:
    text: str
    index: list[int]  # index[j] = position in the original text of normalized character j (+ one past the end)
    homoglyphs: int
    invisible: int

    def span(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) span of the normalized text back to the original text."""
        if end <= start:
            return self.index[start], self.index[start]
        return self.index[start], self.index[end - 1] + 1


def normalize(text: str) -> Normalized:
    """Undo common detector-evasion tricks: look-alike letters, zero-width characters, odd spaces and
    fullwidth forms. Each kept character records where it came from so results map back to the original."""
    out, index = [], []
    homoglyphs = invisible = 0
    for i, ch in enumerate(text):
        if ch in _INVISIBLE:
            invisible += 1
            continue
        if ch in _SPACES:
            out.append(" ")
            index.append(i)
            continue
        mapped = ch.translate(_HOMOGLYPHS)
        if mapped == ch and 0xFF01 <= ord(ch) <= 0xFF5E:  # fullwidth ASCII forms
            mapped = unicodedata.normalize("NFKC", ch)
        if mapped != ch and mapped.isascii():
            homoglyphs += 1
        out.append(mapped if len(mapped) == 1 else ch)
        index.append(i)
    index.append(len(text))
    return Normalized("".join(out), index, homoglyphs, invisible)


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def count_words(text: str) -> int:
    return sum(1 for _ in _WORD.finditer(text))


def _is_abbreviation(text: str, dot: int) -> bool:
    before = text[:dot].rsplit(None, 1)
    if not before:
        return False
    token = before[-1].lstrip("(\"'“‘[").lower()
    if token in ABBREVIATIONS or _DOTTED.match(token):
        return True
    return len(token) == 1 and token.isalpha()  # an initial, as in "J. Smith"


def _next_char(text: str, pos: int) -> str:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return text[pos] if pos < len(text) else ""


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end or not _WORD.search(text, start, end):
        return None
    return Span(start, end)


def split_sentences(text: str) -> list[Span]:
    """Split on sentence punctuation and on line breaks (headings and list items count as sentences)."""
    cuts = {m.start() for m in re.finditer(r"\n", text)}
    for m in _END.finditer(text):
        punct = m.group().rstrip("\"'”’)]")
        if punct == "." and _is_abbreviation(text, m.start()):
            continue
        if _next_char(text, m.end()).islower():
            continue
        cuts.add(m.end())

    spans, prev = [], 0
    for cut in sorted(cuts) + [len(text)]:
        span = _trim(text, prev, cut)
        if span:
            spans.append(span)
        prev = cut
    return spans
