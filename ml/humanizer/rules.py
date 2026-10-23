"""Deterministic word- and phrase-level rewrites of common AI patterns.

Every rule keeps grammar intact on its own (verb forms are carried over), so its edits can be offered as
individual suggestions. Where a rule has several plain alternatives, the choice is varied by position so a
long text doesn't trade one repeated tell for another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ml.detector.text import Span, split_sentences


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str
    reason: str


_VERB_FORMS = {  # base, 3rd person, past, -ing
    "look": ("look", "looks", "looked", "looking"),
    "use": ("use", "uses", "used", "using"),
    "encourage": ("encourage", "encourages", "encouraged", "encouraging"),
    "build": ("build", "builds", "built", "building"),
    "share": ("share", "shares", "shared", "sharing"),
    "show": ("show", "shows", "showed", "showing"),
    "highlight": ("highlight", "highlights", "highlighted", "highlighting"),
    "deal": ("deal", "deals", "dealt", "dealing"),
    "start": ("start", "starts", "started", "starting"),
    "simplify": ("simplify", "simplifies", "simplified", "simplifying"),
    "boost": ("boost", "boosts", "boosted", "boosting"),
    "play": ("play", "plays", "played", "playing"),
    "help": ("help", "helps", "helped", "helping"),
    "dig": ("dig", "digs", "dug", "digging"),
    "strengthen": ("strengthen", "strengthens", "strengthened", "strengthening"),
    "earn": ("earn", "earns", "earned", "earning"),
    "explain": ("explain", "explains", "explained", "explaining"),
    "make": ("make", "makes", "made", "making"),
    "reach": ("reach", "reaches", "reached", "reaching"),
    "tap": ("tap", "taps", "tapped", "tapping"),
}
_SUFFIX_INDEX = {"": 0, "e": 0, "s": 1, "es": 1, "ed": 2, "d": 2, "ing": 3, "ding": 3}


def _verb(target: str, suffix: str) -> str:
    return _VERB_FORMS[target][_SUFFIX_INDEX.get(suffix.lower(), 0)]


def _match_case(src: str, out: str) -> str:
    if src[:1].isupper() and out:
        return out[0].upper() + out[1:]
    return out


Rule = tuple[re.Pattern, Callable[[re.Match, int], str], str]


def _r(pattern: str, make: Callable[[re.Match, int], str], reason: str) -> Rule:
    return re.compile(pattern, re.IGNORECASE), make, reason


def _pick(*options: str) -> Callable[[re.Match, int], str]:
    return lambda m, n: options[n % len(options)]


def _verb_rule(target: str, tail: str = "") -> Callable[[re.Match, int], str]:
    return lambda m, n: _verb(target, m.group(1) or "") + tail


_AI_WORD = "AI-typical wording, swapped for a plainer word."
_FILLER = "Filler that AI adds; the sentence works without it."
_TRANSITION = "Formal transition opener; AI text starts far more sentences this way."

RULES: list[Rule] = [
    # stock openers and filler
    _r(r"\bit(?:'s|’s| is) (?:important|worth|crucial|essential|vital) (?:to )?(?:note|noting|mention|remember|recogni[sz]e) that\s+",
       _pick(""), _FILLER),
    _r(r"\bin today(?:'s|’s) (?:(?:fast-paced|digital|modern|ever-changing|interconnected|rapidly changing) )*(?:world|age|era|society|landscape)(,?)\s*",
       lambda m, n: ("these days" if n % 2 else "today") + (", " if m.group(1) else " "), "Cliché scene-setter that AI overuses."),
    _r(r"\bin the (?:ever-evolving|ever-changing|rapidly evolving|fast-paced) (?:world|landscape|field|realm) of\b",
       _pick("in"), "Cliché scene-setter that AI overuses."),
    # phrases
    _r(r"\bplay(s|ed|ing)? an? (?:crucial|pivotal|vital|key|significant|central|critical|essential) role\b",
       lambda m, n: _verb("play", m.group(1) or "") + (" a big part" if n % 2 == 0 else " a real part"),
       "“Plays a crucial role” is one of the most common AI phrases."),
    _r(r"\bdelv(e|es|ed|ing) (?:deeper )?into\b", lambda m, n: _verb("dig" if n % 2 else "look", m.group(1)) + (" into" if n % 2 else " at"), _AI_WORD),
    _r(r"\ba (?:wide|diverse|vast|broad) (?:range|array|variety|spectrum) of\b", _pick("all kinds of", "many", "lots of"), _AI_WORD),
    _r(r"\b(in|into|within) the realm of\s+", lambda m, n: m.group(1) + " ", _FILLER),
    _r(r"\ba (?:myriad|plethora) of\b", _pick("plenty of", "lots of", "countless"), _AI_WORD),
    _r(r"\bnavigat(e|es|ed|ing) the (?:complexities|intricacies|challenges|nuances) of\b",
       lambda m, n: _verb("deal", m.group(1)) + " with", _AI_WORD),
    _r(r"\bembark(s|ed|ing)? on\b", lambda m, n: _verb("start", m.group(1) or ""), _AI_WORD),
    _r(r"\bserv(es|ed|ing) as a testament to\b", lambda m, n: _verb("show", m.group(1)), _AI_WORD),
    _r(r"\b(is|was|are|were|stands as) a testament to\b",
       lambda m, n: {"is": "shows", "stands as": "shows", "was": "showed", "are": "show", "were": "showed"}[m.group(1).lower()],
       _AI_WORD),
    _r(r"\bshed(s|ding)? (?:new )?light on\b", lambda m, n: _verb("explain", m.group(1) or ""), _AI_WORD),
    _r(r"\b(?:unlock|unleash)(s|ed|ing)? the (?:full |true )?potential of\b",
       lambda m, n: _verb("make", m.group(1) or "") + " the most of", _AI_WORD),
    _r(r"\b(?:unlock|unleash)(s|ed|ing)? (its|their|your|our|his|her) (?:full |true )?potential\b",
       lambda m, n: f"{_verb('reach', m.group(1) or '')} {m.group(2)} potential", _AI_WORD),
    _r(r"\bharness(es|ed|ing)? the power of\b", lambda m, n: _verb("use", m.group(1) or ""), _AI_WORD),
    _r(r"\bharness(es|ed|ing)?\b", lambda m, n: _verb("tap", m.group(1) or "") + " into", _AI_WORD),
    _r(r"\bthe (educational|digital|political|business|media|tech|technological|competitive|cultural|economic|social|modern|current) landscape\b",
       lambda m, n: f"the {m.group(1)} world" if n % 2 == 0 else f"today's {m.group(1)} scene", _AI_WORD),
    _r(r"\bprofound(ly)?\b", lambda m, n: ("deeply" if m.group(1) else ("big" if n % 2 == 0 else "deep")), _AI_WORD),
    _r(r"\bfoster(s|ed|ing)? a sense of\b", lambda m, n: _verb("build", m.group(1) or "") + " a sense of", _AI_WORD),
    _r(r"\ba (?:deeper|more nuanced) understanding\b", _pick("a better understanding", "a clearer picture"), _AI_WORD),
    # single words (verbs keep their form)
    _r(r"\butili[sz](e|es|ed|ing)\b", lambda m, n: _verb("use", m.group(1)), _AI_WORD),
    _r(r"\bleverag(e|es|ed|ing)\b", lambda m, n: _verb("use", m.group(1)), _AI_WORD),
    _r(r"\bfoster(s|ed|ing)?\b", lambda m, n: _verb("encourage", m.group(1) or ""), _AI_WORD),
    _r(r"\bshowcas(e|es|ed|ing)\b", lambda m, n: _verb("share", m.group(1)), _AI_WORD),
    _r(r"\bunderscor(e|es|ed|ing)\b", lambda m, n: _verb("show" if n % 2 else "highlight", m.group(1)), _AI_WORD),
    _r(r"\bstreamlin(e|es|ed|ing)\b", lambda m, n: _verb("simplify", m.group(1)), _AI_WORD),
    _r(r"\bbolster(s|ed|ing)?\b", lambda m, n: _verb("strengthen", m.group(1) or ""), _AI_WORD),
    _r(r"\bgarner(s|ed|ing)?\b", lambda m, n: _verb("earn", m.group(1) or ""), _AI_WORD),
    _r(r"\bpivotal\b", _pick("key", "major"), _AI_WORD),
    _r(r"\bcrucial\b", _pick("important", "key"), _AI_WORD),
    _r(r"\bcrucially\b", _pick("importantly"), _AI_WORD),
    _r(r"\bseamless\b", _pick("smooth"), _AI_WORD),
    _r(r"\bseamlessly\b", _pick("smoothly"), _AI_WORD),
    _r(r"\brobust\b", _pick("strong", "solid"), _AI_WORD),
    _r(r"\bmultifaceted\b", _pick("complex", "many-sided"), _AI_WORD),
    _r(r"\bmeticulous(ly)?\b", lambda m, n: "careful" + ("ly" if m.group(1) else ""), _AI_WORD),
    _r(r"\bintricate\b", _pick("complex", "detailed"), _AI_WORD),
    _r(r"\bparamount\b", _pick("essential", "top priority"), _AI_WORD),
    _r(r"\bmyriad\b", _pick("countless", "many"), _AI_WORD),
    _r(r"\bplethora\b", _pick("wealth", "lot"), _AI_WORD),
    _r(r"\bgroundbreaking\b", _pick("new", "pioneering"), _AI_WORD),
    _r(r"\btransformative\b", _pick("game-changing", "major"), _AI_WORD),
    _r(r"\bholistic\b", _pick("whole-picture", "overall"), _AI_WORD),
    _r(r"\bcommendable\b", _pick("admirable", "impressive"), _AI_WORD),
    _r(r"\bvibrant\b", _pick("lively", "colorful"), _AI_WORD),
    _r(r"\bbustling\b", _pick("busy", "crowded"), _AI_WORD),
    _r(r"\bunwavering\b", _pick("steady", "firm"), _AI_WORD),
]

# Sentence-initial transitions: dropped (the sentence still stands) or swapped for a plain connector.
_OPENERS = re.compile(
    r"(?:Moreover|Furthermore|Additionally|In addition|Consequently|Notably|Importantly|Interestingly|"
    r"Ultimately|Overall|In essence|In summary|In conclusion|To sum up|Indeed),\s+")
_OPENER_SWAPS = {"Additionally": "Also, ", "In addition": "Also, ", "Moreover": "Also, ", "Consequently": "So ",
                 "Ultimately": "In the end, ", "In essence": "Basically, ", "In conclusion": "So ",
                 "In summary": "So ", "To sum up": "So "}

_CONTRACTIONS = [
    (r"\bit is\b", "it's"), (r"\bthat is\b", "that's"), (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"), (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"), (r"\bwas not\b", "wasn't"),
    (r"\bcannot\b", "can't"), (r"\bwill not\b", "won't"), (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"), (r"\bcould not\b", "couldn't"), (r"\bthey are\b", "they're"),
    (r"\bwe are\b", "we're"), (r"\byou are\b", "you're"), (r"\bthere is\b", "there's"), (r"\blet us\b", "let's"),
]
_CONTRACTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in _CONTRACTIONS]


def find_edits(text: str, contractions: bool = True, sentences: list[Span] | None = None) -> list[Edit]:
    """All rule edits for `text`, non-overlapping, in text order."""
    sentences = sentences if sentences is not None else split_sentences(text)
    edits: list[Edit] = []
    count = 0
    for pattern, make, reason in RULES:
        for m in pattern.finditer(text):
            out = make(m, count)
            count += 1
            edits.append(Edit(m.start(), m.end(), _match_case(m.group(0), out) if out else out, reason))

    for s in sentences:
        m = _OPENERS.match(text, s.start, s.end)
        if m:
            word = m.group(0).split(",")[0]
            count += 1
            swap = _OPENER_SWAPS.get(word, "") if count % 2 else ""
            edits.append(Edit(m.start(), m.end(), swap, _TRANSITION))

    if contractions:
        for pattern, contracted in _CONTRACTIONS:
            for m in pattern.finditer(text):
                if text[m.end():m.end() + 1] in (".", "!", "?", ",") and contracted.endswith(("'s", "'re")):
                    continue  # "that's what it is." reads oddly contracted at a clause end
                edits.append(Edit(m.start(), m.end(), _match_case(m.group(0), contracted),
                                  "Contractions make prose sound less stiff."))

    edits.sort(key=lambda e: (e.start, -(e.end - e.start)))
    kept: list[Edit] = []
    for e in edits:
        if kept and e.start < kept[-1].end:
            continue
        kept.append(e)
    return _self_contained(text, kept)


def _self_contained(text: str, edits: list[Edit]) -> list[Edit]:
    """Make each edit independently applicable: a deletion at a sentence start also capitalises the next word,
    merging with an edit that begins right where the deletion ends."""
    out: list[Edit] = []
    i = 0
    while i < len(edits):
        e = edits[i]
        if e.replacement == "" and _at_sentence_start(text, e.start):
            if i + 1 < len(edits) and edits[i + 1].start == e.end:
                nxt = edits[i + 1]
                merged = Edit(e.start, nxt.end, _capitalize_first(nxt.replacement), e.reason)
                edits = edits[:i] + [merged] + edits[i + 2:]
                continue  # the merged edit may itself be an empty sentence-start deletion
            if e.end < len(text) and text[e.end].islower():
                e = Edit(e.start, e.end + 1, text[e.end].upper(), e.reason)
        out.append(e)
        i += 1
    return out


def _at_sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \t")
    return not before or before[-1] in ".!?\n\"”"


def _capitalize_first(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
        if not ch.isspace():
            return s
    return s


def apply_edits(text: str, edits: list[Edit], start: int = 0, end: int | None = None) -> str:
    """Apply the edits that fall inside [start, end) and return that slice of text, rewritten."""
    end = len(text) if end is None else end
    out, pos, cap_next = [], start, False
    for e in edits:
        if e.start < start or e.end > end:
            continue
        gap = text[pos:e.start]
        out.append(_capitalize_first(gap) if cap_next and gap else gap)
        cap_next = cap_next and not gap
        if e.replacement:
            out.append(_capitalize_first(e.replacement) if cap_next else e.replacement)
            cap_next = False
        elif _at_sentence_start(text, e.start):
            cap_next = True  # a sentence-opening filler was dropped: the next word starts the sentence
        pos = e.end
    tail = text[pos:end]
    out.append(_capitalize_first(tail) if cap_next else tail)
    return tidy("".join(out))


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
