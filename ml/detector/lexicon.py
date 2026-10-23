"""Words, phrases and sentence shapes that appear far more often in AI-generated text than in human writing.

Weights: 3 = strong tell, 2 = common tell, 1 = weak on its own (humans use these too), 0.5 = only matters in bulk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import Span, count_words


@dataclass(frozen=True)
class Pattern:
    label: str
    regex: re.Pattern
    weight: float
    kind: str  # "word" | "phrase" | "structure"
    why: str
    sentence_start: bool = False


@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    label: str
    kind: str
    weight: float
    why: str


def _p(label, pattern, weight, kind, why, sentence_start=False):
    flags = re.IGNORECASE if not sentence_start else 0
    return Pattern(label, re.compile(pattern, flags), weight, kind, why, sentence_start)


_WORD_WHY = "A word AI models use far more often than people do."
_PHRASE_WHY = "A stock phrase that is much more common in AI-generated text."

_WORDS = [
    # strong
    ("delve", r"\bdelv(?:e|es|ed|ing)\b", 3),
    ("tapestry", r"\btapestr(?:y|ies)\b", 3),
    ("testament", r"\btestament\b", 3),
    ("intricate", r"\bintrica(?:te|tely|cies|cy)\b", 3),
    ("multifaceted", r"\bmultifaceted\b", 3),
    ("realm", r"\brealms?\b", 3),
    ("underscore", r"\bunderscor(?:e|es|ed|ing)\b", 3),
    ("meticulous", r"\bmeticulous(?:ly)?\b", 3),
    ("bustling", r"\bbustling\b", 3),
    ("ever-evolving", r"\bever[- ](?:evolving|changing|growing|expanding)\b", 3),
    ("embark", r"\bembark(?:s|ed|ing)?\b", 3),
    ("showcase", r"\bshowcas(?:e|es|ed|ing)\b", 3),
    ("beacon", r"\bbeacons?\b", 3),
    ("unwavering", r"\bunwavering\b", 3),
    ("pivotal", r"\bpivotal\b", 3),
    ("interplay", r"\binterplay\b", 3),
    # common
    ("crucial", r"\bcrucial(?:ly)?\b", 2),
    ("foster", r"\bfoster(?:s|ed|ing)?\b", 2),
    ("seamless", r"\bseamless(?:ly)?\b", 2),
    ("robust", r"\brobust\b", 2),
    ("holistic", r"\bholistic(?:ally)?\b", 2),
    ("paramount", r"\bparamount\b", 2),
    ("vibrant", r"\bvibrant\b", 2),
    ("nuanced", r"\bnuanced?\b", 2),
    ("landscape", r"\blandscapes?\b", 2),
    ("leverage", r"\bleverag(?:e|es|ed|ing)\b", 2),
    ("elevate", r"\belevat(?:e|es|ed|ing)\b", 2),
    ("unlock", r"\bunlock(?:s|ed|ing)?\b", 2),
    ("transformative", r"\btransformative\b", 2),
    ("groundbreaking", r"\bgroundbreaking\b", 2),
    ("cutting-edge", r"\bcutting[- ]edge\b", 2),
    ("myriad", r"\bmyriad\b", 2),
    ("plethora", r"\bplethora\b", 2),
    ("invaluable", r"\binvaluable\b", 2),
    ("resonate", r"\bresonat(?:e|es|ed|ing)\b", 2),
    ("captivating", r"\bcaptivat(?:e|es|ed|ing)\b", 2),
    ("commendable", r"\bcommendable\b", 2),
    ("streamline", r"\bstreamlin(?:e|es|ed|ing)\b", 2),
    ("cornerstone", r"\bcornerstones?\b", 2),
    ("endeavor", r"\bendeavou?rs?\b", 2),
    ("garner", r"\bgarner(?:s|ed|ing)?\b", 2),
    ("bolster", r"\bbolster(?:s|ed|ing)?\b", 2),
    ("spearhead", r"\bspearhead(?:s|ed|ing)?\b", 2),
    ("underpin", r"\bunderpin(?:s|ned|ning)?\b", 2),
    ("profound", r"\bprofound(?:ly)?\b", 2),
    ("harness", r"\bharness(?:es|ed|ing)?\b", 2),
    # weak
    ("comprehensive", r"\bcomprehensive(?:ly)?\b", 1),
    ("enhance", r"\benhanc(?:e|es|ed|ing|ement)\b", 1),
    ("utilize", r"\butili[sz](?:e|es|ed|ing|ation)\b", 1),
    ("facilitate", r"\bfacilitat(?:e|es|ed|ing)\b", 1),
    ("optimize", r"\boptimi[sz](?:e|es|ed|ing)\b", 1),
    ("innovative", r"\binnovative\b", 1),
    ("dynamic", r"\bdynamic\b", 1),
    ("navigate", r"\bnavigat(?:e|es|ed|ing)\b", 1),
]

_PHRASES = [
    ("it's important to note", r"\bit(?:'s|’s| is) (?:important|worth|crucial|essential|vital) (?:to )?(?:note|noting|remember|consider|recogni[sz]e|understand|mention)\b", 3),
    ("in today's world", r"\bin today(?:'s|’s) (?:fast-paced |digital |modern |ever-changing |interconnected |rapidly changing )*(?:world|age|era|landscape|society)\b", 3),
    ("plays a crucial role", r"\bplay(?:s|ed|ing)? an? (?:crucial|pivotal|vital|key|significant|central|critical|essential) role\b", 3),
    ("rich tapestry", r"\brich tapestry\b", 3),
    ("a testament to", r"\ba testament to\b", 3),
    ("shed light on", r"\bsh(?:ed|eds|edding) (?:new )?light on\b", 3),
    ("pave the way", r"\bpav(?:e|es|ed|ing) the way\b", 3),
    ("embark on a journey", r"\bembark(?:s|ed|ing)? on (?:a|an|this|the|your) (?:\w+ )?journey\b", 3),
    ("unlock the potential", r"\b(?:unlock|unleash)(?:s|ed|ing)? (?:the|its|their|your) (?:full |true )?potential\b", 3),
    ("deeper understanding", r"\ba (?:deeper|better|more nuanced|greater) understanding\b", 2),
    ("first and foremost", r"\bfirst and foremost\b", 2),
    ("in the ever-evolving world", r"\bin the (?:ever-evolving|ever-changing|dynamic|rapidly evolving|fast-paced) (?:world|landscape|field|realm|era)\b", 3),
    ("dive into", r"\b(?:dive|delve|dives|delves) (?:deeper |deep )?into\b", 3),
    ("navigate the complexities", r"\bnavigat(?:e|es|ing) (?:the )?(?:complexities|intricacies|challenges|nuances)\b", 3),
    ("in the realm of", r"\bin the realm of\b", 3),
    ("fostering a sense of", r"\bfoster(?:s|ed|ing)? a sense of\b", 3),
    ("let's dive in", r"\blet(?:'s|’s| us) (?:dive|explore|take a closer look|delve|unpack)\b", 3),
    ("I hope this helps", r"\bI hope this (?:helps|essay|article|guide)\b", 3),
    ("as an AI", r"\bas an AI\b", 3),
    ("serves as a", r"\bserv(?:es|ed|ing) as (?:a|an) (?:powerful |stark |vital |important )?(?:reminder|testament|catalyst|cornerstone|bridge|foundation)\b", 2),
    ("stands as a", r"\bstands? as (?:a|an)\b", 2),
    ("a wide range of", r"\ba (?:wide|diverse|vast|broad) (?:range|array|variety|spectrum) of\b", 2),
    ("it is essential to", r"\bit(?:'s|’s| is) (?:essential|crucial|imperative|vital|paramount) (?:to|that|for)\b", 2),
    ("valuable insights", r"\b(?:valuable|key|crucial|deeper) insights?\b", 2),
    ("whether you're", r"\bwhether you(?:'re|’re| are)\b", 2),
    ("in conclusion", r"\bin (?:conclusion|summary)\b", 2),
    ("this essay will", r"\bthis (?:essay|article|paper) (?:will|explores|delves|examines|aims to)\b", 2),
    ("when it comes to", r"\bwhen it comes to\b", 1),
    ("at the end of the day", r"\bat the end of the day\b", 1),
    ("to sum up", r"\bto sum (?:it )?up\b", 1),
    ("not only ... but also", r"\bnot only\b[^.!?]{1,80}?\bbut also\b", 1),
]

_STRUCTURES = [
    _p("“not just X, but Y” framing",
       r"\b(?:is|was|are|were|it(?:'s|’s)|isn(?:'t|’t)|aren(?:'t|’t)) not (?:just|only|merely|simply) (?:about )?[^.;!?]{1,60}?(?:[,;:]| —| –|—)\s*(?:but|it(?:'s|’s)|it is|they(?:'re|’re))\b",
       2, "structure", "Contrast framing (“it’s not just X — it’s Y”) is a signature AI move."),
    _p("Transition opener",
       r"(?:Moreover|Furthermore|Additionally|In addition|Consequently|Ultimately|Overall|Notably|Importantly|Interestingly|Indeed|Thus|Hence|In essence|In summary|In conclusion),",
       1, "structure", "AI text opens many sentences with a formal transition word.", sentence_start=True),
    _p("“Here's …” opener", r"Here(?:'s|’s| is| are)\b", 1, "structure",
       "Chat-style lead-in that AI assistants use to introduce lists and answers.", sentence_start=True),
    _p("Bold heading", r"\*\*[^*\n]{2,60}\*\*", 1, "structure",
       "Markdown bold headings are typical of pasted chatbot output."),
    _p("Em dash", r"\s?—\s?", 0.5, "structure",
       "Recent AI models use em dashes heavily. Fine in moderation; a tell in bulk."),
]

PATTERNS: list[Pattern] = (
    [_p(label, rx, w, "word", _WORD_WHY) for label, rx, w in _WORDS]
    + [_p(label, rx, w, "phrase", _PHRASE_WHY) for label, rx, w in _PHRASES]
    + _STRUCTURES
)


def find_patterns(text: str, sentences: list[Span]) -> list[Hit]:
    """All non-overlapping pattern hits, longest match first where two overlap."""
    hits: list[Hit] = []
    for pat in PATTERNS:
        if pat.sentence_start:
            for s in sentences:
                m = pat.regex.match(text, s.start, s.end)
                if m:
                    hits.append(Hit(m.start(), m.end(), pat.label, pat.kind, pat.weight, pat.why))
        else:
            for m in pat.regex.finditer(text):
                if m.end() > m.start():
                    hits.append(Hit(m.start(), m.end(), pat.label, pat.kind, pat.weight, pat.why))

    hits.sort(key=lambda h: (h.start, -(h.end - h.start), -h.weight))
    kept: list[Hit] = []
    for h in hits:
        if kept and h.start < kept[-1].end:
            continue
        kept.append(h)
    return kept


def density(hits: list[Hit], text: str, start: int = 0, end: int | None = None) -> float:
    """Summed hit weight per 100 words inside [start, end)."""
    end = len(text) if end is None else end
    n_words = count_words(text[start:end])
    if n_words == 0:
        return 0.0
    weight = sum(h.weight for h in hits if start <= h.start < end)
    return 100.0 * weight / n_words
