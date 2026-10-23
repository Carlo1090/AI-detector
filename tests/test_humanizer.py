"""Humanizer behaviour with stand-in detector, rewriter and embedder (no model downloads)."""

import pytest

from ml.humanizer.humanizer import Humanizer, _passages
from ml.humanizer.meaning import missing_spans, must_keep
from ml.humanizer.rules import apply_edits, find_edits
from ml.detector.text import split_sentences


def _splice(text, changes, key="after"):
    out, pos = "", 0
    for c in changes:
        out += text[pos:c["start"]] + c[key]
        pos = c["end"]
    return out + text[pos:]


# ---------- rules ----------

def test_rules_keep_verb_forms():
    text = "She utilized the tool. They leverage data. He is leveraging trends. It fostered growth."
    out = apply_edits(text, find_edits(text, contractions=False))
    assert out == "She used the tool. They use data. He is using trends. It encouraged growth."


def test_dropping_an_opener_capitalises_the_next_word():
    text = "Moreover, it is important to note that cats sleep a lot. Furthermore, dogs bark."
    assert apply_edits(text, find_edits(text)) == "Cats sleep a lot. Dogs bark."


def test_every_rule_edit_applies_on_its_own():
    text = "In today's fast-paced world, teams leverage tools. Moreover, it plays a crucial role here."
    edits = find_edits(text)
    for e in edits:  # splicing any single edit must give a well-formed sentence start
        single = text[:e.start] + e.replacement + text[e.end:]
        for s in split_sentences(single):
            assert single[s.start].isupper() or not single[s.start].isalpha()


def test_academic_tone_skips_contractions():
    text = "It is clear that we do not know."
    assert apply_edits(text, find_edits(text, contractions=False)) == text
    assert apply_edits(text, find_edits(text, contractions=True)) == "It's clear that we don't know."


# ---------- meaning ----------

def test_must_keep_finds_numbers_names_citations_quotes():
    text = 'In 2019, sales rose 12% at Acme Corp (Smith, 2020), and the CEO said "we are thrilled".'
    keep = must_keep(text)
    for item in ["2019", "12%", "Acme", "Corp", "(Smith, 2020)", '"we are thrilled"']:
        assert item in keep
    assert missing_spans(text, "Sales went up 12% in 2019 at Acme Corp (Smith, 2020).") == ['"we are thrilled"', "CEO"]


def test_passages_group_consecutive_sentences_within_a_paragraph():
    text = "One. Two. Three. Four.\n\nFive. Six."
    s = split_sentences(text)
    assert _passages(text, s, [0, 1, 2, 3, 4, 5], max_sentences=3) == [[0, 1, 2], [3], [4, 5]]
    assert _passages(text, s, [0, 1, 2, 3, 4, 5]) == [[0, 1], [2, 3], [4, 5]]
    assert _passages(text, s, [0, 2, 3]) == [[0], [2, 3]]


def test_register_and_reading_level_checks():
    from ml.humanizer.meaning import new_informal_words, reading_grade

    essay = "Social media plays a significant role in contemporary education and communication."
    assert new_informal_words(essay, "Social media is super important and cool for kids.") == ["cool", "kids", "super"]
    assert new_informal_words(essay, "Social media matters a great deal in modern education.") == []
    assert reading_grade("The cat sat. It was fun. We ran.") < reading_grade(essay) - 3


# ---------- pipeline ----------

class FakeCalibration:
    class window:
        threshold = 0.5


class FakeDetector:
    """Sentences containing 'delve' or 'crucial' look AI; scores fall as AI words disappear."""

    calibration = FakeCalibration()

    class scorer:
        device = "cpu"

    def _p(self, text):
        hits = sum(text.lower().count(w) for w in ("delve", "crucial", "moreover", "tapestry"))
        return min(0.99, 0.2 + 0.4 * hits)

    def analyze(self, text):
        sentences = [{"probability": self._p(text[s.start:s.end])} for s in split_sentences(text)]
        share = sum(s["probability"] >= 0.5 for s in sentences) / max(1, len(sentences))
        return {"sentences": sentences, "verdict": "likely_ai" if share > 0.5 else "likely_human",
                "ai_share": share, "document_probability": share, "words": len(text.split())}

    def passage_probability(self, passage, before=""):
        return self._p(passage)


class FakeRewriter:
    name = "fake"

    def __init__(self, outputs):
        self.outputs = outputs

    def rewrite(self, passages, n=4, tone="natural", push=False, temperature=0.9):
        return [self.outputs(p) for p in passages]

    def to(self, device):
        pass


class FakeEmbedder:
    def similarity(self, original, candidates):
        return [0.9 if "Acme" in c or "Acme" not in original else 0.3 for c in candidates]

    def coverage(self, original_sentences, rewrite_sentences):
        return 0.9


TEXT = ("We delve into the tapestry of Acme history. Plain sentence here.\n\n"
        "Moreover, this is crucial for Acme in 2020.")


def test_best_valid_candidate_wins_and_changes_compose():
    def outputs(passage):
        return ["We look closely at the long history of Acme.",  # good
                "We look closely at the long company history.",  # drops the name Acme -> rejected
                "我们 look closely at the long Acme history.",    # garbled -> rejected
                passage]                                          # no better than the rule-fixed version
    h = Humanizer(FakeDetector(), FakeRewriter(outputs), FakeEmbedder()).humanize(TEXT, "balanced")
    rewrites = [c for c in h["changes"] if c["kind"] == "rewrite"]
    assert [c["after"] for c in rewrites] == ["We look closely at the long history of Acme."]
    # the second passage was already fixed by rules, so it stays as individual word-level edits
    assert any(c["kind"] == "rule" and c["before"].startswith("Moreover") for c in h["changes"])
    assert h["text"] == _splice(TEXT, h["changes"])
    assert "Acme" in h["text"] and "2020" in h["text"]
    assert h["stats"]["rejected"].get("dropped a name/number/quote", 0) >= 1
    assert h["after"]["ai_share"] < h["before"]["ai_share"]
    # turning every change off gives back the original text
    assert _splice(TEXT, h["changes"], key="before") == TEXT


def test_light_mode_uses_rules_only():
    h = Humanizer(FakeDetector(), None, None).humanize(TEXT, "light")
    assert h["backend"] == "rules"
    assert all(c["kind"] == "rule" for c in h["changes"])
    assert "delve" not in h["text"].lower()


def test_sentence_alternatives_are_gated_and_sorted():
    text = "Intro line here. We delve into the tapestry of Acme history. Outro line."
    start = text.index("We delve")
    end = text.index("history.") + len("history.")
    outputs = lambda p: ["We look closely at the long history of Acme.",   # noqa: E731  good
                         "We look closely at the long company history.",   # drops Acme -> rejected
                         "Moreover we delve into the rich tapestry of Acme history."]  # more AI phrases -> rejected
    alt = Humanizer(FakeDetector(), FakeRewriter(outputs), FakeEmbedder()).alternatives(text, start, end)
    texts = [o["text"] for o in alt["options"]]
    assert "We look closely at the long history of Acme." in texts
    assert all("company history" not in t and "Moreover" not in t for t in texts)
    assert [o["ai"] for o in alt["options"]] == sorted(o["ai"] for o in alt["options"])
    assert alt["original"] == text[start:end]


def test_no_candidate_passes_keeps_rule_edits():
    h = Humanizer(FakeDetector(), FakeRewriter(lambda p: ["nonsense"]), FakeEmbedder()).humanize(TEXT, "balanced")
    assert not [c for c in h["changes"] if c["kind"] == "rewrite"]
    assert h["stats"]["rejected"]
