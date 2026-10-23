from ml.detector.lexicon import density, find_patterns
from ml.detector.text import count_words, normalize, split_sentences


def test_normalize_undoes_lookalikes_and_hidden_characters():
    tricked = "We dеlvе in​to this topic."  # Cyrillic е, zero-width space, no-break space
    n = normalize(tricked)
    assert n.text == "We delve into this topic."
    assert (n.homoglyphs, n.invisible) == (2, 1)
    start = n.text.index("into")
    a, b = n.span(start, start + 4)
    assert tricked[a:b] == "in​to"  # highlights land on the original characters
    assert find_patterns(n.text, split_sentences(n.text))  # the hidden "delve into" is found again


def _sentences(text):
    return [text[s.start:s.end] for s in split_sentences(text)]


def test_splits_on_punctuation_and_keeps_offsets():
    text = "First one. Second one?  Third one!"
    assert _sentences(text) == ["First one.", "Second one?", "Third one!"]


def test_does_not_split_abbreviations_initials_or_decimals():
    text = "Dr. Smith met J. Doe at 3.5 p.m. on Monday. They talked, e.g. about rain."
    assert _sentences(text) == ["Dr. Smith met J. Doe at 3.5 p.m. on Monday.", "They talked, e.g. about rain."]


def test_line_breaks_end_sentences():
    text = "Heading\n\nA paragraph without a full stop\n- a list item"
    assert _sentences(text) == ["Heading", "A paragraph without a full stop", "- a list item"]


def test_quotes_stay_with_their_sentence():
    assert _sentences('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]


def test_count_words_handles_contractions_and_hyphens():
    assert count_words("It's a well-known fact, isn't it?") == 6


def test_finds_words_phrases_and_sentence_openers():
    text = "Moreover, it's important to note that we must delve into this rich tapestry."
    labels = {h.label for h in find_patterns(text, split_sentences(text))}
    assert {"Transition opener", "it's important to note", "dive into", "rich tapestry"} <= labels


def test_overlapping_hits_keep_the_longest():
    text = "This plays a crucial role here."
    hits = find_patterns(text, split_sentences(text))
    assert [h.label for h in hits] == ["plays a crucial role"]


def test_plain_human_sentence_has_no_hits():
    text = "I missed the bus again, so I walked and got soaked."
    assert find_patterns(text, split_sentences(text)) == []


def test_density_is_weight_per_hundred_words():
    text = "We delve. " + "word " * 98
    hits = find_patterns(text, split_sentences(text))
    assert density(hits, text) == 3.0
