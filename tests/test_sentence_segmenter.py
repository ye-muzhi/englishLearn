from englishlearn.processing.sentence_segmenter import sentence_segment


def test_subtitle_mode_keeps_abbreviation_with_next_word():
    result = sentence_segment([
        {"id": 4, "start": 0.0, "end": 0.6, "text": "Dr."},
        {"id": 8, "start": 0.65, "end": 1.3, "text": "Smith is here."},
    ], min_pause_sec=0.55, max_words=28)

    assert len(result) == 1
    assert result[0]["text"] == "Dr. Smith is here."
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.3


def test_subtitle_mode_breaks_on_natural_punctuation_and_pause():
    result = sentence_segment([
        {"id": 0, "start": 0.0, "end": 1.0, "text": "We can start now."},
        {"id": 1, "start": 1.7, "end": 2.5, "text": "But keep the door open."},
    ], min_pause_sec=0.55, max_words=28)

    assert [item["text"] for item in result] == [
        "We can start now.", "But keep the door open."
    ]
    assert [item["id"] for item in result] == [0, 1]


def test_sentence_end_inside_closing_quote_is_respected():
    result = sentence_segment([
        {"id": 0, "start": 0.0, "end": 0.8, "text": 'That is really more than enough."'},
        {"id": 1, "start": 0.9, "end": 1.8, "text": "Let's move on and leave this place."},
    ], min_pause_sec=0.55, max_words=28)

    assert len(result) == 2


def test_long_cue_splits_near_clause_boundary_and_preserves_timing():
    result = sentence_segment([
        {"id": 0, "start": 2.0, "end": 12.0,
         "text": "This is a long subtitle cue, but it still needs to remain readable for the learner while preserving the original timing."},
    ], min_pause_sec=0.55, max_words=12)

    assert len(result) > 1
    assert all(len(item["text"].split()) <= 12 for item in result)
    assert result[0]["start"] == 2.0
    assert result[-1]["end"] == 12.0


def test_multiple_sentences_inside_one_source_cue_split_at_punctuation():
    result = sentence_segment([
        {"id": 0, "start": 1.0, "end": 7.0,
         "text": "This is the first sentence. This is the second sentence! Is this the third?"},
    ], min_pause_sec=0.55, max_words=18)

    assert [item["text"] for item in result] == [
        "This is the first sentence.",
        "This is the second sentence!",
        "Is this the third?",
    ]
    assert result[0]["start"] == 1.0
    assert result[-1]["end"] == 7.0
    assert all(result[i]["end"] == result[i + 1]["start"] for i in range(2))


def test_consecutive_tiny_sentences_keep_explicit_punctuation_boundaries():
    result = sentence_segment([
        {"id": 0, "start": 0.0, "end": 3.0,
         "text": "This is Chongqing. Chongqing. Chongqing."},
    ], min_pause_sec=0.55, max_words=18)

    assert [item["text"] for item in result] == [
        "This is Chongqing.", "Chongqing.", "Chongqing.",
    ]
    assert result[0]["start"] == 0.0
    assert result[-1]["end"] == 3.0


def test_speaker_turns_inside_one_cue_are_separate_sentences():
    result = sentence_segment([{
        "id": 0,
        "start": 0.0,
        "end": 4.0,
        "text": "Sarah: Hi Mike! Mike: Hello Sarah!",
    }], min_pause_sec=0.55, max_words=18)

    assert [item["text"] for item in result] == [
        "Sarah: Hi Mike!", "Mike: Hello Sarah!",
    ]


def test_real_caption_pattern_joins_only_the_unfinished_clause():
    result = sentence_segment([
        {"id": 0, "start": 1.0, "end": 5.0,
         "text": "Chongqing received a huge amount of online attention. Yet,"},
        {"id": 1, "start": 5.0, "end": 9.0,
         "text": "until a few years ago, nobody outside China had heard of it."},
    ], min_pause_sec=0.55, max_words=18)

    assert [item["text"] for item in result] == [
        "Chongqing received a huge amount of online attention.",
        "Yet, until a few years ago, nobody outside China had heard of it.",
    ]


def test_small_length_overflow_finishes_a_natural_sentence():
    sentence = (
        "In this video we set out to see what lives up to the hype "
        "and what makes this city so popular."
    )
    result = sentence_segment([
        {"id": 0, "start": 0.0, "end": 3.0,
         "text": "In this video we set out to see what lives"},
        {"id": 1, "start": 3.0, "end": 7.0,
         "text": "up to the hype and what makes this city so popular."},
    ], min_pause_sec=0.55, max_words=18)

    assert [item["text"] for item in result] == [sentence]


def test_long_sentence_prefers_comma_boundary_and_normalises_source_lines():
    result = sentence_segment([
        {"id": 0, "start": 0.0, "end": 8.0,
         "text": "This source has a line break,\nand the sentence continues with enough words to require a readable subtitle split."},
    ], min_pause_sec=0.55, max_words=10)

    assert len(result) >= 2
    assert result[0]["text"].endswith(",")
    assert "\n" not in " ".join(item["text"] for item in result)
    assert all(len(item["text"].split()) <= 10 for item in result)
