from englishlearn.processing.ai_segmenter import _preserves_source


def test_ai_segmentation_validation_accepts_punctuation_only_changes():
    source = [{"start": 0.0, "end": 2.0, "text": "Sarah: hello Mike"}]
    output = [{"start": 0.0, "end": 2.0, "text": "Sarah: Hello, Mike!"}]

    assert _preserves_source(source, output)


def test_ai_segmentation_validation_rejects_missing_or_reordered_words():
    source = [{"start": 0.0, "end": 2.0, "text": "one two three four"}]

    assert not _preserves_source(source, [
        {"start": 0.0, "end": 2.0, "text": "one two four"},
    ])
    assert not _preserves_source(source, [
        {"start": 0.0, "end": 2.0, "text": "one three two four"},
    ])


def test_ai_segmentation_validation_rejects_invalid_timeline():
    source = [{"start": 1.0, "end": 3.0, "text": "Keep every word."}]
    output = [{"start": 0.0, "end": 4.0, "text": "Keep every word."}]

    assert not _preserves_source(source, output)
