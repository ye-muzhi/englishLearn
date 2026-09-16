import json
from types import SimpleNamespace

import pytest

from englishlearn.processing.asr_engine import ASREngine
from englishlearn.media.media_manager import _parse_json3, _parse_srt
from englishlearn.processing.sentence_segmenter import sentence_segment


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def test_asr_preserves_faster_whisper_word_timestamps():
    segment = SimpleNamespace(
        start=0.8,
        end=3.4,
        text="Hello timing world.",
        words=[
            SimpleNamespace(word=" Hello", start=1.05, end=1.31),
            SimpleNamespace(word=" timing", start=1.72, end=2.08),
            SimpleNamespace(word=" world.", start=2.61, end=3.12),
        ],
    )

    class FakeWhisper:
        def transcribe(self, *_args, **_kwargs):
            return iter([segment]), SimpleNamespace()

    engine = ASREngine()
    engine.model = FakeWhisper()
    result = engine.transcribe("unused.wav")

    assert result[0]["start"] == 1.05
    assert result[0]["end"] == 3.12
    assert result[0]["words"] == [
        _word("Hello", 1.05, 1.31),
        _word("timing", 1.72, 2.08),
        _word("world.", 2.61, 3.12),
    ]


def test_sentence_split_uses_word_boundaries_not_duration_ratio():
    source = [{
        "id": 0,
        "start": 0.0,
        "end": 10.0,
        "text": "Fast words end. Slow words begin.",
        "words": [
            _word("Fast", 0.2, 0.4),
            _word("words", 0.4, 0.65),
            _word("end.", 0.65, 0.9),
            _word("Slow", 6.0, 6.4),
            _word("words", 6.4, 6.8),
            _word("begin.", 6.8, 7.3),
        ],
    }]

    result = sentence_segment(source, min_pause_sec=0.55, max_words=18)

    assert result == [
        {"id": 0, "start": 0.2, "end": 0.9, "text": "Fast words end."},
        {"id": 1, "start": 6.0, "end": 7.3, "text": "Slow words begin."},
    ]
    assert all("words" not in cue for cue in result)


def test_pipeline_can_keep_word_timing_for_future_retranslation():
    source = [{
        "id": 0,
        "start": 0.0,
        "end": 2.0,
        "text": "Keep exact timing.",
        "words": [
            _word("Keep", 0.2, 0.5),
            _word("exact", 0.7, 1.0),
            _word("timing.", 1.2, 1.7),
        ],
    }]

    result = sentence_segment(
        source, min_pause_sec=0.55, max_words=18, keep_word_timing=True,
    )

    assert result[0]["words"] == source[0]["words"]


def test_long_split_uses_exact_adjacent_word_timestamps():
    tokens = "one two three four five six seven eight nine ten".split()
    source = [{
        "id": 0,
        "start": 1.0,
        "end": 9.0,
        "text": " ".join(tokens),
        "words": [
            _word(token, 1.0 + index * 0.35, 1.2 + index * 0.35)
            for index, token in enumerate(tokens)
        ],
    }]

    result = sentence_segment(source, min_pause_sec=0.55, max_words=5)

    assert len(result) == 2
    assert result[0]["end"] == pytest.approx(2.6)
    assert result[1]["start"] == pytest.approx(2.75)


def test_json3_uses_relative_offsets_and_does_not_fill_long_silence():
    raw = json.dumps({
        "events": [{
            "tStartMs": 1000,
            "dDurationMs": 7000,
            "segs": [
                {"utf8": "Fast ", "tOffsetMs": 0},
                {"utf8": "sentence. ", "tOffsetMs": 250},
                {"utf8": "Slow ", "tOffsetMs": 4500},
                {"utf8": "sentence.", "tOffsetMs": 5000},
            ],
        }],
    })

    parsed = _parse_json3(raw)

    assert parsed[0]["words"][0]["start"] == 1.0
    assert parsed[0]["words"][1]["start"] == 1.25
    assert parsed[0]["words"][1]["end"] < 2.0
    assert parsed[0]["words"][2]["start"] == 5.5
    segmented = sentence_segment(parsed, min_pause_sec=0.55, max_words=18)
    assert segmented[0]["end"] < 2.0
    assert segmented[1]["start"] == 5.5


def test_json3_without_offsets_falls_back_to_monotonic_token_timing():
    raw = json.dumps({
        "events": [{
            "tStartMs": 1000,
            "dDurationMs": 800,
            "segs": [{"utf8": "Hello "}, {"utf8": "world"}],
        }],
    })

    parsed = _parse_json3(raw)
    words = parsed[0]["words"]

    assert words == [
        _word("Hello", 1.0, 1.4),
        _word("world", 1.4, 1.8),
    ]


def test_srt_overlaps_are_handed_off_at_the_next_cue():
    raw = (
        "1\n00:00:01,000 --> 00:00:04,000\nFirst cue\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nSecond cue\n"
    )

    parsed = _parse_srt(raw)

    assert parsed[0]["end"] == 3.0
    assert parsed[1]["start"] == 3.0


def test_project_storage_keeps_timing_only_in_raw_file(monkeypatch, tmp_path):
    from englishlearn.storage import project_store

    monkeypatch.setattr(project_store, "WORK_DIR", str(tmp_path))
    project = {"id": "timing-project", "status": "processing"}
    subtitles = [{
        "id": 0,
        "start": 0.2,
        "end": 1.7,
        "text": "Keep exact timing.",
        "translation": "保留精确时间。",
        "words": [
            _word("Keep", 0.2, 0.5),
            _word("exact", 0.7, 1.0),
            _word("timing.", 1.2, 1.7),
        ],
    }]

    project_store.save_subtitles_to_project(project, subtitles)
    raw = project_store.load_project_raw_subtitles(project["id"])
    translated = project_store.load_project_subtitles(project["id"])

    assert raw[0]["words"] == subtitles[0]["words"]
    assert "words" not in translated[0]


def test_reprocess_prefers_word_timing_and_resets_old_manual_offset(
    monkeypatch, tmp_path,
):
    from englishlearn.processing import pipeline_runner

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    video_path.touch()
    audio_path.touch()
    project = {
        "id": "timing-project",
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "subtitle_offset": -1.8,
        "status": "completed",
    }
    captured = {}

    class FakeASR:
        def __init__(self, model_size):
            assert model_size == "base"

        def load_model(self):
            return None

        def transcribe(self, _audio_path):
            return [{
                "id": 0,
                "start": 0.0,
                "end": 8.0,
                "text": "First sentence. Second sentence.",
                "words": [
                    _word("First", 0.2, 0.5),
                    _word("sentence.", 0.5, 0.9),
                    _word("Second", 5.2, 5.6),
                    _word("sentence.", 5.6, 6.1),
                ],
            }]

    def fake_translate(subtitles, **_kwargs):
        return [{**subtitle, "translation": "译文"} for subtitle in subtitles]

    def fake_save(proj, subtitles):
        captured["project"] = dict(proj)
        captured["subtitles"] = subtitles
        proj["status"] = "completed"
        return proj

    monkeypatch.setattr(pipeline_runner, "ASREngine", FakeASR)
    monkeypatch.setattr(pipeline_runner, "can_use_ai_segment", lambda _model: True)
    monkeypatch.setattr(
        pipeline_runner,
        "ai_sentence_segment",
        lambda *_args, **_kwargs: pytest.fail(
            "AI segmentation must not discard exact word timestamps"
        ),
    )
    monkeypatch.setattr(pipeline_runner, "translate_subtitles", fake_translate)
    monkeypatch.setattr(pipeline_runner, "load_projects", lambda: [project])
    monkeypatch.setattr(pipeline_runner, "save_subtitles_to_project", fake_save)
    monkeypatch.setattr(pipeline_runner, "update_project", lambda proj: proj)

    task = {}
    pipeline_runner.run_pipeline_thread(task, {
        "kind": "reprocess",
        "project_id": project["id"],
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "asr_model": "base",
        "model_name": "instruction-model",
        "engine_type": "openai",
        "api_base": "http://localhost:1234/v1",
        "api_key": "test",
        "target_lang": "Chinese",
        "target_lang_code": "zh",
    }, str(tmp_path))

    assert task["completed"] is True
    assert task["stage"] == "complete"
    assert captured["project"]["subtitle_offset"] == 0.0
    assert captured["project"]["timing_source"] == "word"
    assert [(cue["start"], cue["end"]) for cue in captured["subtitles"]] == [
        (0.2, 0.9),
        (5.2, 6.1),
    ]
