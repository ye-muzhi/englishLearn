from englishlearn.processing.subtitle_contract import (
    build_quality_report,
    normalize_raw_cues,
    normalize_segmented_cues,
    normalize_translated_cues,
)


def test_contract_assigns_stable_cue_ids_and_source_lineage():
    raw = normalize_raw_cues([
        {"id": 8, "start": 0.0, "end": 1.0, "text": "Hello"},
        {"id": 9, "start": 1.0, "end": 2.0, "text": "world."},
    ])
    segmented = normalize_segmented_cues([
        {"start": 0.0, "end": 2.0, "text": "Hello world."},
    ], raw)

    assert [cue["id"] for cue in raw] == [0, 1]
    assert len({cue["cue_id"] for cue in raw}) == 2
    assert segmented[0]["source_cue_ids"] == [raw[0]["cue_id"], raw[1]["cue_id"]]
    assert segmented[0]["cue_id"].startswith("cue_")


def test_translation_alignment_prefers_cue_identity_over_response_order():
    raw = normalize_raw_cues([
        {"start": 0, "end": 1, "text": "First."},
        {"start": 1, "end": 2, "text": "Second."},
    ])
    segmented = normalize_segmented_cues(raw, raw)
    reversed_translations = [
        {**segmented[1], "translation": "第二。"},
        {**segmented[0], "translation": "第一。"},
    ]

    translated = normalize_translated_cues(reversed_translations, segmented)

    assert [cue["translation"] for cue in translated] == ["第一。", "第二。"]
    assert [cue["cue_id"] for cue in translated] == [cue["cue_id"] for cue in segmented]


def test_quality_report_flags_real_integrity_errors():
    raw = normalize_raw_cues([
        {"start": 0, "end": 1, "text": "First."},
        {"start": 1, "end": 2, "text": "Second."},
    ])
    segmented = normalize_segmented_cues(raw, raw)
    translated = normalize_translated_cues([
        {**segmented[0], "translation": "第一。"},
        {**segmented[1], "translation": ""},
    ], segmented)

    report = build_quality_report(raw, segmented, translated)

    assert report["passed"] is False
    assert report["errors"] == 1
    assert any(issue["code"] == "missing_translation" for issue in report["issues"])


def test_project_store_persists_three_layers_and_lazy_quality(monkeypatch, tmp_path):
    from englishlearn.storage import project_store

    monkeypatch.setattr(project_store, "WORK_DIR", str(tmp_path))
    project = {"id": "layered-project", "status": "processing"}
    raw = [{"start": 0, "end": 1, "text": "Hello"}, {"start": 1, "end": 2, "text": "world."}]
    segmented = [{"start": 0, "end": 2, "text": "Hello world."}]
    translated = [{"start": 0, "end": 2, "text": "Hello world.", "translation": "你好，世界。"}]

    saved = project_store.save_subtitle_layers(project, raw, segmented, translated)

    assert saved["subtitle_schema_version"] == 2
    assert project_store.load_project_raw_subtitles(project["id"])[0]["text"] == "Hello"
    assert project_store.load_project_segmented_subtitles(project["id"])[0]["text"] == "Hello world."
    assert project_store.load_project_subtitles(project["id"])[0]["translation"] == "你好，世界。"
    assert project_store.load_project_subtitle_quality(project["id"])["passed"] is True
