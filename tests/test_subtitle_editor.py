def _seed(monkeypatch, tmp_path):
    from englishlearn.storage import project_store, subtitle_editor

    monkeypatch.setattr(project_store, "WORK_DIR", str(tmp_path))
    monkeypatch.setattr(subtitle_editor, "work_dir", lambda: tmp_path)
    project = {"id": "editable-project", "status": "processing"}
    raw = [
        {"start": 0, "end": 1, "text": "First sentence."},
        {"start": 1, "end": 2, "text": "Second sentence."},
    ]
    translated = [
        {"start": 0, "end": 1, "text": "First sentence.", "translation": "第一句。"},
        {"start": 1, "end": 2, "text": "Second sentence.", "translation": "第二句。"},
    ]
    project_store.save_subtitle_layers(project, raw, raw, translated)
    return project_store, subtitle_editor, project


def test_text_and_timing_edit_persists_in_segmented_and_translated_layers(monkeypatch, tmp_path):
    project_store, editor, project = _seed(monkeypatch, tmp_path)
    cue = project_store.load_project_subtitles(project["id"])[0]

    editor.update_cue(
        project["id"], cue["cue_id"], text="Edited source.",
        translation="修改后的译文。", start=0.1, end=1.2,
    )

    translated = project_store.load_project_subtitles(project["id"])
    segmented = project_store.load_project_segmented_subtitles(project["id"])
    assert translated[0]["text"] == "Edited source."
    assert translated[0]["translation"] == "修改后的译文。"
    assert segmented[0]["text"] == "Edited source."
    assert translated[0]["start"] == 0.1


def test_split_merge_undo_and_redo_survive_disk_reload(monkeypatch, tmp_path):
    project_store, editor, project = _seed(monkeypatch, tmp_path)
    cue = project_store.load_project_subtitles(project["id"])[0]

    split = editor.split_cue(project["id"], cue["cue_id"], len("First"))
    assert len(split) == 3
    assert [item["text"] for item in split[:2]] == ["First", "sentence."]
    assert editor.history_status(project["id"]) == {"undo": 1, "redo": 0}

    restored = editor.undo(project["id"])
    assert len(restored) == 2
    assert project_store.load_project_subtitles(project["id"])[0]["text"] == "First sentence."
    assert editor.history_status(project["id"]) == {"undo": 0, "redo": 1}

    redone = editor.redo(project["id"])
    assert len(redone) == 3
    merged = editor.merge_with_next(project["id"], redone[0]["cue_id"])
    assert len(merged) == 2
    assert merged[0]["text"] == "First sentence."
