import json
import time


def _reset(runner):
    with runner.active_tasks_lock:
        runner.active_tasks.clear()
        runner._task_params.clear()
        runner._task_store_path = None


def test_task_state_is_persisted_and_running_work_becomes_interrupted(monkeypatch, tmp_path):
    from englishlearn.processing import pipeline_runner as runner

    _reset(runner)
    runner.configure_task_store(str(tmp_path))
    task = {
        "task_id": "persisted-task",
        "kind": "new",
        "project_id": "project-1",
        "stage": "translate",
        "step_label": "Translating",
        "started_at": time.time(),
        "completed": False,
        "progress": 0.5,
    }
    with runner.active_tasks_lock:
        runner.active_tasks.setdefault("session-a", {})[task["task_id"]] = task
        runner._persist_tasks_locked()

    payload = json.loads((tmp_path / "tasks.json").read_text())
    assert payload["tasks"][0]["progress"] == 0.5

    with runner.active_tasks_lock:
        runner.active_tasks.clear()
        runner._load_persisted_tasks_locked()
    restored = runner.active_tasks_for_session("session-a")["persisted-task"]
    assert restored["completed"] is True
    assert restored["stage"] == "interrupted"
    _reset(runner)


def test_task_snapshots_cannot_mutate_runner_state(tmp_path):
    from englishlearn.processing import pipeline_runner as runner

    _reset(runner)
    runner.configure_task_store(str(tmp_path))
    with runner.active_tasks_lock:
        runner.active_tasks["session-a"] = {
            "task": {"task_id": "task", "stage": "source", "completed": False}
        }
    snapshot = runner.all_active_tasks()
    snapshot["session-a:task"]["stage"] = "corrupted"

    assert runner.active_tasks["session-a"]["task"]["stage"] == "source"
    _reset(runner)


def test_cancel_request_is_durable_and_cooperative(tmp_path):
    from englishlearn.processing import pipeline_runner as runner

    _reset(runner)


def test_partial_retranslate_replaces_only_selected_cue(monkeypatch, tmp_path):
    from englishlearn.processing import pipeline_runner as runner

    project = {
        "id": "project-1", "video_path": str(tmp_path / "video.mp4"),
        "audio_path": None, "status": "completed",
    }
    segmented = [
        {"id": 0, "cue_id": "cue-a", "start": 0, "end": 1, "text": "First."},
        {"id": 1, "cue_id": "cue-b", "start": 1, "end": 2, "text": "Second."},
    ]
    translated = [
        {**segmented[0], "translation": "人工第一句。"},
        {**segmented[1], "translation": "旧第二句。"},
    ]
    saved = {}
    monkeypatch.setattr(runner, "load_projects", lambda: [project])
    monkeypatch.setattr(runner, "load_project_segmented_subtitles", lambda _pid: segmented)
    monkeypatch.setattr(runner, "load_project_raw_subtitles", lambda _pid: segmented)
    monkeypatch.setattr(runner, "load_project_subtitles", lambda _pid: translated)
    monkeypatch.setattr(
        runner, "translate_subtitles",
        lambda cues, **_kwargs: [{**cue, "translation": "新第二句。"} for cue in cues],
    )
    monkeypatch.setattr(
        runner, "save_subtitle_layers",
        lambda _project, _raw, _segmented, final: saved.update(final=final),
    )
    monkeypatch.setattr(runner, "update_project", lambda _project: True)

    task = {}
    runner.run_pipeline_thread(task, {
        "kind": "partial_retranslate", "project_id": "project-1",
        "cue_ids": ["cue-b"], "video_path": project["video_path"],
        "audio_path": None, "model_name": "mt-model", "engine_type": "ollama",
        "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama",
        "target_lang": "Chinese", "target_lang_code": "zh",
    }, str(tmp_path))

    assert task["stage"] == "complete"
    assert [cue["translation"] for cue in saved["final"]] == ["人工第一句。", "新第二句。"]
    runner.configure_task_store(str(tmp_path))
    task = {"task_id": "task", "stage": "source", "completed": False}
    with runner.active_tasks_lock:
        runner.active_tasks["session-a"] = {"task": task}

    assert runner.cancel_task("session-a:task") is True
    assert task["cancel_requested"] is True
    try:
        runner._check_cancelled(task)
    except runner.TaskCancelled:
        pass
    else:
        raise AssertionError("safe pipeline boundary must honor cancel request")
    _reset(runner)
