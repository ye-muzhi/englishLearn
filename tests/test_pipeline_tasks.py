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
