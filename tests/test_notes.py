import json
import time
from pathlib import Path


def _wait_for_note(note_agent, task_id: str, timeout: float = 2.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = note_agent.get(task_id)
        if task and task.get("completed"):
            return task
        time.sleep(0.01)
    raise AssertionError("note task did not finish")


def test_translation_only_model_keeps_an_editable_note_draft():
    from englishlearn.notes.note_agent import refine_note

    result = refine_note(
        "Precisely 在这里可以理解成的确，也可以说 exactly。",
        {"source_text": "Precisely.", "source_translation": "准确地。"},
        engine="hy_mt2_local",
        model="Hy-MT2-1.8B",
        api_base="",
        api_key="",
    )
    assert result["refined_by"] == "draft"
    assert result["title"] == "Precisely"
    assert "Precisely" in result["body"]
    assert result["source_text"] == "Precisely."
    assert result["warning"]


def test_written_note_is_refined_in_background(monkeypatch, tmp_path):
    from englishlearn.notes import note_agent

    note_agent._reset_for_tests()
    note_agent.configure(tmp_path)
    monkeypatch.setattr(note_agent, "refine_note", lambda raw_text, context, **kwargs: {
        "title": "Precisely 与 exactly",
        "body": "两者都可以用于确认。",
        "summary": "表达准确确认",
        "key_points": ["Precisely 更强调准确"],
        "vocabulary": [],
        "tags": ["表达"],
        "source_text": context["source_text"],
        "source_translation": "",
        "refined_by": "test:model",
        "warning": "",
    })
    task_id = note_agent.start({
        "project_id": "p1",
        "raw_text": "precisely 也可以说 exactly",
        "context": {"source_text": "Precisely."},
        "engine": "openai",
        "model": "test-model",
        "api_base": "http://127.0.0.1:9999/v1",
        "api_key": "test",
    })
    task = _wait_for_note(note_agent, task_id)
    assert task["error"] is None
    assert task["result"]["title"] == "Precisely 与 exactly"
    persisted = json.loads((tmp_path / "note_tasks.json").read_text(encoding="utf-8"))
    assert persisted["tasks"][0]["result"]["refined_by"] == "test:model"
    assert "api_key" not in json.dumps(persisted)
    note_agent.discard(task_id)


def test_voice_note_transcribes_locally_before_refinement(monkeypatch, tmp_path):
    from englishlearn.notes import note_agent

    class FakeASR:
        def __init__(self, model_size):
            self.model_size = model_size

        def load_model(self):
            return None

        def transcribe(self, audio_path, initial_prompt=None):
            assert Path(audio_path).exists()
            assert "学习笔记" in initial_prompt
            return [{"text": "Precisely means exactly in this context."}]

    note_agent._reset_for_tests()
    note_agent.configure(tmp_path)
    monkeypatch.setattr(note_agent, "ASREngine", FakeASR)
    monkeypatch.setattr(note_agent, "refine_note", lambda raw_text, context, **kwargs: {
        **note_agent._draft_result(raw_text, context),
        "title": "Precisely",
        "refined_by": "test:model",
        "warning": "",
    })
    audio_path = note_agent.save_audio(b"RIFFfake", tmp_path, "p1")
    task_id = note_agent.start({
        "project_id": "p1", "audio_path": audio_path, "raw_text": "",
        "context": {"source_text": "Precisely."}, "asr_model": "tiny",
        "engine": "openai", "model": "test", "api_key": "test",
    })
    task = _wait_for_note(note_agent, task_id)
    assert task["result"]["title"] == "Precisely"
    assert "means exactly" in task["raw_text"]
    note_agent.discard(task_id)
    assert not Path(audio_path).exists()


def test_player_and_notes_page_contracts():
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert 'id="noteBtn"' in source
    assert 'id="onlineNoteBtn"' in source
    assert "function openNoteComposer()" in source
    assert "video.pause();\n    var now" in source
    assert "sendAction('open_note'" in source
    assert 'st.audio_input(' in source
    assert "note_agent.start({" in source
    assert "notes_store.add(_note_entry(" in source
    assert "def notes_page():" in source
    assert "st.Page(notes_page" in source
    assert "def collections_page():" not in source


def test_legacy_sentence_favorites_migrate_once(tmp_path):
    from englishlearn.notes import migrate_legacy_favorites
    from englishlearn.storage.collections_store import CollectionStore

    favorites = CollectionStore(str(tmp_path / "favorites.json"))
    notes = CollectionStore(str(tmp_path / "notes.json"))
    favorite = favorites.add({
        "project_id": "p1", "subtitle_id": "cue-7", "time": 12.5,
        "text": "Precisely. That is exactly what I mean.",
        "translation": "的确，这正是我的意思。",
    })
    assert migrate_legacy_favorites(favorites, notes) == 1
    assert migrate_legacy_favorites(favorites, notes) == 0
    migrated = notes.load()
    assert len(migrated) == 1
    assert migrated[0]["legacy_favorite_id"] == favorite["id"]
    assert migrated[0]["title"] == "Precisely"
    assert migrated[0]["time"] == 12.5
    assert favorites.load() == [favorite]


def test_note_task_cleanup_refuses_running_work(monkeypatch, tmp_path):
    from englishlearn.notes import note_agent

    gate = __import__("threading").Event()
    note_agent._reset_for_tests()
    note_agent.configure(tmp_path)

    def blocked_refine(*args, **kwargs):
        gate.wait(1)
        return note_agent._draft_result(args[0], args[1])

    monkeypatch.setattr(note_agent, "refine_note", blocked_refine)
    task_id = note_agent.start({
        "project_id": "p1", "raw_text": "one idea", "context": {},
        "engine": "openai", "model": "test", "api_key": "test",
    })
    assert not note_agent.clear_finished_tasks()
    gate.set()
    _wait_for_note(note_agent, task_id)
    assert note_agent.clear_finished_tasks()
    assert note_agent.get(task_id) is None
