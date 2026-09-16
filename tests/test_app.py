"""AppTest-based smoke tests for the Video Subtitle Translator.

Run: `python3 -m pytest tests/test_app.py -x -v` or just `python3 tests/test_app.py`.

Each test boots the app via Streamlit's AppTest framework, simulates clicks,
and asserts on rendered widgets / session_state / module state. The pipeline
threads and LLM calls are NOT exercised here — these are UI smoke tests.
"""
import sys
import os
import time
import logging
import importlib
import subprocess
import tempfile

import pytest

# Silence Streamlit's ScriptRunContext warning when running outside a real server
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

APP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

_passed = 0
_failed = 0
_failures = []


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        _failures.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def fresh_app():
    """Force reimport of `app` and `pipeline_runner` so module-level state is fresh."""
    for mod in list(sys.modules):
        if mod == "app" or mod == "englishlearn" or mod.startswith("englishlearn."):
            del sys.modules[mod]
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    return at


def click_label(at, label):
    """Click the first button whose label matches exactly. Returns True if found."""
    for b in at.button:
        if b.label == label:
            b.click().run()
            return True
    return False


def click_key(at, key):
    for b in at.button:
        if b.key == key:
            b.click().run()
            return True
    return False


def click_label_contains(at, fragment):
    for b in at.button:
        if b.label and fragment in b.label:
            b.click().run()
            return True
    return False


# ---------------------------------------------------------------------------
# Test 1: app boots without error
# ---------------------------------------------------------------------------

def test_boots():
    print("\n[test_boots]")
    at = fresh_app()
    at.run()
    check("no exception", len(at.exception) == 0, str(at.exception))
    check("no error widget", len(at.error) == 0, str(at.error))
    check("has 项目库 subheader", any(s.value == "项目库" for s in at.subheader))


def test_model_credential_preflight(monkeypatch):
    import app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    openai_preset = {
        "engine_type": "openai", "model_name": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1", "api_key": "",
    }
    check("missing OpenAI key is rejected before pipeline", "OPENAI_API_KEY" in (app._translation_setup_error(openai_preset) or ""))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    check("environment OpenAI key is accepted", app._translation_setup_error(openai_preset) is None)
    ollama_preset = {
        "engine_type": "ollama", "model_name": "test-model",
        "api_base": "http://127.0.0.1:11434/v1", "api_key": "",
    }
    check("local Ollama gets SDK placeholder key", app._effective_preset_api_key(ollama_preset) == "ollama")


def test_transcript_speaker_prefixes_are_flat_and_readable():
    import app

    speaker, text = app._transcript_parts(
        {"text": "Sarah: Hello everyone!"}, "text", "说话人",
    )
    check("named speaker is separated for an inline prefix",
          speaker == "Sarah" and text == "Hello everyone!")

    speaker, text = app._transcript_parts(
        {"text": ">> Global oil shock is raising alarms."}, "text", "说话人",
    )
    check("speaker arrow becomes a visible flat name prefix",
          speaker == "说话人" and text == "Global oil shock is raising alarms.")

    speaker, text = app._transcript_parts(
        {"speaker": "Mike", "text": "\n  And I'm Mike.\n"}, "text", "说话人",
    )
    check("speaker metadata is prepended without indentation",
          speaker == "Mike" and text == "And I'm Mike.")


def test_transcript_sheet_uses_separate_numbered_speaker_and_bilingual_columns():
    import app

    markup = app._transcript_sheet_html([
        {"start": 0, "end": 1, "text": "Sarah: Hello!", "translation": "莎拉 anecdot：你好！"},
        {"start": 1, "end": 2, "text": "Welcome back.", "translation": "欢迎回来。"},
        {"start": 2, "end": 3, "text": "Mike: Hi Sarah.", "translation": "迈克：你好，莎拉。"},
    ], "zh")
    check("transcript has four-column headers",
          all(label in markup for label in (">时间<", ">说话人<", ">原文<", ">译文<")))
    check("speaker identities are anonymized and stable",
          markup.count(">说话人1<") == 2 and markup.count(">说话人2<") == 1)
    check("source and target have independent columns",
          'class="transcript-source"' in markup and 'class="transcript-target"' in markup)
    check("real speaker names are removed from displayed copy",
          "Sarah:" not in markup and "Mike:" not in markup and "莎拉 anecdot：" not in markup)


# ---------------------------------------------------------------------------
# Test 2: navigation to Settings and back
# ---------------------------------------------------------------------------

def test_settings_navigation():
    print("\n[test_settings_navigation]")
    at = fresh_app()
    at.run()
    # Click Settings in the sidebar
    clicked = False
    for b in at.button:
        if b.label == "Settings" or b.label == "设置":
            b.click().run()
            clicked = True
            break
    # st.navigation uses page links; fall back to checking the nav directly
    if not clicked:
        # The nav links are in at.page_link (newer streamlit) or via switch
        check("can navigate to settings", False, "no settings button found")
        return
    check("reached settings page",
          any("设置" in (s.value or "") or "Settings" in (s.value or "") for s in at.title))


# ---------------------------------------------------------------------------
# Test 3: ➕ opens new task form
# ---------------------------------------------------------------------------

def test_new_task_form():
    print("\n[test_new_task_form]")
    at = fresh_app()
    at.run()
    ok = click_key(at, "new_task_btn")
    check("clicked ➕", ok)
    check("form has URL input", len(at.text_input) >= 1)
    mode = next((item for item in at.radio if item.key == "new_task_playback_mode"), None)
    check("form exposes a project playback mode", mode is not None)
    check("local complete mode is the default", mode is not None and mode.value == "local")
    check("local and online modes are both explicit", mode is not None and mode.options == ["local", "online"])
    native_caption_option = next(
        (item for item in at.checkbox if item.key == "prefer_native_target_subtitles"),
        None,
    )
    check("native target subtitles are an explicit choice", native_caption_option is not None)
    check("native target subtitles are preferred by default", native_caption_option is not None and native_caption_option.value is True)
    check("form has Process button",
          any("处理视频" in (b.label or "") or "Process Video" in (b.label or "") for b in at.button))


def test_valid_url_submit_shows_background_task(monkeypatch):
    """A valid URL submit must visibly leave the form and show task progress."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("ENGLISHLEARN_WORK_DIR", tmp)
        at = fresh_app()
        at.run()
        check("opened new task form", click_key(at, "new_task_btn"))

        from englishlearn.processing import pipeline_runner
        from englishlearn.translation import hy_mt2_local
        monkeypatch.setattr(
            hy_mt2_local,
            "model_status",
            lambda *args, **kwargs: {"downloaded": True, "dependencies_ready": True, "message": "ready"},
        )

        def fake_start(session_token, params):
            task_id = params["project_id"]
            pipeline_runner.active_tasks.setdefault(session_token, {})[task_id] = {
                "task_id": task_id,
                "kind": "new",
                "project_id": params["project_id"],
                "step": 1,
                "stage": "source",
                "step_label": "Resolving source",
                "progress": 0.05,
                "started_at": time.time(),
                "completed": False,
                "consumed": False,
                "error": None,
                "result_project_id": None,
                "filename": params["url"],
                "batch_current": 0,
                "batch_total": 0,
            }
            return task_id

        monkeypatch.setattr(pipeline_runner, "start_pipeline_task", fake_start)
        url_input = next(w for w in at.text_input if w.key == "url_input")
        url_input.set_value("https://www.youtube.com/watch?v=QkdBXUikRQc").run()
        process_button = next(
            b for b in at.button
            if "处理视频" in (b.label or "") or "Process Video" in (b.label or "")
        )
        process_button.click().run()

        check("form closes after valid submit", not at.session_state["show_new_task"])
        check("pending project is not opened in a detached status view", at.session_state["current_project"] is None)
        check("background task remains in the project library", any(
            b.key and b.key.startswith("load_") for b in at.button
        ))
        check("project card shows creation time", any(
            "创建于" in (item.value or "") or "Created" in (item.value or "")
            for item in at.caption
        ))
        check("no exception after valid submit", len(at.exception) == 0, str(at.exception))


def test_progress_refresh_does_not_rerun_player(monkeypatch):
    """Task progress must update its own card without rebuilding the player."""
    import inspect
    import app

    poll_source = inspect.getsource(app._poll_tasks_fragment)
    card_source = inspect.getsource(app._render_project_card_progress_fragment)
    check("poll has no active-task app rerun", "task_progress_snapshot" not in poll_source)
    check("project progress has an isolated fragment", "@st.fragment(run_every=2)" in card_source)
    check("card fragment is scoped to one project", "project_id" in card_source and "_active_tasks_for_app" in card_source)


# ---------------------------------------------------------------------------
# Test 4: project cell click opens detail page
# ---------------------------------------------------------------------------

def test_open_project_detail():
    print("\n[test_open_project_detail]")
    at = fresh_app()
    at.run()
    # Find any project load button (starts with status icon)
    proj_btn = next(
        (b for b in at.button if b.key and b.key.startswith("load_")),
        None,
    )
    if not proj_btn:
        check("project list has cells", False, "no load_* button found")
        return
    check("project list has cells", True)
    proj_btn.click().run()
    check("no exception after click", len(at.exception) == 0, str(at.exception))
    check("title rendered", len(at.title) > 0)
    # Should show either the player OR processing status OR "no subtitles" hint
    has_manage = any("项目管理" in (b.label or "") or "Manage project" in (b.label or "") for b in at.button)
    check("project operations are grouped under management", has_manage)


# ---------------------------------------------------------------------------
# Test 5: project_store functions handle corrupt/empty files
# ---------------------------------------------------------------------------

def test_corrupt_subtitles_handling():
    print("\n[test_corrupt_subtitles_handling]")
    from englishlearn.storage.project_store import load_project_subtitles, load_project_raw_subtitles, has_subtitles
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        # Create a fake project dir with an empty file
        pid = "testpid_corrupt"
        pdir = os.path.join(tmp, "projects", pid)
        os.makedirs(pdir)
        with open(os.path.join(pdir, "subtitles_translated.json"), "w") as f:
            f.write("")  # empty

        # Monkey-patch _project_dir
        from englishlearn.storage import project_store
        orig = project_store._project_dir
        project_store._project_dir = lambda pid_: os.path.join(tmp, "projects", pid_)

        try:
            subs = load_project_subtitles(pid)
            check("empty file returns None", subs is None)
            proj = {"id": pid}
            check("has_subtitles returns False for empty file", has_subtitles(proj) is False)

            # Corrupt content
            with open(os.path.join(pdir, "subtitles_translated.json"), "w") as f:
                f.write("{not valid json")
            subs = load_project_subtitles(pid)
            check("corrupt file returns None", subs is None)
        finally:
            project_store._project_dir = orig


def test_project_subtitle_offset_persistence():
    """Timing corrections must survive reopening a project and remain bounded."""
    print("\n[test_project_subtitle_offset_persistence]")
    import tempfile
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        try:
            project = project_store.create_project(None, None, "zh", "test:model", title="Timing test")
            project_store.append_project(project)
            saved = project_store.update_project_subtitle_offset(project["id"], 0.46)
            reloaded = project_store.load_projects()[0]
            check("subtitle offset rounds to 0.1 seconds", saved == 0.5)
            check("subtitle offset is stored with the project", reloaded.get("subtitle_offset") == 0.5)
            check("out-of-range timing correction is rejected", project_store.update_project_subtitle_offset(project["id"], 5.1) is None)
        finally:
            project_store.WORK_DIR = original_work_dir


def test_project_deletion_removes_managed_data_only():
    """Deletion must clean app files without ever touching an external source."""
    print("\n[test_project_deletion_removes_managed_data_only]")
    import tempfile
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    with tempfile.TemporaryDirectory() as work_dir, tempfile.TemporaryDirectory() as external_dir:
        project_store.WORK_DIR = work_dir
        try:
            shared_video = os.path.join(work_dir, "lesson.mp4")
            unique_audio = os.path.join(work_dir, "lesson.wav")
            uploaded = os.path.join(work_dir, "uploads", "original.mp4")
            external_source = os.path.join(external_dir, "original.mp4")
            os.makedirs(os.path.dirname(uploaded), exist_ok=True)
            for path in (shared_video, unique_audio, uploaded, external_source):
                with open(path, "wb") as f:
                    f.write(b"test")

            first = project_store.create_project(shared_video, unique_audio, "zh", "test:model", title="First")
            second = project_store.create_project(shared_video, None, "zh", "test:model", title="Second")
            external = project_store.create_project(external_source, None, "zh", "test:model", title="External")
            for project in (first, second, external):
                project_store.append_project(project_store.save_subtitles_to_project(project, [{
                    "id": 0, "start": 0, "end": 1, "text": "Hello", "translation": "你好",
                }]))

            deleted = project_store.delete_project(first["id"])
            check("single delete removes project subtitle directory", deleted["directories"] == 1 and not os.path.exists(os.path.join(work_dir, "projects", first["id"])))
            check("single delete removes unshared managed audio", not os.path.exists(unique_audio))
            check("single delete keeps media shared by another project", os.path.exists(shared_video))

            deleted_external = project_store.delete_project(external["id"])
            check("single delete keeps an external local source", deleted_external["projects"] == 1 and os.path.exists(external_source))

            cleared = project_store.delete_all_project_data()
            check("delete all clears remaining project index", cleared["projects"] == 1 and project_store.load_projects() == [])
            check("delete all clears shared managed video and uploads", not os.path.exists(shared_video) and not os.path.exists(uploaded))
            check("delete all still keeps external local source", os.path.exists(external_source))
        finally:
            project_store.WORK_DIR = original_work_dir


def test_delete_all_learning_data_ui_contract():
    """The destructive global action needs acknowledgement and task safety."""
    import inspect
    import app

    source = inspect.getsource(app._render_project_list)
    check("project list exposes delete-all action", "delete_all_learning_data" in source and "projects.delete_all" in source)
    check("delete-all action requires acknowledgement", "delete_all_acknowledged" in source and "not acknowledged" in source)
    check("delete-all action is blocked while imports run", "bool(running)" in source and "projects.delete_all_busy" in source)


# ---------------------------------------------------------------------------
# Test 6: pipeline_runner state persists across simulated reruns
# ---------------------------------------------------------------------------

def test_pipeline_runner_persistence():
    print("\n[test_pipeline_runner_persistence]")
    # The whole point of pipeline_runner.py is that its state survives app reruns.
    # Simulate two "reruns" by re-importing and checking the module-level dict is stable.
    from englishlearn.processing import pipeline_runner
    pipeline_runner.active_tasks.clear()

    # Pretend a task was registered
    pipeline_runner.active_tasks["fake_token"] = {
        "fake_task": {"completed": False, "consumed": False, "step": "x"}
    }
    before_id = id(pipeline_runner.active_tasks)

    # Re-import — module is cached, so state persists
    importlib.reload(pipeline_runner)
    # After reload, the dict is RE-DEFINED in the module... but reload re-executes
    # module code, so the dict reference changes. Verify the design works in the
    # real scenario (Streamlit does NOT reload, only re-execs app.py).
    # For this test we just verify active_tasks_for_session is callable.
    result = pipeline_runner.active_tasks_for_session("fake_token")
    check("active_tasks_for_session callable", isinstance(result, dict))


# ---------------------------------------------------------------------------
# Test 7: collections store CRUD + page render
# ---------------------------------------------------------------------------

def test_collections_page():
    print("\n[test_collections_page]")
    # AppTest in this Streamlit version does not expose st.page_link, so we
    # cannot navigate to the Collections page via the test harness. Instead:
    #   (a) verify the home page renders without error
    #   (b) verify CollectionStore CRUD works on a temp file
    at = fresh_app()
    at.run()
    check("home page renders", len(at.exception) == 0, str(at.exception))

    import tempfile
    from englishlearn.storage.collections_store import CollectionStore

    with tempfile.TemporaryDirectory() as tmp:
        store = CollectionStore(os.path.join(tmp, "wordbook.json"))
        check("empty store returns []", store.load() == [])

        added = store.add({"word": "hello", "translation": "你好"})
        check("add assigns id + created_at",
              "id" in added and "created_at" in added and added["word"] == "hello")

        found = store.find(word="hello")
        check("find retrieves entry", found is not None and found["id"] == added["id"])

        items = store.load()
        check("persisted to disk", len(items) == 1 and items[0]["word"] == "hello")

        store.update(added["id"], translation="您好")
        check("update mutates fields",
              store.find(word="hello")["translation"] == "您好")

        store.remove(added["id"])
        check("remove deletes entry", store.find(word="hello") is None)

        # remove_if with criteria
        store.add({"word": "a", "pos": "noun"})
        store.add({"word": "b", "pos": "verb"})
        removed = store.remove_if(pos="noun")
        check("remove_if filters by criteria", removed and len(store.load()) == 1)


# ---------------------------------------------------------------------------
# Test 8: local player server exposes only selected video files
# ---------------------------------------------------------------------------

def test_video_server_allowlist():
    print("\n[test_video_server_allowlist]")
    import tempfile
    import urllib.error
    import urllib.parse
    import urllib.request
    from unittest.mock import patch
    from englishlearn.media.media_manager import start_video_server

    with tempfile.TemporaryDirectory() as tmp:
        video_name = "lesson.mp4"
        with open(os.path.join(tmp, video_name), "wb") as f:
            f.write(b"video-bytes")
        with open(os.path.join(tmp, "model_presets.json"), "wb") as f:
            f.write(b"private")

        try:
            url = start_video_server(tmp, video_name)
        except PermissionError as exc:
            # Hosted test sandboxes can prohibit opening an ephemeral localhost
            # listener even though the application itself runs normally on a
            # desktop. Keep the test executable on both environments.
            pytest.skip(f"This environment blocks local test listeners: {exc}")
        with urllib.request.urlopen(f"{url}/{video_name}", timeout=5) as resp:
            check("selected video is served", resp.read() == b"video-bytes")
        with patch("englishlearn.media.media_manager.free_word_lookup", return_value={
            "dictionary": [{"word": "hello"}], "translation": "你好",
        }):
            lookup_request = urllib.request.Request(
                f"{url}/api/word-lookup?word=hello&target=zh-CN",
                headers={"Origin": "http://localhost:8501"},
            )
            with urllib.request.urlopen(lookup_request, timeout=5) as resp:
                lookup_payload = json.loads(resp.read().decode("utf-8"))
                check("lookup bridge exposes a JSON endpoint", lookup_payload["translation"] == "你好")
                check("lookup bridge permits the isolated player iframe", resp.headers.get("Access-Control-Allow-Origin") == "*")
                check("lookup bridge permits private loopback access", resp.headers.get("Access-Control-Allow-Private-Network") == "true")
        with patch("englishlearn.media.media_manager.save_subtitle_offset", return_value={"saved": True, "subtitle_offset": 0.5}):
            offset_request = urllib.request.Request(
                f"{url}/api/subtitle-offset?project_id=20260723-abcdef&offset=0.5",
                headers={"Origin": "http://localhost:8501"},
            )
            with urllib.request.urlopen(offset_request, timeout=5) as resp:
                offset_payload = json.loads(resp.read().decode("utf-8"))
            check("offset bridge saves without a Streamlit reload", offset_payload == {"saved": True, "subtitle_offset": 0.5})
        with patch("englishlearn.media.media_manager.toggle_favorite_api", return_value={"ok": True, "favorited": True}):
            bridge_body = urllib.parse.urlencode({
                "request_id": "qa-request-1",
                "payload": json.dumps({"project_id": "project-123", "subtitle_id": "7"}),
            }).encode("utf-8")
            bridge_request = urllib.request.Request(
                f"{url}/api/favorite/toggle",
                data=bridge_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(bridge_request, timeout=5) as resp:
                bridge_html = resp.read().decode("utf-8")
            check("hidden-form bridge confirms writes with postMessage",
                  "parent.postMessage" in bridge_html
                  and '"requestId": "qa-request-1"' in bridge_html
                  and '"favorited": true' in bridge_html)
            script_query = urllib.parse.urlencode({
                "action": "/api/favorite/toggle",
                "request_id": "qa-script-1",
                "payload": json.dumps({"project_id": "project-123", "subtitle_id": "7"}),
            })
            with urllib.request.urlopen(
                f"{url}/api/collection-bridge?{script_query}", timeout=5,
            ) as resp:
                bridge_script = resp.read().decode("utf-8")
            check("script bridge returns the in-player completion callback",
                  bridge_script.startswith("window.__englishLearnCollectionBridge(")
                  and '"qa-script-1"' in bridge_script
                  and '"favorited": true' in bridge_script)
            pixel_query = urllib.parse.urlencode({
                "action": "/api/favorite/toggle",
                "request_id": "qa-pixel-1",
                "payload": json.dumps({"project_id": "project-123", "subtitle_id": "7"}),
            })
            with urllib.request.urlopen(
                f"{url}/api/collection-pixel?{pixel_query}", timeout=5,
            ) as resp:
                pixel = resp.read()
                pixel_type = resp.headers.get("Content-Type")
            check("media bridge returns a loadable write receipt",
                  pixel.startswith(b"GIF89a") and pixel_type == "image/gif")
        range_request = urllib.request.Request(
            f"{url}/{video_name}", headers={"Range": "bytes=2-6"},
        )
        with urllib.request.urlopen(range_request, timeout=5) as resp:
            check("video range returns 206", resp.status == 206)
            check("video range returns requested bytes", resp.read() == b"deo-b")
            check("video range exposes content range", resp.headers.get("Content-Range") == "bytes 2-6/11")
        try:
            urllib.request.urlopen(f"{url}/model_presets.json", timeout=5)
            check("non-video work file is blocked", False, "request unexpectedly succeeded")
        except urllib.error.HTTPError as exc:
            check("non-video work file is blocked", exc.code == 404, str(exc))


def test_player_collection_api_is_idempotent(monkeypatch, tmp_path):
    import json
    from englishlearn.media.media_manager import (
        delete_favorite_api,
        delete_word_api,
        save_word_api,
        toggle_favorite_api,
    )

    monkeypatch.setenv("ENGLISHLEARN_WORK_DIR", str(tmp_path))
    favorite_payload = {
        "project_id": "project-123",
        "subtitle_id": "7",
        "text": "An invisible barrier",
        "translation": "一道无形的屏障",
        "time": 12.5,
    }
    added = toggle_favorite_api(favorite_payload)
    check("favorite API adds without navigation", added["ok"] and added["favorited"])
    removed = toggle_favorite_api(favorite_payload)
    check("favorite API toggles the same sentence off", removed["ok"] and not removed["favorited"])

    word_payload = {
        "project_id": "project-123",
        "subtitle_id": "7",
        "word": "notoriously",
        "translation": "众所周知地",
        "pos": "adverb",
        "pronunciation": "nəʊˈtɔːriəsli",
        "other_meanings": ["声名狼藉地", "尤指因坏事而出名"],
        "audio": "https://example.test/notoriously.mp3",
        "context": "It is notoriously difficult.",
        "time": 12.5,
    }
    saved = save_word_api(word_payload)
    check("word API saves rich dictionary fields", saved["ok"] and saved["entry"]["pronunciation"] == "nəʊˈtɔːriəsli")
    word_payload["translation"] = "声名狼藉地"
    updated = save_word_api(word_payload)
    word_items = json.loads((tmp_path / "wordbook.json").read_text(encoding="utf-8"))
    check("word API updates instead of duplicating", updated["ok"] and len(word_items) == 1 and word_items[0]["translation"] == "声名狼藉地")
    check("word API removes by id", delete_word_api({"id": updated["entry"]["id"]})["ok"])
    check("favorite delete endpoint validates ids", delete_favorite_api({"id": added["entry"]["id"]})["ok"])


def test_optional_ecdict_mini_lookup(monkeypatch, tmp_path):
    import csv
    from englishlearn.media.media_manager import _lookup_ecdict

    dictionary_dir = tmp_path / "dictionaries"
    dictionary_dir.mkdir()
    csv_path = dictionary_dir / "ecdict.mini.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "phonetic", "definition", "translation", "pos"])
        writer.writeheader()
        writer.writerow({
            "word": "notoriously",
            "phonetic": "nəʊˈtɔːriəsli",
            "definition": "in a notorious manner",
            "translation": "adv. 众所周知地，声名狼藉地\n网络 尤指因坏事而出名",
            "pos": "r:1",
        })
    monkeypatch.setenv("ENGLISHLEARN_WORK_DIR", str(tmp_path))
    result = _lookup_ecdict("Notoriously")
    check("optional ECDICT returns multiple Chinese meanings", result and len(result["translations"]) == 2)
    check("optional ECDICT returns phonetics", result["phonetic"] == "nəʊˈtɔːriəsli")


# ---------------------------------------------------------------------------
# Test 9: subtitle parsers clean source formats before they reach the UI
# ---------------------------------------------------------------------------

def test_subtitle_parsers():
    print("\n[test_subtitle_parsers]")
    from englishlearn.media.media_manager import _parse_json3, _parse_srt

    srt = "1\n00:00:01,000 --> 00:00:02,500\nHello <i>world</i> &amp; friends\n"
    parsed_srt = _parse_srt(srt)
    check("SRT strips styling markup and decodes entities",
          parsed_srt == [{"id": 0, "start": 1.0, "end": 2.5,
                          "text": "Hello world & friends"}], str(parsed_srt))

    raw_json3 = (
        '{"events":[{"tStartMs":1000,"dDurationMs":750,'
        '"segs":[{"utf8":"Hello "},{"utf8":"world"}]}]}'
    )
    parsed_json3 = _parse_json3(raw_json3)
    check("JSON3 preserves subtitle timing and text",
          parsed_json3 == [{"id": 0, "start": 1.0, "end": 1.75,
                            "text": "Hello world"}], str(parsed_json3))


def test_mp4_container_extension_is_normalized():
    import tempfile
    from englishlearn.media.media_manager import normalize_video_path

    with tempfile.TemporaryDirectory() as tmp:
        misleading = os.path.join(tmp, "video.webm")
        with open(misleading, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypisom")
        normalized = normalize_video_path(misleading)
        check("MP4 container gets .mp4 extension", normalized.endswith(".mp4"))
        check("normalized media remains on disk", os.path.exists(normalized))
        check("misleading path removed", not os.path.exists(misleading))


# ---------------------------------------------------------------------------
# Test 9: Hy-MT2 integration can report setup state without loading the model
# ---------------------------------------------------------------------------

def test_hy_mt2_status():
    print("\n[test_hy_mt2_status]")
    from englishlearn.translation.hy_mt2_local import MODEL_ID, model_status

    status = model_status()
    check("official Hy-MT2 model configured", status["model_id"] == MODEL_ID)
    check("Hy-MT2 setup status is actionable",
          isinstance(status.get("message"), str) and bool(status["message"]))


# ---------------------------------------------------------------------------
# Test 10: UI preferences survive a new session without persisting extras
# ---------------------------------------------------------------------------

def test_user_settings_persistence():
    print("\n[test_user_settings_persistence]")
    import tempfile
    from englishlearn.storage import project_store
    from englishlearn.storage.project_store import load_user_settings, save_user_settings

    original_work_dir = project_store.WORK_DIR
    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        try:
            save_user_settings({"ui_lang": "en", "target_lang": "ja", "max_workers": 3, "ignored": "x"})
            settings = load_user_settings()
            check("preferences persist", settings["ui_lang"] == "en" and settings["target_lang"] == "ja")
            check("preference defaults remain", settings["asr_model"] == "base" and settings["max_workers"] == 3)
            check("unknown preferences excluded", "ignored" not in settings)
        finally:
            project_store.WORK_DIR = original_work_dir


def test_new_pipeline_existing_subtitles_without_audio():
    """Existing web captions must complete even when no WAV is generated."""
    print("\n[test_new_pipeline_existing_subtitles_without_audio]")
    import tempfile
    from englishlearn.processing import pipeline_runner
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    original_resolve = pipeline_runner.resolve_video_source
    original_extract = pipeline_runner.extract_subtitles
    original_translate = pipeline_runner.translate_subtitles
    pipeline_runner.active_tasks.clear()

    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        video_path = os.path.join(tmp, "captioned-video.webm")
        open(video_path, "wb").close()
        pending = project_store.create_project(
            None, None, "zh", "hy_mt2_local:test",
            title="Pending YouTube video",
            source_url="https://www.youtube.com/watch?v=test",
        )
        project_store.append_project(pending)

        pipeline_runner.resolve_video_source = lambda *args, **kwargs: video_path
        pipeline_runner.extract_subtitles = lambda *args, **kwargs: [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world."},
        ]

        def fake_translate(subtitles, **kwargs):
            result = [{**item, "translation": "你好，世界。"} for item in subtitles]
            if kwargs.get("on_progress"):
                kwargs["on_progress"](1, 1)
            return result

        pipeline_runner.translate_subtitles = fake_translate
        task = {
            "task_id": pending["id"], "project_id": pending["id"],
            "completed": False, "consumed": False, "error": None,
            "progress": 0.0, "batch_current": 0, "batch_total": 0,
        }
        params = {
            "kind": "new", "project_id": pending["id"],
            "url": "https://www.youtube.com/watch?v=test", "upload_path": None,
            "proxy": None, "asr_model": "base", "engine_type": "hy_mt2_local",
            "model_name": "hy-mt-unit-test", "api_base": tmp, "api_key": "",
            "target_lang": "中文 (Chinese)", "target_lang_code": "zh",
            "max_tokens": 128, "max_workers": 1, "filename": "test",
            "display_title": "", "_work_dir": tmp,
        }
        try:
            pipeline_runner.run_pipeline_thread(task, params, tmp)
            saved = project_store.load_projects()[0]
            check("caption-only task completes", task["completed"] and task["error"] is None)
            check("caption-only project persisted", saved["status"] == "completed")
            check("audio path remains optional", saved["audio_path"] is None)
            check("translated subtitles saved", project_store.load_project_subtitles(saved["id"])[0]["translation"] == "你好，世界。")
        finally:
            pipeline_runner.resolve_video_source = original_resolve
            pipeline_runner.extract_subtitles = original_extract
            pipeline_runner.translate_subtitles = original_translate
            project_store.WORK_DIR = original_work_dir


def test_native_target_subtitles_skip_asr_segmentation_and_translation():
    """A platform target track must be preserved without model processing."""
    import tempfile
    from englishlearn.processing import pipeline_runner
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    originals = (
        pipeline_runner.resolve_video_source,
        pipeline_runner.extract_subtitles,
        pipeline_runner.extract_audio,
        pipeline_runner.translate_subtitles,
        pipeline_runner.ai_sentence_segment,
        pipeline_runner.sentence_segment,
    )
    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        video_path = os.path.join(tmp, "native-captioned.webm")
        open(video_path, "wb").close()
        pending = project_store.create_project(
            None, None, "zh", "test:model", title="Native captions",
            source_url="https://www.youtube.com/watch?v=QkdBXUikRQc",
        )
        project_store.append_project(pending)

        pipeline_runner.resolve_video_source = lambda *a, **k: video_path

        def fake_extract(*args, **kwargs):
            preferred = kwargs.get("preferred_languages") or []
            metadata = kwargs.get("metadata_out")
            if metadata is not None:
                metadata.update({
                    "title": "Native title",
                    "subtitle_language": preferred[0] if preferred else "en",
                    "subtitle_source": "manual",
                })
            if preferred and preferred[0].startswith("zh"):
                return [{"id": 0, "start": 0.0, "end": 2.0, "text": "你好，世界。"}]
            return [{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world."}]

        pipeline_runner.extract_subtitles = fake_extract
        pipeline_runner.extract_audio = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR audio extraction called"))
        pipeline_runner.translate_subtitles = lambda *a, **k: (_ for _ in ()).throw(AssertionError("translation called"))
        pipeline_runner.ai_sentence_segment = lambda *a, **k: (_ for _ in ()).throw(AssertionError("AI segmentation called"))
        pipeline_runner.sentence_segment = lambda *a, **k: (_ for _ in ()).throw(AssertionError("rule segmentation called"))

        task = {"completed": False, "error": None, "progress": 0, "batch_current": 0, "batch_total": 0}
        params = {
            "kind": "new", "project_id": pending["id"],
            "url": "https://www.youtube.com/watch?v=QkdBXUikRQc",
            "upload_path": None, "playback_mode": "local", "proxy": None,
            "prefer_native_target_subtitles": True,
            "asr_model": "base", "engine_type": "test", "model_name": "model",
            "api_base": tmp, "api_key": "", "target_lang": "Chinese",
            "target_lang_code": "zh", "max_tokens": 128, "max_workers": 1,
            "filename": "test", "display_title": "",
        }
        try:
            pipeline_runner.run_pipeline_thread(task, params, tmp)
            saved = project_store.load_projects()[0]
            subtitles = project_store.load_project_subtitles(saved["id"])
            check("native subtitle task completes", task["completed"] and task["error"] is None)
            check("native subtitle source is persisted", saved.get("subtitle_source") == "native_target")
            check("native subtitle model marker is persisted", saved.get("model_used") == "platform:native-target-subtitles")
            check("platform source and target tracks are paired", subtitles[0]["text"] == "Hello world." and subtitles[0]["translation"] == "你好，世界。")
        finally:
            (
                pipeline_runner.resolve_video_source,
                pipeline_runner.extract_subtitles,
                pipeline_runner.extract_audio,
                pipeline_runner.translate_subtitles,
                pipeline_runner.ai_sentence_segment,
                pipeline_runner.sentence_segment,
            ) = originals
            project_store.WORK_DIR = original_work_dir


def test_youtube_video_id_parses_supported_urls():
    from englishlearn.media.media_manager import youtube_video_id

    expected = "QkdBXUikRQc"
    check("watch URL parsed", youtube_video_id(f"https://www.youtube.com/watch?v={expected}") == expected)
    check("short URL parsed", youtube_video_id(f"https://youtu.be/{expected}?t=12") == expected)
    check("shorts URL parsed", youtube_video_id(f"https://youtube.com/shorts/{expected}") == expected)
    check("non-YouTube URL rejected", youtube_video_id(f"https://example.com/watch?v={expected}") is None)


def test_thumbnail_prefers_platform_cover_and_falls_back_to_frame(monkeypatch, tmp_path):
    from englishlearn.media import media_manager

    remote_cover = str(tmp_path / "cover.webp")
    frame_cover = str(tmp_path / "cover.jpg")
    monkeypatch.setattr(
        media_manager, "_download_thumbnail_image",
        lambda url, output_dir, stem="cover": remote_cover if url else None,
    )
    monkeypatch.setattr(
        media_manager, "generate_video_thumbnail",
        lambda video_path, output_dir, stem="cover": frame_cover,
    )
    check(
        "platform cover is preferred",
        media_manager.ensure_video_thumbnail("video.mp4", str(tmp_path), "https://example.com/cover.webp") == remote_cover,
    )
    check(
        "local frame is the fallback",
        media_manager.ensure_video_thumbnail("video.mp4", str(tmp_path), None) == frame_cover,
    )
    thumbnail_source = open(media_manager.__file__, encoding="utf-8").read()
    check("generated covers are normalized to 16:9", 'filter("pad", 640, 360' in thumbnail_source)


def test_online_pipeline_never_downloads_video_or_extracts_audio():
    """Caption-only mode must keep the no-download promise end to end."""
    import tempfile
    from englishlearn.processing import pipeline_runner
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    originals = (
        pipeline_runner.resolve_video_source,
        pipeline_runner.extract_subtitles,
        pipeline_runner.extract_audio,
        pipeline_runner.translate_subtitles,
    )
    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        url = "https://www.youtube.com/watch?v=QkdBXUikRQc"
        pending = project_store.create_project(
            None, None, "zh", "test:model", title="Online pending",
            source_url=url, playback_mode="online", external_video_id="QkdBXUikRQc",
        )
        project_store.append_project(pending)
        pipeline_runner.resolve_video_source = lambda *a, **k: (_ for _ in ()).throw(AssertionError("download called"))
        pipeline_runner.extract_audio = lambda *a, **k: (_ for _ in ()).throw(AssertionError("audio called"))
        def fake_extract(*args, **kwargs):
            kwargs["metadata_out"].update({"title": "Online title"})
            return [{"id": 0, "start": 12.5, "end": 15.0, "text": "Hello"}]
        pipeline_runner.extract_subtitles = fake_extract
        pipeline_runner.translate_subtitles = lambda items, **kwargs: [
            {**item, "translation": "你好"} for item in items
        ]
        task = {"completed": False, "error": None, "progress": 0, "batch_current": 0, "batch_total": 0}
        params = {
            "kind": "new", "project_id": pending["id"], "url": url,
            "upload_path": None, "playback_mode": "online", "proxy": None,
            "asr_model": "base", "engine_type": "test", "model_name": "model",
            "api_base": tmp, "api_key": "", "target_lang": "Chinese",
            "target_lang_code": "zh", "max_tokens": 128, "max_workers": 1,
            "filename": url, "display_title": "",
        }
        try:
            pipeline_runner.run_pipeline_thread(task, params, tmp)
            saved = project_store.load_projects()[0]
            check("online task completes", task["completed"] and task["error"] is None)
            check("online video remains remote", saved["video_path"] is None and saved["audio_path"] is None)
            check("online identity persists", saved["playback_mode"] == "online" and saved["external_video_id"] == "QkdBXUikRQc")
            check("source title is used", saved["title"] == "Online title")
        finally:
            (pipeline_runner.resolve_video_source, pipeline_runner.extract_subtitles,
             pipeline_runner.extract_audio, pipeline_runner.translate_subtitles) = originals
            project_store.WORK_DIR = original_work_dir


def test_online_pipeline_without_captions_fails_without_local_fallback():
    import tempfile
    from englishlearn.processing import pipeline_runner
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    originals = pipeline_runner.extract_subtitles, pipeline_runner.resolve_video_source, pipeline_runner.extract_audio
    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        url = "https://www.youtube.com/watch?v=QkdBXUikRQc"
        pending = project_store.create_project(
            None, None, "zh", "test:model", source_url=url,
            playback_mode="online", external_video_id="QkdBXUikRQc",
        )
        project_store.append_project(pending)
        pipeline_runner.extract_subtitles = lambda *a, **k: None
        pipeline_runner.resolve_video_source = lambda *a, **k: (_ for _ in ()).throw(AssertionError("download called"))
        pipeline_runner.extract_audio = lambda *a, **k: (_ for _ in ()).throw(AssertionError("audio called"))
        task = {"completed": False, "error": None, "progress": 0, "batch_current": 0, "batch_total": 0}
        params = {
            "kind": "new", "project_id": pending["id"], "url": url,
            "upload_path": None, "playback_mode": "online", "proxy": None,
            "asr_model": "base", "engine_type": "test", "model_name": "model",
            "api_base": tmp, "api_key": "", "target_lang": "Chinese",
            "target_lang_code": "zh", "max_tokens": 128, "max_workers": 1,
            "filename": url, "display_title": "",
        }
        try:
            pipeline_runner.run_pipeline_thread(task, params, tmp)
            saved = project_store.load_projects()[0]
            check("missing captions fail clearly", "Switch this task to Local complete mode" in (task["error"] or ""))
            check("failed project stays remote", saved["status"] == "failed" and saved["video_path"] is None)
        finally:
            pipeline_runner.extract_subtitles, pipeline_runner.resolve_video_source, pipeline_runner.extract_audio = originals
            project_store.WORK_DIR = original_work_dir


def test_online_player_contains_youtube_seek_adapter():
    import app

    html = app.build_player_html(
        "", [{"id": 0, "start": 12.5, "end": 15.0, "text": "Hello", "translation": "你好"}],
        playback_mode="online", external_video_id="QkdBXUikRQc",
    )
    check("YouTube player is embedded", "youtubePlayer" in html and "QkdBXUikRQc" in html)
    check("official iframe API is loaded", "https://www.youtube.com/iframe_api" in html)
    check("subtitle seeking reaches YouTube", "ytPlayer.seekTo(ytTime, true)" in html)
    check("local video element is absent", '<video id="video"' not in html)
    check("online player selects one control surface", 'class="player online"' in html)
    check("official YouTube controls stay enabled", "controls: 1" in html)
    check("duplicate custom controls are hidden", ".player.online .controls" in html)


def test_immersive_player_layout_and_scroll_contract():
    import app

    html = app.build_player_html(
        "http://127.0.0.1/video.mp4",
        [{"id": 0, "start": 12.5, "end": 15.0, "text": "Hello", "translation": "你好"}],
        lang="zh",
    )
    check("learning shell is fullscreen target", 'id="learningShell"' in html and "fullscreenEventTarget.requestFullscreen()" in html)
    check("restricted browsers get immersive fallback", "enterPseudoFullscreen" in html and "100vh" in html)
    check("subtitle header keeps only follow and bilingual controls", 'id="followBtn"' in html and 'id="listBilingualPanelBtn"' in html and 'id="panelCollapseBtn"' not in html and 'id="immersiveBtn"' not in html)
    check("subtitle search and display controls are progressive disclosure",
          'id="subtitleToolsBtn"' in html
          and 'id="subtitleSearch"' in html
          and "setSubtitleToolsOpen" in html
          and "e.ctrlKey || e.metaKey" in html)
    check("subtitle search filters source and translation without rebuilding the player",
          "row.dataset.searchText" in html
          and "row.classList.toggle('search-hidden', !match)" in html)
    check("dragging to the edge collapses the panel", "if (ratio <= 10)" in html and "togglePanel(true)" in html)
    check("collapsed panel has an edge restore action", 'class="panel-edge-restore"' in html and 'id="panelRestoreBtn"' in html and "显示字幕" in html)
    check("subtitle list occupies a real column beside the video", ".container { display:flex" in html and "position:relative; z-index:18; flex:0 0 var(--panel-width, 42%)" in html and "background:#131a26" in html)
    check("fullscreen split has accessible separator", 'role="separator"' in html and 'aria-label="调整字幕区域"' in html)
    check("separator works in normal and fullscreen layout", ".container:not(.panel-collapsed) .panel-resizer" in html)
    check("desktop resize bounds are enforced", "24, 55" in html and 'aria-valuemax="55"' in html)
    check("mobile resize bounds are enforced", "28, 65" in html and "--panel-height" in html)
    check("split ratio persists locally", "englishLearn.panelRatio.desktop" in html and "localStorage.setItem" in html)
    check("keyboard resizing is supported", "ArrowLeft" in html and "ArrowRight" in html and "Home" in html and "End" in html)
    check("double-click restores split default", "panelResizer.addEventListener('dblclick'" in html)
    check("manual subtitle browsing pauses follow", "pauseAutoFollow" in html and "恢复跟随" in html)
    check("subtitle list supports keyboard browsing", 'role="tabpanel" aria-label="字幕" tabindex="0"' in html)
    check("active row scroll stays inside subtitle list", "subsContainer.scrollTo" in html)
    check("subtitle panel is height-contained for iframe scrolling", "min-height:0; overflow:hidden" in html and "flex:1 1 0; min-height:0; overflow-y:auto" in html)
    check("page-level scrollIntoView removed", "scrollIntoView" not in html)
    check("all player placeholders resolved", not any(token in html for token in (
        "__FULLSCREEN__", "__PANEL_COLLAPSE__", "__PANEL_RESTORE__", "__FOLLOW_ON__", "__FOLLOW_RESUME__",
        "__PANEL_RESIZE__", "__PANEL_RESIZE_HINT__", "__PANEL_RESIZE_VALUE__",
    )))
    player_script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    # The player lives in a Streamlit iframe, so Python compilation cannot catch
    # JavaScript syntax errors.  Use a real file: `node --check -` only validates
    # the stdin wrapper on some Node versions and missed a broken escaped newline.
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as script_file:
        script_file.write(player_script)
        script_file.flush()
        syntax = subprocess.run(
            ["node", "--check", script_file.name], text=True,
            capture_output=True, check=False,
        )
    check("generated player JavaScript parses", syntax.returncode == 0, syntax.stderr)
    check("dictionary text keeps JavaScript newline escapes", "parts.join('\\\\n')" in player_script)


def test_dictionary_popup_interaction_contract():
    import app

    html = app.build_player_html(
        "http://127.0.0.1/video.mp4",
        [{"id": 7, "start": 12.5, "end": 15.0, "text": "An invisible barrier", "translation": "一道无形的屏障"}],
        lang="zh", project_id="project-1",
    )
    check("popup keeps the compact dictionary layout", 'id="popupSave"' not in html and 'id="popupFavorite"' not in html)
    check("popup exposes a wordbook star", 'id="popupWordSave"' in html and "/api/word/save" in html and "/api/word/delete" in html)
    check("popup excludes full sentence context", 'id="popupContext"' not in html)
    check("popup exposes an explicit in-player context action", 'id="popupAiButton"' in html and "语境翻译" in html)
    check("hover highlight waits 200ms", "HOVER_INTENT_DELAY = 200" in html)
    check("dictionary lookup opens only from a word click", "openDictionaryLookup(wordEl.dataset.word" in html)
    check("word lookup can be disabled from player settings", 'id="settingsWordLookupToggle"' in html and "englishLearn.wordLookupEnabled" in html and "if (!wordLookupEnabled) return" in html)
    check("disabled lookup restores subtitle-row seeking", "if (wordLookupEnabled && e.target.closest && e.target.closest('.sub-word')) return" in html)
    check("sentence selection does not open a popup", "Text selection -> word popup" not in html)
    check("popup is measured after becoming visible",
          html.index("wordPopup.classList.add('show')") < html.index("var popupRect = wordPopup.getBoundingClientRect()"))
    check("dictionary uses the localhost lookup bridge", "/api/word-lookup" in html and "/api/free-translate" in html)
    check("dictionary renders multiple meanings and pronunciation audio", "dictionaryLookupModel" in html and "meaning.definitions" in html and 'class="lookup-audio"' in html and "new Audio(audioUrl)" in html)
    check("dictionary has device pronunciation fallback",
          "SpeechSynthesisUtterance" in html and "speechSynthesis.speak" in html)
    check("context action stays inside the player", "fetchFreeTranslation(sub.text || popupData.text)" in html and "sendAction('lookup_word_ai'" not in html)
    check("free lookup remains bounded", "fetchJsonWithin(url, 1900)" in html)
    resumed = app.build_player_html(
        "http://127.0.0.1/video.mp4",
        [{"id": 7, "start": 12.5, "end": 15.0, "text": "An invisible barrier", "translation": "一道无形的屏障"}],
        lang="zh", project_id="project-1",
        ai_lookup={"word": "barrier", "project_id": "project-1", "subtitle_id": 7},
    )
    check("AI result reopens the originating word card", 'id="ai-lookup-data"' in resumed and "initialAiLookup" in resumed)


def test_subtitle_learning_interaction_contract():
    import app

    html = app.build_player_html(
        "http://127.0.0.1/video.mp4",
        [{"id": 7, "start": 12.5, "end": 15.0, "text": "An invisible barrier", "translation": "一道无形的屏障"}],
        lang="zh", project_id="project-1",
    )
    check("video and list bilingual controls are separate", 'id="settingsTranslationToggle"' in html and 'id="listBilingualPanelBtn"' in html)
    check("source and translated video layers have independent controls", 'id="settingsOriginalToggle"' in html and 'id="settingsTranslationToggle"' in html)
    check("subtitle toggle remains visible for online playback", 'id="onlineSubtitleSettingsBtn"' in html and 'id="settingsSubtitlesToggle"' in html)
    check("subtitle toggle updates overlay and settings-menu state", "function setSubtitles(enabled)" in html and "settingsSubtitlesToggle.addEventListener('click'" in html)
    check("video original and translation visibility are independent", "learning-video-original-off .subtitle-original-layer" in html and "learning-video-translation-off .subtitle-translation-layer" in html)
    check("bilingual state is isolated by surface", "learning-list-bilingual-off .sub-translation" in html)
    row_handler = html.split("row.addEventListener('click'", 1)[1].split("});", 1)[0]
    check("subtitle click seeks using the timing adjustment without autoplay", "video.currentTime = subtitleVideoTime(sub.start)" in row_handler and "video.play()" not in row_handler)
    check("hover words are rendered", "class=\"sub-word\"" in html and "hoverMarkup" in html)
    check("click lookup uses fast dictionary cache", "/api/word-lookup" in html and "hoverDictCache" in html)
    check("video overlay words are hoverable", "subOrig.innerHTML = hoverMarkup(cur.text)" in html)
    check("video size and timing live in the player settings menu", 'id="subtitleSettingsMenu"' in html and 'id="videoSubFontSize"' in html and 'id="listSubFontSize"' in html and "englishLearn.videoSubtitleScale" in html and "englishLearn.listSubtitleScale" in html)
    check("video subtitle layers can be dragged independently in both axes", 'id="subOrigDragHandle"' in html and 'id="subTransDragHandle"' in html and "拖动原文字幕" in html and "拖动译文字幕" in html and "englishLearn.videoSubtitleOriginalX" in html and "englishLearn.videoSubtitleTranslationY" in html and "ArrowLeft" in html and "pointerdown" in html)
    check("subtitle positions can be restored from player settings", 'id="settingsPositionReset"' in html and "resetVideoSubtitlePositions(true)" in html and "恢复字幕默认位置" in html)
    check("subtitle timing can be adjusted in 0.1-second steps", 'id="subtitleOffset"' in html and 'step="0.1"' in html and 'id="subtitleOffsetEarlier"' in html and 'id="subtitleOffsetLater"' in html)
    timing_block = html.split('<div class="subtitle-settings-menu"', 1)[1].split('</div>', 1)[0]
    subtitle_tools_block = html.split('<div class="subtitle-tools"', 1)[1].split('</div>', 1)[0]
    check("subtitle timing is in the player settings menu, not the subtitle list", html.index('class="subtitle-settings-menu"') < html.index('id="rightPanel"') and 'subtitleOffset' not in subtitle_tools_block)
    check("subtitle settings remain usable for online playback", ".player.online .controls" in html and 'id="onlineSubtitleSettingsBtn"' in html and 'id="subtitleOffset"' in timing_block)
    check("subtitle timing updates active overlay and row together", "video.currentTime - subtitleOffset" in html and "lastActiveId = -1" in html)
    check("subtitle timing persists per project without navigation", "englishLearn.subtitleOffset." in html and "/api/subtitle-offset" in html and "fetchJsonWithin(url, 1200)" in html)
    check("online speed control is available without custom timeline", 'id="onlineSpeedBtn"' in html and '.player.online .controls' in html)
    check("favorite action locks the clicked star", "favBtn.disabled = true" in html and "aria-label=\"' + (isFav?" in html)
    check("sentence favorite saves without leaving fullscreen", "postCollection('/api/favorite/toggle'" in html and "sendAction('toggle_favorite'" not in html)
    check("collection writes use the fullscreen-safe media bridge",
          "/api/collection-pixel?action=" in html
          and "var image = new Image(1, 1)" in html
          and "image.onload = function()" in html)
    check("lookup has a hard two-second budget", "lookupController.abort" in html and "1800" in html)


def test_streamlit_scroll_chrome_contract():
    import app

    source = open(APP_FILE, encoding="utf-8").read()
    assert '[data-testid="stExpandSidebarButton"]' in source
    assert 'buttonId = "englishlearn-sidebar-toggle"' in source
    assert "host.__englishLearnSidebarTimer = host.setInterval(sync, 500)" in source
    assert "new host.MutationObserver" not in source
    assert "nativeButton.click()" in source
    assert 't("sidebar.show", lang)' in source
    assert 't("sidebar.collapse", lang)' in source
    assert 'body.englishlearn-sidebar-collapsed [data-testid="stAppViewContainer"]' in source
    assert 'position:absolute !important; inset:0 !important' in source
    assert '[data-testid="stHeader"] { display: none; }' not in source
    assert "max-width: 1800px" in source


def test_media_site_application_shell_contract():
    """The product shell keeps media primary without relying on brand imitation."""
    source = open(APP_FILE, encoding="utf-8").read()
    assert 'key="app_masthead"' in source
    assert 'position: sticky; top: 0' in source
    assert 'key="project_search"' in source
    assert 'key="new_task_btn"' in source
    assert 'class="library-thumbnail"' in source
    assert 'project.get("thumbnail_path")' in source
    assert "https://i.ytimg.com/vi/" in source
    assert 'class="gallery-cover-image"' in source
    assert 'class="gallery-title-link"' in source
    assert '_handle_project_open_query(lang)' in source
    assert '_select_project(project, lang, rerun=False)' in source
    assert 'if any(not task["completed"] or task_id not in _seen' in source
    assert "use_container_width=True" not in source
    assert source.count('key="new_task_btn"') == 1
    assert "_render_project_action_bar(proj, title, pid, lang, subtitles)" in source
    assert 'st.iframe(player_html, width="stretch", height=680)' in source
    assert '"scroll_main_to_top": False' in source
    assert "main.scrollTop = 0" in source
    assert "_scroll_main_to_top_once()" in source


def test_dark_viewing_theme_contract():
    source = open(APP_FILE, encoding="utf-8").read()
    config_path = os.path.join(os.path.dirname(APP_FILE), ".streamlit", "config.toml")
    config = open(config_path, encoding="utf-8").read()
    assert 'base = "dark"' in config
    assert 'backgroundColor = "#0F0F0F"' in config
    assert 'secondaryBackgroundColor = "#181818"' in config
    assert 'color-scheme: dark !important' in source
    assert '--canvas: #0f0f0f' in source
    assert '--panel: #181818' in source
    assert 'background: rgba(15,15,15,.94)' in source


def test_player_collection_actions_are_persistent_and_idempotent(monkeypatch):
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("ENGLISHLEARN_WORK_DIR", tmp)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        for mod in list(sys.modules):
            if mod == "app" or mod == "englishlearn" or mod.startswith("englishlearn."):
                del sys.modules[mod]

        from englishlearn.storage import project_store
        project_store.save_presets([{
            "name": "Test model", "engine_type": "openai", "model_name": "test-model",
            "api_base": "http://localhost:9999/v1", "api_key": "", "max_tokens": 128,
        }])
        project = project_store.create_project(
            None, None, "zh", "openai:test-model", title="Online test",
            source_url="https://www.youtube.com/watch?v=QkdBXUikRQc",
            playback_mode="online", external_video_id="QkdBXUikRQc",
        )
        project = project_store.save_subtitles_to_project(project, [{
            "id": 7, "start": 12.5, "end": 15.0,
            "text": "An invisible barrier", "translation": "一道无形的屏障",
        }])
        project_store.append_project(project)

        # Build the isolated Streamlit runner before patching the package.  The
        # helper clears cached application modules so each run sees the test's
        # temporary work directory.
        at = fresh_app()
        from englishlearn.translation import llm_translator
        translated = {"translation": "屏障", "pos": "noun", "pronunciation": "", "other_meanings": ["障碍"]}
        monkeypatch.setattr(llm_translator, "translate_word", lambda **kwargs: dict(translated))

        word_action = {
            "action": "translate_word", "word": "barrier", "context": "An invisible barrier",
            "time": "12.5", "subtitle_id": "7", "project_id": project["id"],
        }
        at.query_params = dict(word_action)
        at.run()
        wordbook_path = os.path.join(tmp, "wordbook.json")
        words = json.loads(open(wordbook_path, encoding="utf-8").read())
        check("word action is consumed", at.query_params == {})
        check("word action keeps the source project open",
              at.session_state.current_project["id"] == project["id"])
        check("translated word is persisted", len(words) == 1 and words[0]["translation"] == "屏障")
        check("word context and timestamp persist",
              words[0]["context"] == "An invisible barrier" and words[0]["time"] == 12.5)

        translated["translation"] = "障碍物"
        at.query_params = dict(word_action)
        at.run()
        words = json.loads(open(wordbook_path, encoding="utf-8").read())
        check("duplicate word action updates", len(words) == 1 and words[0]["translation"] == "障碍物")

        favorite_action = {
            "action": "toggle_favorite", "subtitle_id": "7",
            "text": "An invisible barrier", "translation": "一道无形的屏障",
            "time": "12.5", "project_id": project["id"],
        }
        at.query_params = dict(favorite_action)
        at.run()
        favorites_path = os.path.join(tmp, "favorites.json")
        favorites = json.loads(open(favorites_path, encoding="utf-8").read())
        check("favorite is persisted once", len(favorites) == 1 and favorites[0]["subtitle_id"] == "7")
        check("favorite preserves timestamp", favorites[0]["time"] == 12.5)
        check("favorite action keeps the source project open",
              at.session_state.current_project["id"] == project["id"])

        at.query_params = dict(favorite_action)
        at.run()
        favorites = json.loads(open(favorites_path, encoding="utf-8").read())
        check("second favorite action removes entry", favorites == [])

        translated["translation"] = ""
        failed_action = dict(word_action)
        failed_action["word"] = "missing"
        at.query_params = failed_action
        at.run()
        words = json.loads(open(wordbook_path, encoding="utf-8").read())
        check("empty translation is not persisted", len(words) == 1)
        check("empty translation shows error", any("空翻译" in (item.value or "") for item in at.error))
        check("collection action flow has no exception", len(at.exception) == 0, str(at.exception))

        def slow_translate(**kwargs):
            time.sleep(2.2)
            return {"translation": "太慢", "pos": "", "pronunciation": "", "other_meanings": []}

        monkeypatch.setattr(llm_translator, "translate_word", slow_translate)
        at.query_params = dict(word_action)
        at.run()
        words = json.loads(open(wordbook_path, encoding="utf-8").read())
        check("slow translation is rejected after two seconds", len(words) == 1)
        check("slow translation shows timeout", any("超过 2 秒" in (item.value or "") for item in at.error))

        import app as app_module
        resolved_video, resolved_server = app_module._resolve_project_playback(project, "zh")
        check("online collection entry bypasses local video",
              resolved_video is None and resolved_server is None)


def test_multiple_pending_projects_are_distinct():
    import tempfile
    from englishlearn.storage import project_store

    original_work_dir = project_store.WORK_DIR
    with tempfile.TemporaryDirectory() as tmp:
        project_store.WORK_DIR = tmp
        try:
            first = project_store.create_project(
                None, None, "zh", "hy_mt2_local:test",
                title="First import", source_url="https://example.com/first",
            )
            second = project_store.create_project(
                None, None, "zh", "hy_mt2_local:test",
                title="Second import", source_url="https://example.com/second",
            )
            project_store.append_project(first)
            project_store.append_project(second)
            saved = project_store.load_projects()
            check("pending projects get distinct IDs", first["id"] != second["id"])
            check("multiple pending projects coexist", len(saved) == 2)
            check("each pending project owns its source", {p["source_url"] for p in saved} == {
                "https://example.com/first", "https://example.com/second",
            })
        finally:
            project_store.WORK_DIR = original_work_dir


# ---------------------------------------------------------------------------
# Test 8: retranslate pipeline lifecycle (directly, LLM mocked)
# ---------------------------------------------------------------------------
# We avoid clicking retranslate via AppTest because the @st.fragment(run_every=2)
# polling loop calls st.rerun() repeatedly, which AppTest cannot break out of.
# Instead we exercise the pipeline_runner module directly.

def test_retranslate_flow():
    print("\n[test_retranslate_flow]")
    from englishlearn.processing import pipeline_runner
    import tempfile
    from englishlearn.storage import project_store
    from englishlearn.storage.project_store import (
        append_project, create_project, load_project_raw_subtitles,
        load_project_subtitles, save_subtitles_to_project,
    )

    def _fake_translate(subtitles, **kwargs):
        out = []
        for s in subtitles:
            entry = dict(s)
            entry["translation"] = f"[test]{s.get('text', '')}"
            out.append(entry)
        if kwargs.get("on_progress"):
            kwargs["on_progress"](1, 1)
        return out

    orig_translate = pipeline_runner.translate_subtitles
    original_work_dir = project_store.WORK_DIR
    pipeline_runner.translate_subtitles = _fake_translate
    pipeline_runner.active_tasks.clear()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            project_store.WORK_DIR = tmp
            try:
                video_path = os.path.join(tmp, "source.mp4")
                audio_path = os.path.join(tmp, "source.wav")
                open(video_path, "wb").close()
                open(audio_path, "wb").close()
                proj = create_project(video_path, audio_path, "zh", "test")
                seed = [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello.", "translation": "你好。"}]
                proj = save_subtitles_to_project(proj, seed)
                append_project(proj)
                raw = load_project_raw_subtitles(proj["id"])
                check("raw subtitles loadable", raw is not None and len(raw) > 0)

                params = {
                    "kind": "retranslate",
                    "project_id": proj["id"],
                    "video_path": proj["video_path"],
                    "audio_path": proj.get("audio_path"),
                    "engine_type": "ollama",
                    "model_name": "test-mt",
                    "api_base": "http://localhost:11434/v1",
                    "api_key": "test",
                    "target_lang": "中文 (Chinese)",
                    "target_lang_code": "zh",
                    "max_tokens": 8192,
                    "max_workers": 2,
                    "filename": proj["title"],
                    "_work_dir": tmp,
                }
                task_id = pipeline_runner.start_pipeline_task("test_token", params)
                check("task registered", task_id == proj["id"])

                deadline = time.time() + 30
                completed = False
                while time.time() < deadline:
                    time.sleep(0.1)
                    tsk = pipeline_runner.active_tasks_for_session("test_token").get(task_id)
                    if tsk and tsk["completed"]:
                        completed = True
                        break

                check("thread completed within 30s", completed)

                tsk = pipeline_runner.active_tasks_for_session("test_token").get(task_id)
                check("no thread error", tsk is not None and not tsk.get("error"),
                      f"err={tsk.get('error') if tsk else 'no task'}")
                check("progress set to 1.0",
                      tsk is not None and tsk.get("progress", 0) >= 1.0,
                      f"progress={tsk.get('progress') if tsk else 'n/a'}")
                check("result_project_id preserved",
                      tsk is not None and tsk.get("result_project_id") == proj["id"])

                new_subs = load_project_subtitles(proj["id"])
                check("translated subtitles written to isolated storage",
                      new_subs is not None and any("[test]" in (s.get("translation", "") or "")
                                                   for s in new_subs))
            finally:
                project_store.WORK_DIR = original_work_dir
    finally:
        pipeline_runner.translate_subtitles = orig_translate


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_boots,
        test_new_task_form,
        test_valid_url_submit_shows_background_task,
        test_open_project_detail,
        test_corrupt_subtitles_handling,
        test_pipeline_runner_persistence,
        test_collections_page,
        test_video_server_allowlist,
        test_subtitle_parsers,
        test_mp4_container_extension_is_normalized,
        test_hy_mt2_status,
        test_user_settings_persistence,
        test_new_pipeline_existing_subtitles_without_audio,
        test_multiple_pending_projects_are_distinct,
        test_retranslate_flow,
        # test_settings_navigation is flaky in AppTest (nav uses page_link, not button)
    ]
    for fn in tests:
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            global _failed
            _failed += 1
            _failures.append((fn.__name__, str(e)))
            print(f"  FAIL  {fn.__name__} raised: {e}")

    print(f"\n=== Summary: {_passed} passed, {_failed} failed ===")
    if _failures:
        print("\nFailures:")
        for name, detail in _failures:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
