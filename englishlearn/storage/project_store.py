"""
Project Store: Persistence for project library and model presets.

Data layout:
    work/
      projects.json          # project index (list of project dicts)
      model_presets.json     # model preset configurations
      projects/
        {project_id}/        # per-project directory
          subtitles_raw.json
          subtitles_translated.json
"""
import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..paths import work_dir

# Absolute path to the work/ directory
WORK_DIR = str(work_dir())
_STORE_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PRESETS: list[dict] = [
    {
        "name": "Hy-MT2 1.8B (on-device)",
        "engine_type": "hy_mt2_local",
        "model_name": "tencent/Hy-MT2-1.8B",
        "api_base": os.path.join(WORK_DIR, "models", "Hy-MT2-1.8B"),
        "api_key": "",
        "max_tokens": 4096,
    },
    {
        "name": "hy-mt2 1.8B (fast)",
        "engine_type": "ollama",
        "model_name": "kaelri/hy-mt2:1.8b",
        "api_base": "http://192.168.101.202:11434/v1",
        "api_key": "ollama",
        "max_tokens": 8192,
    },
    {
        "name": "hy-mt2 7B (quality)",
        "engine_type": "ollama",
        "model_name": "kaelri/hy-mt2:7b",
        "api_base": "http://192.168.101.202:11434/v1",
        "api_key": "ollama",
        "max_tokens": 32768,
    },
    {
        "name": "OpenAI GPT-4o-mini",
        "engine_type": "openai",
        "model_name": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key": "",
        "max_tokens": 32768,
    },
]

DEFAULT_USER_SETTINGS: dict = {
    "ui_lang": "zh",
    "target_lang": "zh",
    "asr_model": "base",
    "proxy": "",
    "selected_preset_index": 0,
    "model_override": "",
    "max_workers": 6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _projects_json_path() -> str:
    return os.path.join(WORK_DIR, "projects.json")


def _presets_json_path() -> str:
    return os.path.join(WORK_DIR, "model_presets.json")


def _user_settings_json_path() -> str:
    return os.path.join(WORK_DIR, "user_settings.json")


def _projects_dir() -> str:
    p = os.path.join(WORK_DIR, "projects")
    os.makedirs(p, exist_ok=True)
    return p


def _project_dir(project_id: str) -> str:
    p = os.path.join(_projects_dir(), project_id)
    os.makedirs(p, exist_ok=True)
    return p


def _generate_project_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def _write_json(path: str, data: object) -> None:
    """Atomically persist JSON and keep local configuration private by default."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Model Presets CRUD
# ---------------------------------------------------------------------------

def load_presets() -> list[dict]:
    """Load model presets from work/model_presets.json.
    Creates the file with defaults if it doesn't exist."""
    path = _presets_json_path()
    if not os.path.exists(path):
        os.makedirs(WORK_DIR, exist_ok=True)
        save_presets(DEFAULT_PRESETS)
        return [dict(p) for p in DEFAULT_PRESETS]
    try:
        with open(path, "r", encoding="utf-8") as f:
            presets = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [dict(p) for p in DEFAULT_PRESETS]
    if not isinstance(presets, list):
        return [dict(p) for p in DEFAULT_PRESETS]
    # Non-destructive migration: existing remote presets remain untouched while
    # the on-device Hy-MT2 option becomes available immediately.
    if not any(p.get("engine_type") == "hy_mt2_local" for p in presets):
        presets.append(dict(DEFAULT_PRESETS[0]))
        save_presets(presets)
    return presets


def save_presets(presets: list[dict]) -> None:
    with _STORE_LOCK:
        _write_json(_presets_json_path(), presets)


def load_user_settings() -> dict:
    """Load durable, non-secret UI preferences with safe defaults."""
    path = _user_settings_json_path()
    if not os.path.exists(path):
        return dict(DEFAULT_USER_SETTINGS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_USER_SETTINGS)
    if not isinstance(saved, dict):
        return dict(DEFAULT_USER_SETTINGS)
    return {**DEFAULT_USER_SETTINGS, **{k: saved[k] for k in DEFAULT_USER_SETTINGS if k in saved}}


def save_user_settings(settings: dict) -> None:
    """Persist only allowlisted UI preferences; model keys remain in presets."""
    safe = {k: settings.get(k, DEFAULT_USER_SETTINGS[k]) for k in DEFAULT_USER_SETTINGS}
    with _STORE_LOCK:
        _write_json(_user_settings_json_path(), safe)


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

def load_projects() -> list[dict]:
    """Load the project index from work/projects.json.
    Returns empty list if the file doesn't exist.
    If the file doesn't exist and legacy files are detected, runs migration.
    """
    path = _projects_json_path()
    if not os.path.exists(path):
        # Attempt legacy migration
        legacy = _migrate_legacy_projects()
        if legacy:
            save_projects(legacy)
            return legacy
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_projects(projects: list[dict]) -> None:
    with _STORE_LOCK:
        _write_json(_projects_json_path(), projects)


def append_project(project: dict) -> None:
    """Append a project without losing another background task's update."""
    with _STORE_LOCK:
        projects = load_projects()
        projects = [p for p in projects if p.get("id") != project.get("id")]
        projects.append(project)
        _write_json(_projects_json_path(), projects)


def update_project(project: dict) -> bool:
    """Replace a project record in-place. Returns False when it no longer exists."""
    with _STORE_LOCK:
        projects = load_projects()
        for idx, existing in enumerate(projects):
            if existing.get("id") == project.get("id"):
                projects[idx] = project
                _write_json(_projects_json_path(), projects)
                return True
    return False


def update_project_subtitle_offset(project_id: str, offset_seconds: float) -> Optional[float]:
    """Persist a project's playback-only subtitle timing adjustment.

    Positive values delay subtitles; negative values advance them.  The
    bounded value keeps an accidental slider gesture from making a project
    appear permanently broken while still covering normal ASR/source drift.
    """
    try:
        offset = round(float(offset_seconds) * 10) / 10
    except (TypeError, ValueError):
        return None
    if not project_id or not -5.0 <= offset <= 5.0:
        return None
    with _STORE_LOCK:
        projects = load_projects()
        for project in projects:
            if project.get("id") == project_id:
                project["subtitle_offset"] = offset
                _write_json(_projects_json_path(), projects)
                return offset
    return None


def create_project(
    video_path: Optional[str],
    audio_path: Optional[str],
    target_lang: str,
    model_used: str,
    title: Optional[str] = None,
    source_url: Optional[str] = None,
    playback_mode: str = "local",
    external_video_id: Optional[str] = None,
) -> dict:
    """Create a project record, including a not-yet-downloaded pending project."""
    pid = _generate_project_id()
    fallback_title = Path(video_path).stem if video_path else (source_url or "New video")
    title = (title or fallback_title).strip() or fallback_title
    project = {
        "id": pid,
        "title": title,
        "custom_title": None,
        "source_url": source_url or None,
        "playback_mode": playback_mode if playback_mode in {"local", "online"} else "local",
        "external_video_id": external_video_id or None,
        "video_path": os.path.abspath(video_path) if video_path else None,
        "audio_path": os.path.abspath(audio_path) if audio_path else None,
        "thumbnail_path": None,
        "thumbnail_url": None,
        "subtitles_raw_path": None,
        "subtitles_translated_path": None,
        "target_lang": target_lang,
        "model_used": model_used,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
        "error": None,
    }
    return project


def save_subtitles_to_project(project: dict, subtitles: list[dict]) -> dict:
    """Write subtitles_raw.json and subtitles_translated.json
    into the project directory. Update project paths and status.
    Returns the updated project dict.
    """
    pid = project["id"]
    proj_dir = _project_dir(pid)

    raw_path = os.path.join(proj_dir, "subtitles_raw.json")
    raw_data = [
        {
            k: s[k]
            for k in ("id", "start", "end", "text", "words")
            if k in s
        }
        for s in subtitles
    ]
    _write_json(raw_path, raw_data)

    trans_path = os.path.join(proj_dir, "subtitles_translated.json")
    translated_data = [
        {k: value for k, value in subtitle.items() if k not in {"words", "wc"}}
        for subtitle in subtitles
    ]
    _write_json(trans_path, translated_data)

    project["subtitles_raw_path"] = raw_path
    project["subtitles_translated_path"] = trans_path
    project["status"] = "completed"
    return project


def load_project_subtitles(project_id: str) -> Optional[list[dict]]:
    """Load translated subtitles from a project's disk storage.

    Returns None if the file is missing, empty, or unparseable (e.g. left
    over from a crashed write).
    """
    proj_dir = _project_dir(project_id)
    trans_path = os.path.join(proj_dir, "subtitles_translated.json")
    if not os.path.exists(trans_path) or os.path.getsize(trans_path) == 0:
        return None
    try:
        with open(trans_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def load_project_raw_subtitles(project_id: str) -> Optional[list[dict]]:
    """Load the raw (pre-translation) subtitles from a project's disk storage."""
    proj_dir = _project_dir(project_id)
    raw_path = os.path.join(proj_dir, "subtitles_raw.json")
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return None
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def save_translated_subtitles(project: dict, subtitles: list[dict]) -> dict:
    """Write only subtitles_translated.json (preserves existing raw data).
    Used by the re-translate flow so that re-segmentation doesn't overwrite
    the original raw subtitles on disk.
    """
    pid = project["id"]
    proj_dir = _project_dir(pid)

    trans_path = os.path.join(proj_dir, "subtitles_translated.json")
    translated_data = [
        {k: value for k, value in subtitle.items() if k not in {"words", "wc"}}
        for subtitle in subtitles
    ]
    _write_json(trans_path, translated_data)

    project["subtitles_translated_path"] = trans_path
    project["status"] = "completed"
    return project


def update_project_title(project_id: str, custom_title: str) -> None:
    with _STORE_LOCK:
        projects = load_projects()
        for p in projects:
            if p["id"] == project_id:
                p["custom_title"] = custom_title or None
                _write_json(_projects_json_path(), projects)
                return


def has_subtitles(project: dict) -> bool:
    """Return True if the project has a non-empty translated subtitles file."""
    trans_path = project.get("subtitles_translated_path")
    if trans_path and os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
        return True
    proj_dir = _project_dir(project["id"])
    p = os.path.join(proj_dir, "subtitles_translated.json")
    return os.path.exists(p) and os.path.getsize(p) > 0


def _is_managed_path(path: str) -> bool:
    """Return whether a file is inside the app's work directory."""
    if not path:
        return False
    try:
        return os.path.commonpath((os.path.realpath(path), os.path.realpath(WORK_DIR))) == os.path.realpath(WORK_DIR)
    except ValueError:
        return False


def _remove_managed_file(path: Optional[str]) -> bool:
    """Delete one app-managed file, never a user file outside ``work/``."""
    if not path or not _is_managed_path(path):
        return False
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
            return True
    except OSError:
        pass
    return False


def _remove_project_directory(project_id: str) -> bool:
    if not project_id or project_id != os.path.basename(project_id):
        return False
    path = os.path.join(_projects_dir(), project_id)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True
    except OSError:
        pass
    return False


def delete_project(project_id: str) -> dict[str, int]:
    """Permanently delete a project and its app-managed media/subtitle files.

    A locally chosen original outside ``WORK_DIR`` is intentionally preserved.
    Downloaded/uploaded media inside ``work/`` is removed only when no other
    project still refers to the same path.
    """
    with _STORE_LOCK:
        projects = load_projects()
        removed = [p for p in projects if p.get("id") == project_id]
        remaining = [p for p in projects if p.get("id") != project_id]
        if not removed:
            return {"projects": 0, "files": 0, "directories": 0}
        referenced_paths = {
            os.path.realpath(path)
            for project in remaining
            for path in (project.get("video_path"), project.get("audio_path"))
            if path
        }
        files_removed = 0
        directories_removed = 0
        for project in removed:
            if _remove_project_directory(project.get("id", "")):
                directories_removed += 1
            for path in (project.get("video_path"), project.get("audio_path")):
                if path and os.path.realpath(path) not in referenced_paths and _remove_managed_file(path):
                    files_removed += 1
        _write_json(_projects_json_path(), remaining)
        return {"projects": len(removed), "files": files_removed, "directories": directories_removed}


def delete_all_project_data() -> dict[str, int]:
    """Delete every project plus media/subtitles created under ``work/``.

    Model downloads, presets, and user settings are deliberately left intact;
    this is a learning-library deletion, not an app reinstall.
    """
    with _STORE_LOCK:
        projects = load_projects()
        files_removed = 0
        directories_removed = 0
        for project in projects:
            if _remove_project_directory(project.get("id", "")):
                directories_removed += 1
            for path in (project.get("video_path"), project.get("audio_path")):
                if _remove_managed_file(path):
                    files_removed += 1
        uploads_dir = os.path.join(WORK_DIR, "uploads")
        try:
            if os.path.isdir(uploads_dir):
                shutil.rmtree(uploads_dir)
                directories_removed += 1
        except OSError:
            pass
        _write_json(_projects_json_path(), [])
        return {"projects": len(projects), "files": files_removed, "directories": directories_removed}


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

def _migrate_legacy_projects() -> list[dict]:
    """Scan work/ for legacy .mp4/.wav files and create project records."""
    if not os.path.isdir(WORK_DIR):
        return []

    legacy = []
    for f in sorted(os.listdir(WORK_DIR)):
        if not f.endswith(".mp4"):
            continue
        stem = f[:-4]
        mp4_path = os.path.join(WORK_DIR, f)
        wav_path = os.path.join(WORK_DIR, f"{stem}.wav")

        if not os.path.exists(wav_path):
            continue

        pid = f"legacy-{stem[:60]}"
        # Sanitize id for use as directory name
        pid = "".join(c if c.isalnum() or c in "_-" else "_" for c in pid).lower()
        project = {
            "id": pid,
            "title": stem,
            "custom_title": None,
            "video_path": os.path.abspath(mp4_path),
            "audio_path": os.path.abspath(wav_path),
            "subtitles_raw_path": None,
            "subtitles_translated_path": None,
            "target_lang": "zh",
            "model_used": "unknown",
            "created_at": datetime.fromtimestamp(
                os.path.getmtime(mp4_path), tz=timezone.utc
            ).isoformat(),
            "status": "legacy",
        }
        legacy.append(project)

    return legacy
