"""Persistent subtitle editing with undo/redo and structural operations."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Optional

from ..paths import work_dir
from ..processing.subtitle_contract import build_quality_report
from .project_store import (
    load_project_raw_subtitles,
    load_project_segmented_subtitles,
    load_project_subtitles,
)


_LOCK = threading.RLock()
_MAX_HISTORY = 100


def _project_dir(project_id: str) -> str:
    return os.path.join(str(work_dir()), "projects", project_id)


def _history_path(project_id: str) -> str:
    return os.path.join(_project_dir(project_id), "subtitle_edits.json")


def _translated_path(project_id: str) -> str:
    return os.path.join(_project_dir(project_id), "subtitles_translated.json")


def _segmented_path(project_id: str) -> str:
    return os.path.join(_project_dir(project_id), "subtitles_segmented.json")


def _quality_path(project_id: str) -> str:
    return os.path.join(_project_dir(project_id), "subtitle_quality.json")


def _write_json(path: str, value: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".edit-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _load_history(project_id: str) -> dict:
    path = _history_path(project_id)
    if not os.path.exists(path):
        return {"version": 1, "undo": [], "redo": []}
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {"version": 1, "undo": [], "redo": []}
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "undo": [], "redo": []}


def history_status(project_id: str) -> dict:
    history = _load_history(project_id)
    return {"undo": len(history.get("undo", [])), "redo": len(history.get("redo", []))}


def _snapshot(subtitles: list[dict], operation: str) -> dict:
    return {
        "operation": operation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subtitles": copy.deepcopy(subtitles),
    }


def _persist(project_id: str, subtitles: list[dict]) -> None:
    for index, cue in enumerate(subtitles):
        cue["id"] = index
        cue["start"] = round(float(cue.get("start", 0)), 3)
        cue["end"] = round(max(cue["start"], float(cue.get("end", cue["start"]))), 3)
        cue.pop("words", None)
        cue.pop("wc", None)
    segmented = [
        {key: copy.deepcopy(value) for key, value in cue.items() if key != "translation"}
        for cue in subtitles
    ]
    _write_json(_segmented_path(project_id), segmented)
    _write_json(_translated_path(project_id), subtitles)
    raw = load_project_raw_subtitles(project_id) or subtitles
    _write_json(_quality_path(project_id), build_quality_report(raw, segmented, subtitles))


def _commit(project_id: str, before: list[dict], after: list[dict], operation: str) -> list[dict]:
    history = _load_history(project_id)
    undo = history.setdefault("undo", [])
    undo.append(_snapshot(before, operation))
    history["undo"] = undo[-_MAX_HISTORY:]
    history["redo"] = []
    _persist(project_id, after)
    _write_json(_history_path(project_id), history)
    return after


def update_cue(
    project_id: str, cue_id: str, *, text: str, translation: str,
    start: float, end: float,
) -> Optional[list[dict]]:
    with _LOCK:
        subtitles = load_project_subtitles(project_id)
        if not subtitles:
            return None
        index = next((i for i, cue in enumerate(subtitles) if str(cue.get("cue_id")) == cue_id), None)
        if index is None:
            return None
        start = round(max(0.0, float(start)), 3)
        end = round(max(start + 0.05, float(end)), 3)
        before = copy.deepcopy(subtitles)
        subtitles[index].update({
            "text": str(text).strip(), "translation": str(translation).strip(),
            "start": start, "end": end, "edited": True,
        })
        return _commit(project_id, before, subtitles, "update")


def split_cue(project_id: str, cue_id: str, position: int) -> Optional[list[dict]]:
    with _LOCK:
        subtitles = load_project_subtitles(project_id)
        if not subtitles:
            return None
        index = next((i for i, cue in enumerate(subtitles) if str(cue.get("cue_id")) == cue_id), None)
        if index is None:
            return None
        cue = subtitles[index]
        text = str(cue.get("text") or "")
        position = int(position)
        if position <= 0 or position >= len(text) or not text[:position].strip() or not text[position:].strip():
            return None
        before = copy.deepcopy(subtitles)
        ratio = max(0.1, min(0.9, position / len(text)))
        midpoint = float(cue["start"]) + (float(cue["end"]) - float(cue["start"])) * ratio
        base_id = str(cue.get("cue_id") or f"cue_{index}")
        left = {**cue, "cue_id": f"{base_id}.1", "text": text[:position].strip(), "end": round(midpoint, 3), "translation": "", "edited": True}
        right = {**cue, "cue_id": f"{base_id}.2", "text": text[position:].strip(), "start": round(midpoint, 3), "translation": "", "edited": True}
        subtitles[index:index + 1] = [left, right]
        return _commit(project_id, before, subtitles, "split")


def merge_with_next(project_id: str, cue_id: str) -> Optional[list[dict]]:
    with _LOCK:
        subtitles = load_project_subtitles(project_id)
        if not subtitles:
            return None
        index = next((i for i, cue in enumerate(subtitles) if str(cue.get("cue_id")) == cue_id), None)
        if index is None or index >= len(subtitles) - 1:
            return None
        before = copy.deepcopy(subtitles)
        left, right = subtitles[index], subtitles[index + 1]
        merged = {
            **left,
            "cue_id": f"{left.get('cue_id')}.m",
            "end": right.get("end", left.get("end")),
            "text": " ".join(filter(None, [str(left.get("text") or "").strip(), str(right.get("text") or "").strip()])),
            "translation": " ".join(filter(None, [str(left.get("translation") or "").strip(), str(right.get("translation") or "").strip()])),
            "source_cue_ids": list(dict.fromkeys((left.get("source_cue_ids") or []) + (right.get("source_cue_ids") or []))),
            "edited": True,
        }
        subtitles[index:index + 2] = [merged]
        return _commit(project_id, before, subtitles, "merge")


def undo(project_id: str) -> Optional[list[dict]]:
    return _move_history(project_id, "undo", "redo")


def redo(project_id: str) -> Optional[list[dict]]:
    return _move_history(project_id, "redo", "undo")


def _move_history(project_id: str, source: str, target: str) -> Optional[list[dict]]:
    with _LOCK:
        current = load_project_subtitles(project_id)
        history = _load_history(project_id)
        stack = history.get(source, [])
        if current is None or not stack:
            return None
        snapshot = stack.pop()
        history.setdefault(target, []).append(_snapshot(current, source))
        restored = snapshot.get("subtitles")
        if not isinstance(restored, list):
            return None
        _persist(project_id, restored)
        _write_json(_history_path(project_id), history)
        return restored
