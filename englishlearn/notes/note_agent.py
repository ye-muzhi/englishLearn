"""Background voice-note transcription and AI refinement.

The worker is deliberately independent from Streamlit so recording and model
work survive page reruns. Public task state is persisted atomically; credentials
are kept only in memory and are never written to disk.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..processing.asr_engine import ASREngine
from ..translation.llm_translator import _detect_mode


_lock = threading.RLock()
_asr_slot = threading.Semaphore(1)
_llm_slot = threading.Semaphore(1)
_tasks: dict[str, dict[str, Any]] = {}
_private_params: dict[str, dict[str, Any]] = {}
_asr_models: dict[str, ASREngine] = {}
_store_path: Path | None = None

_PUBLIC_KEYS = {
    "id", "project_id", "status", "stage", "created_at", "updated_at",
    "completed", "error", "warning", "raw_text", "result", "audio_path",
}


def _snapshot(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in task.items()
        if key in _PUBLIC_KEYS
    }


def configure(work_directory: str | os.PathLike[str]) -> None:
    global _store_path
    desired = Path(work_directory).expanduser().resolve() / "note_tasks.json"
    with _lock:
        if desired == _store_path:
            return
        _store_path = desired
        _tasks.clear()
        _private_params.clear()
        if not desired.exists():
            return
        try:
            payload = json.loads(desired.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for saved in payload.get("tasks", []) if isinstance(payload, dict) else []:
            if not isinstance(saved, dict) or not saved.get("id"):
                continue
            task = {key: saved[key] for key in _PUBLIC_KEYS if key in saved}
            if not task.get("completed"):
                task.update({
                    "status": "interrupted",
                    "stage": "interrupted",
                    "completed": True,
                    "error": "The app stopped before this note was completed.",
                    "updated_at": time.time(),
                })
            _tasks[str(task["id"])] = task


def _persist_locked() -> None:
    if _store_path is None:
        return
    _store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "tasks": [_snapshot(task) for task in _tasks.values()]}
    temporary = _store_path.with_name(f".{_store_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, _store_path)
    finally:
        temporary.unlink(missing_ok=True)


def _update(task_id: str, **changes: Any) -> None:
    with _lock:
        task = _tasks[task_id]
        task.update(changes)
        task["updated_at"] = time.time()
        _persist_locked()


def save_audio(
    content: bytes, work_directory: str | os.PathLike[str], project_id: str,
) -> str:
    """Persist one private recording before handing it to the background worker."""
    directory = Path(work_directory).expanduser().resolve() / "projects" / project_id / "note_audio"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"note-{uuid.uuid4().hex}.wav"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    return str(destination)


def _draft_result(raw_text: str, context: dict[str, Any], warning: str = "") -> dict[str, Any]:
    compact = " ".join(raw_text.split()).strip()
    source_text = str(context.get("source_text") or "").strip()
    # A translation-only/no-credential fallback should still look like a note,
    # not a clipped transcript. Prefer the first complete source expression
    # ("Precisely" in the common case), then fall back to the learner's idea.
    title_source = source_text or compact
    first_expression = re.split(r"[.!?。？！]+", title_source, maxsplit=1)[0].strip()
    title_source = first_expression if 1 <= len(first_expression) <= 72 else title_source
    title = title_source[:60] + ("…" if len(title_source) > 60 else "")
    return {
        "title": title or "学习笔记",
        "body": compact,
        "summary": compact,
        "key_points": [],
        "vocabulary": [],
        "tags": [],
        "source_text": source_text,
        "source_translation": str(context.get("source_translation") or "").strip(),
        "refined_by": "draft",
        "warning": warning,
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```\w*\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("AI did not return a JSON note")
        value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("AI note response is not an object")
    return value


def refine_note(
    raw_text: str,
    context: dict[str, Any],
    *,
    engine: str,
    model: str,
    api_base: str,
    api_key: str,
    ui_language: str = "zh",
    max_tokens: int = 1400,
) -> dict[str, Any]:
    """Turn a learner's spoken thought into a concise, editable note."""
    raw_text = " ".join(str(raw_text or "").split()).strip()
    if not raw_text:
        raise ValueError("No spoken or written idea was provided")
    if engine == "hy_mt2_local" or _detect_mode(model) == "line":
        return _draft_result(
            raw_text, context,
            "The selected translation model cannot refine notes; an editable transcript was kept.",
        )
    if not api_key and engine == "openai":
        return _draft_result(
            raw_text, context,
            "The AI preset has no API key; an editable transcript was kept.",
        )

    import httpx

    local_endpoint = str(api_base or "").startswith((
        "http://localhost", "http://127.", "http://192.168.", "http://10.",
    ))
    client = OpenAI(
        base_url=api_base,
        api_key=api_key or "local",
        http_client=httpx.Client(proxy=None) if local_endpoint else None,
    )
    output_language = "Chinese" if ui_language == "zh" else "English"
    source = str(context.get("source_text") or "").strip()
    translation = str(context.get("source_translation") or "").strip()
    system = (
        "You are a language-learning note editor. Preserve the learner's own insight, "
        "correct obvious speech-recognition noise, and make the note concise and memorable. "
        "Do not invent definitions or examples. Return only one JSON object."
    )
    user = (
        f"Output language: {output_language}\n"
        f"Subtitle context: {source}\n"
        f"Subtitle translation: {translation}\n"
        f"Learner's idea: {raw_text}\n\n"
        "Return keys: title (short), body (clear Markdown note), summary (one sentence), "
        "key_points (array of short strings), vocabulary (array of objects with expression, "
        "meaning, alternatives), and tags (array of short strings)."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max(256, min(int(max_tokens or 1400), 3000)),
    )
    content = response.choices[0].message.content or ""
    value = _parse_json_object(content)
    result = _draft_result(raw_text, context)
    result.update({
        "title": str(value.get("title") or result["title"]).strip()[:120],
        "body": str(value.get("body") or result["body"]).strip(),
        "summary": str(value.get("summary") or result["summary"]).strip(),
        "key_points": [str(item).strip() for item in value.get("key_points", []) if str(item).strip()][:8],
        "vocabulary": [item for item in value.get("vocabulary", []) if isinstance(item, dict)][:8],
        "tags": [str(item).strip() for item in value.get("tags", []) if str(item).strip()][:8],
        "refined_by": f"{engine}:{model}",
        "warning": "",
    })
    return result


def _run(task_id: str) -> None:
    params = _private_params[task_id]
    try:
        raw_text = " ".join(str(params.get("raw_text") or "").split()).strip()
        audio_path = str(params.get("audio_path") or "")
        if audio_path:
            _update(task_id, status="working", stage="transcribing")
            with _asr_slot:
                model_size = str(params.get("asr_model") or "base")
                engine = _asr_models.get(model_size)
                if engine is None:
                    engine = ASREngine(model_size=model_size)
                    engine.load_model()
                    _asr_models[model_size] = engine
                segments = engine.transcribe(
                    audio_path,
                    initial_prompt="这是我的英语学习笔记。I am explaining a language-learning insight.",
                )
            transcribed = " ".join(str(item.get("text") or "").strip() for item in segments).strip()
            raw_text = " ".join(part for part in (raw_text, transcribed) if part).strip()
            if not raw_text:
                raise RuntimeError("No speech was detected in the recording.")
            _update(task_id, raw_text=raw_text)

        if not raw_text:
            raise RuntimeError("Write or record an idea before creating a note.")
        _update(task_id, status="working", stage="refining", raw_text=raw_text)
        try:
            with _llm_slot:
                result = refine_note(
                    raw_text,
                    params.get("context") or {},
                    engine=str(params.get("engine") or ""),
                    model=str(params.get("model") or ""),
                    api_base=str(params.get("api_base") or ""),
                    api_key=str(params.get("api_key") or ""),
                    ui_language=str(params.get("ui_language") or "zh"),
                    max_tokens=int(params.get("max_tokens") or 1400),
                )
            warning = str(result.get("warning") or "")
        except Exception as exc:
            warning = f"AI refinement failed; an editable transcript was kept: {exc}"
            result = _draft_result(raw_text, params.get("context") or {}, warning)
        _update(
            task_id, status="complete", stage="complete", completed=True,
            result=result, warning=warning, error=None,
        )
    except Exception as exc:
        _update(
            task_id, status="failed", stage="failed", completed=True,
            error=str(exc),
        )


def start(params: dict[str, Any]) -> str:
    task_id = uuid.uuid4().hex[:16]
    now = time.time()
    task = {
        "id": task_id,
        "project_id": params.get("project_id"),
        "status": "queued",
        "stage": "queued",
        "created_at": now,
        "updated_at": now,
        "completed": False,
        "error": None,
        "warning": "",
        "raw_text": str(params.get("raw_text") or ""),
        "result": None,
        "audio_path": params.get("audio_path") or None,
    }
    with _lock:
        _tasks[task_id] = task
        _private_params[task_id] = copy.deepcopy(params)
        _persist_locked()
    threading.Thread(target=_run, args=(task_id,), daemon=True).start()
    return task_id


def get(task_id: str) -> dict[str, Any] | None:
    with _lock:
        task = _tasks.get(task_id)
        return _snapshot(task) if task else None


def has_running_tasks() -> bool:
    with _lock:
        return any(not task.get("completed") for task in _tasks.values())


def clear_finished_tasks() -> bool:
    """Clear all note-job metadata/audio when no worker is still running."""
    with _lock:
        if any(not task.get("completed") for task in _tasks.values()):
            return False
        audio_paths = [str(task.get("audio_path") or "") for task in _tasks.values()]
        _tasks.clear()
        _private_params.clear()
        _persist_locked()
    for audio_path in audio_paths:
        if audio_path:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass
    return True


def discard(task_id: str, *, delete_audio: bool = True) -> None:
    with _lock:
        task = _tasks.pop(task_id, None)
        _private_params.pop(task_id, None)
        _persist_locked()
    if delete_audio and task and task.get("audio_path"):
        try:
            Path(str(task["audio_path"])).unlink(missing_ok=True)
        except OSError:
            pass


def _reset_for_tests() -> None:
    global _store_path
    with _lock:
        _tasks.clear()
        _private_params.clear()
        _asr_models.clear()
        _store_path = None
