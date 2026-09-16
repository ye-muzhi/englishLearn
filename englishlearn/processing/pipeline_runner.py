"""Background pipeline runner.

This module exists separately so its module-level state (``active_tasks``,
``active_tasks_lock``) persists across Streamlit script reruns. Streamlit
re-executes app.py from top to bottom on every rerun, which would reset any
state defined there. Imported modules are cached, so state here survives.

Thread-safety: each task dict is mutated in place by the background thread.
Dict key updates are atomic under Python's GIL, so reads from the main
thread are safe.
"""
import os
import threading
import time
import uuid
from pathlib import Path

from .asr_engine import ASREngine
from .ai_segmenter import ai_sentence_segment, can_use_ai_segment
from ..translation.llm_translator import translate_subtitles
from ..media.media_manager import (
    resolve_video_source, extract_subtitles, extract_audio, youtube_video_id,
    ensure_video_thumbnail,
)
from ..storage.project_store import (
    load_projects, append_project, update_project,
    create_project, save_subtitles_to_project,
    load_project_raw_subtitles, save_subtitle_layers,
)
from .subtitle_contract import normalize_raw_cues, normalize_segmented_cues
from .sentence_segmenter import sentence_segment, SUBTITLE_MIN_PAUSE, SUBTITLE_MAX_WORDS

# session_token -> {task_id -> task_dict}
active_tasks: dict[str, dict[str, dict]] = {}
active_tasks_lock = threading.Lock()


def _subtitle_language_candidates(language_code: str) -> list[str]:
    """Return common platform caption tags for a configured target language."""
    normalized = (language_code or "").strip().lower()
    variants = {
        "zh": ["zh-Hans", "zh-CN", "zh-Hant", "zh-TW", "zh"],
        "en": ["en", "en-US", "en-GB"],
        "ja": ["ja", "ja-JP"],
        "ko": ["ko", "ko-KR"],
        "fr": ["fr", "fr-FR", "fr-CA"],
        "de": ["de", "de-DE"],
        "es": ["es", "es-ES", "es-419", "es-MX"],
        "pt": ["pt-BR", "pt-PT", "pt"],
        "ru": ["ru", "ru-RU"],
        "ar": ["ar", "ar-SA"],
        "th": ["th", "th-TH"],
        "vi": ["vi", "vi-VN"],
    }
    return variants.get(normalized, [normalized]) if normalized else []


def _merge_native_caption_tracks(
    source_track: list[dict] | None,
    target_track: list[dict],
) -> list[dict]:
    """Pair platform-provided source and target tracks without machine translation.

    Platform tracks rarely share identical cue boundaries, so each source cue
    receives the target cues that overlap its time range. If the platform only
    exposes the target track, it remains untouched as the primary caption.
    """
    if not source_track:
        return [
            {**item, "id": index, "translation": ""}
            for index, item in enumerate(target_track)
        ]

    merged: list[dict] = []
    for index, source in enumerate(source_track):
        start = float(source.get("start", 0.0) or 0.0)
        end = max(start, float(source.get("end", start) or start))
        overlaps = [
            item for item in target_track
            if float(item.get("end", 0.0) or 0.0) > start
            and float(item.get("start", 0.0) or 0.0) < end
        ]
        if not overlaps and target_track:
            midpoint = (start + end) / 2
            nearest = min(
                target_track,
                key=lambda item: abs(
                    (float(item.get("start", 0.0) or 0.0)
                     + float(item.get("end", 0.0) or 0.0)) / 2 - midpoint
                ),
            )
            nearest_midpoint = (
                float(nearest.get("start", 0.0) or 0.0)
                + float(nearest.get("end", 0.0) or 0.0)
            ) / 2
            if abs(nearest_midpoint - midpoint) <= 2.5:
                overlaps = [nearest]
        translation_parts: list[str] = []
        for item in overlaps:
            text = str(item.get("text") or "").strip()
            if text and (not translation_parts or translation_parts[-1] != text):
                translation_parts.append(text)
        merged.append({
            **source,
            "id": index,
            "translation": " ".join(translation_parts),
        })
    return merged


def active_tasks_for_session(session_token: str) -> dict[str, dict]:
    with active_tasks_lock:
        return active_tasks.get(session_token, {})


def all_active_tasks() -> dict[str, dict]:
    """Return every in-process task for this local desktop app.

    The app is intentionally single-user and local. Progress therefore needs
    to remain visible after navigating to another page or opening another tab,
    instead of being hidden behind the Streamlit session that created it.
    Composite keys avoid collisions when two sessions process the same project.
    """
    with active_tasks_lock:
        return {
            f"{session_token}:{task_id}": task
            for session_token, session_tasks in active_tasks.items()
            for task_id, task in session_tasks.items()
        }


def run_pipeline_thread(task: dict, params: dict, work_dir: str) -> None:
    """Background thread entry point. Writes status to *task*; never calls st.*."""
    try:
        kind = params["kind"]
        task["step"] = 1
        task["stage"] = "source"
        task["step_label"] = "Resolving source"

        native_target_used = False
        subtitles = None
        if kind == "new":
            playback_mode = params.get("playback_mode", "local")
            source_metadata = {}
            source_url = params.get("url", "").strip()
            thumbnail_dir = os.path.join(
                work_dir, "projects", params.get("project_id") or "pending",
            )
            thumbnail_path = None
            if playback_mode == "online":
                external_video_id = youtube_video_id(source_url)
                if not external_video_id:
                    raise RuntimeError("Online playback currently supports YouTube links only.")
                video_path = None
                audio_path = None
                task["filename"] = source_url
            elif params.get("upload_path"):
                video_path = params["upload_path"]
                audio_path = None
            else:
                video_path = resolve_video_source(
                    source_url, work_dir,
                    proxy=params.get("proxy") or None,
                    metadata_out=source_metadata,
                    thumbnail_dir=thumbnail_dir,
                )
                audio_path = None

            task["video_path"] = video_path
            if video_path:
                task["filename"] = Path(video_path).name
                thumbnail_path = source_metadata.get("thumbnail_path")
                if not thumbnail_path:
                    thumbnail_path = ensure_video_thumbnail(
                        video_path, thumbnail_dir,
                        source_metadata.get("thumbnail_url"),
                    )
            subtitles_raw = None

            if source_url:
                task["stage"] = "subtitles"
                task["step_label"] = "Checking existing subtitles"
                if params.get("prefer_native_target_subtitles"):
                    target_track = extract_subtitles(
                        source_url, work_dir,
                        proxy=params.get("proxy") or None,
                        metadata_out=source_metadata,
                        preferred_languages=_subtitle_language_candidates(
                            params.get("target_lang_code", "")
                        ),
                        strict_preferred=True,
                    )
                    if target_track:
                        source_track = None
                        if params.get("target_lang_code") != "en":
                            source_track = extract_subtitles(
                                source_url, work_dir,
                                proxy=params.get("proxy") or None,
                                preferred_languages=_subtitle_language_candidates("en"),
                                strict_preferred=True,
                            )
                        subtitles_raw = source_track or target_track
                        subtitles = _merge_native_caption_tracks(source_track, target_track)
                        native_target_used = True
                        task["native_target_used"] = True
                        task["step_label"] = "Using original target-language subtitles"

                if not native_target_used:
                    subtitles_raw = extract_subtitles(
                        source_url, work_dir,
                        proxy=params.get("proxy") or None,
                        metadata_out=source_metadata,
                    )

            if playback_mode == "online" and not subtitles_raw:
                raise RuntimeError(
                    "No usable subtitles were found. Switch this task to Local complete mode "
                    "to download the video and transcribe its audio."
                )
            if playback_mode == "online" and not thumbnail_path:
                thumbnail_path = ensure_video_thumbnail(
                    None, thumbnail_dir, source_metadata.get("thumbnail_url"),
                )
            task["thumbnail_path"] = thumbnail_path
            task["thumbnail_url"] = source_metadata.get("thumbnail_url") or None
        elif kind == "reprocess":
            video_path = params["video_path"]
            audio_path = params["audio_path"]
            subtitles_raw = None
        else:  # retranslate
            task["step"] = 3
            task["stage"] = "segment"
            subtitles_raw = load_project_raw_subtitles(params["project_id"])
            video_path = params["video_path"]
            audio_path = params["audio_path"]

            if subtitles_raw is None:
                raise RuntimeError("Raw subtitles are unavailable. Re-process the video before translating again.")

        if subtitles_raw is None:
            if kind == "reprocess" and audio_path and os.path.exists(audio_path):
                pass
            else:
                task["step"] = 2
                task["stage"] = "audio"
                task["step_label"] = "Extracting audio"
                audio_path = extract_audio(video_path, work_dir)
            task["audio_path"] = audio_path

            task["step"] = 3
            task["stage"] = "transcribe"
            task["step_label"] = "Transcribing"
            task["progress"] = 0.0
            engine = ASREngine(model_size=params["asr_model"])
            engine.load_model()
            subtitles_raw = engine.transcribe(audio_path)
            task["raw_count"] = len(subtitles_raw)

        # Freeze the source layer before any segmentation changes cue
        # boundaries. Every later cue keeps explicit lineage back to this set.
        source_subtitles_raw = normalize_raw_cues(subtitles_raw)
        subtitles_raw = source_subtitles_raw

        has_word_timing = any(
            isinstance(item.get("words"), list) and bool(item.get("words"))
            for item in subtitles_raw
        )
        _seg_model = params["model_name"]
        if native_target_used:
            task["step"] = 4
            task["stage"] = "native_subtitles"
            task["step_label"] = "Using original target-language subtitles"
            task["progress"] = 1.0
            model_used = "platform:native-target-subtitles"
        else:
            if can_use_ai_segment(_seg_model) and not has_word_timing:
                task["step"] = 3
                task["stage"] = "segment"
                task["step_label"] = "AI segmentation"
                task["progress"] = 0.0
                n_raw = len(subtitles_raw)
                def _seg_cb(cur, tot, _t=task):
                    _t["progress"] = cur / tot if tot else 1.0
                subtitles_raw = ai_sentence_segment(
                    subtitles_raw,
                    model=_seg_model,
                    api_base=params["api_base"],
                    api_key=params["api_key"],
                    on_progress=_seg_cb,
                    max_workers=params.get("max_workers", 6),
                )
                task["segmented_count"] = len(subtitles_raw)
                task["raw_count_before_segment"] = n_raw
            else:
                task["step"] = 3
                task["stage"] = "segment"
                task["step_label"] = (
                    "Timing-aligned segmentation"
                    if has_word_timing else "Rule-based segmentation"
                )
                subtitles_raw = sentence_segment(
                    subtitles_raw,
                    min_pause_sec=SUBTITLE_MIN_PAUSE,
                    max_words=SUBTITLE_MAX_WORDS,
                    keep_word_timing=True,
                )

            subtitles_raw = normalize_segmented_cues(
                subtitles_raw, source_subtitles_raw,
            )

            task["step"] = 4
            task["stage"] = "translate"
            task["step_label"] = "Translating"
            task["progress"] = 0.0
            def _trans_cb(cur, tot, _t=task):
                _t["progress"] = cur / tot if tot else 1.0
                _t["batch_current"] = cur
                _t["batch_total"] = tot
            subtitles = translate_subtitles(
                subtitles_raw,
                engine=params["engine_type"],
                model=_seg_model,
                api_base=params["api_base"],
                api_key=params["api_key"],
                target_lang=params["target_lang"],
                max_tokens=params.get("max_tokens", 32768),
                on_progress=_trans_cb,
                max_workers=params.get("max_workers", 6),
            )

            missing = [s for s in subtitles if s.get("text", "").strip() and not s.get("translation", "").strip()]
            if missing:
                raise RuntimeError(
                    f"Translation returned no text for {len(missing)} subtitle segment(s). "
                    "Please check the selected model and try again."
                )

            model_used = f"{params['engine_type']}:{params['model_name']}"
        segmented_subtitles = normalize_segmented_cues(
            subtitles if native_target_used else subtitles_raw,
            source_subtitles_raw,
        )
        if kind == "retranslate":
            projects_list = load_projects()
            proj = next((p for p in projects_list if p["id"] == params["project_id"]), None)
            if proj is None:
                raise RuntimeError("The project no longer exists in the library.")
            proj["model_used"] = model_used
            proj["target_lang"] = params["target_lang_code"]
            proj["timing_source"] = "word" if has_word_timing else "cue"
            save_subtitle_layers(
                proj, source_subtitles_raw, segmented_subtitles, subtitles,
            )
            update_project(proj)
            result_pid = params["project_id"]
        elif kind == "reprocess":
            projects_list = load_projects()
            proj = next((p for p in projects_list if p["id"] == params["project_id"]), None)
            if proj is None:
                raise RuntimeError("The project no longer exists in the library.")
            # Keep the same project ID so favorites, wordbook entries, and deep links stay valid.
            proj["video_path"] = os.path.abspath(video_path)
            proj["audio_path"] = os.path.abspath(audio_path)
            proj["target_lang"] = params["target_lang_code"]
            proj["model_used"] = model_used
            # A manual correction made for the previous transcription should
            # not be applied again to a newly word-aligned timeline.
            proj["subtitle_offset"] = 0.0
            proj["timing_source"] = "word" if has_word_timing else "cue"
            proj = save_subtitles_to_project(
                proj, subtitles,
                raw_subtitles=source_subtitles_raw,
                segmented_subtitles=segmented_subtitles,
            )
            update_project(proj)
            result_pid = params["project_id"]
        else:
            projects_list = load_projects()
            proj = next((p for p in projects_list if p["id"] == params.get("project_id")), None)
            if proj is None:
                # Backward-compatible fallback for callers that predate pending
                # project creation.
                proj = create_project(
                    video_path, audio_path, params["target_lang_code"], model_used,
                    title=params.get("display_title") or params.get("filename") or None,
                    source_url=params.get("url") or None,
                )
                append_project(proj)
            proj["video_path"] = os.path.abspath(video_path) if video_path else None
            proj["audio_path"] = os.path.abspath(audio_path) if audio_path else None
            proj["playback_mode"] = params.get("playback_mode", "local")
            proj["external_video_id"] = (
                youtube_video_id(params.get("url", ""))
                if proj["playback_mode"] == "online" else None
            )
            proj["target_lang"] = params["target_lang_code"]
            proj["model_used"] = model_used
            proj["thumbnail_path"] = (
                os.path.abspath(thumbnail_path) if thumbnail_path else None
            )
            proj["thumbnail_url"] = source_metadata.get("thumbnail_url") or None
            proj["subtitle_source"] = "native_target" if native_target_used else "generated"
            proj["native_subtitle_language"] = (
                source_metadata.get("subtitle_language") if native_target_used else None
            )
            proj["timing_source"] = (
                "platform" if native_target_used
                else ("word" if has_word_timing else "cue")
            )
            proj["error"] = None
            if not proj.get("custom_title"):
                proj["title"] = (
                    source_metadata.get("title")
                    or (Path(video_path).stem if video_path else params.get("display_title"))
                    or proj["title"]
                )
            proj = save_subtitles_to_project(
                proj, subtitles,
                raw_subtitles=source_subtitles_raw,
                segmented_subtitles=segmented_subtitles,
            )
            update_project(proj)
            result_pid = proj["id"]

        task["result_project_id"] = result_pid
        task["completed"] = True
        task["completed_at"] = time.time()
        task["consumed"] = False
        task["stage"] = "complete"
        task["step_label"] = "Completed"
        task["progress"] = 1.0

    except Exception as exc:
        task["error"] = str(exc)
        task["completed"] = True
        task["completed_at"] = time.time()
        task["consumed"] = False
        task["stage"] = "failed"
        task["step_label"] = f"Failed: {exc}"
        project_id = params.get("project_id")
        if project_id:
            projects_list = load_projects()
            proj = next((p for p in projects_list if p.get("id") == project_id), None)
            if proj is not None:
                if task.get("video_path"):
                    proj["video_path"] = os.path.abspath(task["video_path"])
                if task.get("audio_path"):
                    proj["audio_path"] = os.path.abspath(task["audio_path"])
                if task.get("thumbnail_path"):
                    proj["thumbnail_path"] = os.path.abspath(task["thumbnail_path"])
                if task.get("thumbnail_url"):
                    proj["thumbnail_url"] = task["thumbnail_url"]
                proj["status"] = "failed"
                proj["error"] = str(exc)
                update_project(proj)


def start_pipeline_task(session_token: str, params: dict) -> str:
    """Spawn a background thread to run the pipeline. Returns task_id."""
    task_id = params.get("project_id") or f"new_{uuid.uuid4().hex[:8]}"
    task = {
        "task_id": task_id,
        "kind": params["kind"],
        "project_id": params.get("project_id"),
        "step": 0,
        "stage": "starting",
        "step_label": "Starting",
        "progress": 0.0,
        "started_at": time.time(),
        "completed": False,
        "consumed": False,
        "error": None,
        "result_project_id": None,
        "video_path": params.get("video_path"),
        "audio_path": params.get("audio_path"),
        "thumbnail_path": None,
        "thumbnail_url": None,
        "filename": params.get("filename", ""),
        "batch_current": 0,
        "batch_total": 0,
        "raw_count": 0,
        "segmented_count": 0,
    }
    with active_tasks_lock:
        active_tasks.setdefault(session_token, {})[task_id] = task
    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(task, params, params.get("_work_dir", ".")),
        daemon=True,
    )
    thread.start()
    return task_id
