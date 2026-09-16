"""Canonical subtitle layers, cue identities, and quality validation.

The player may keep compact numeric ``id`` values for rendering, but every
persisted cue also owns an immutable ``cue_id``.  Segment cues record the raw
cue identities that produced them so retranslation and future editing never
have to infer alignment from list positions.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable


SCHEMA_VERSION = 2
LONG_CUE_WORDS = 28
_STRONG_END = re.compile(r"[.!?。？！…][\"'”’）》」』】]*$")


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cue_digest(prefix: str, position: int, cue: dict, parents: Iterable[str]) -> str:
    payload = "\x1f".join((
        prefix,
        str(position),
        f"{_float(cue.get('start')):.3f}",
        f"{_float(cue.get('end')):.3f}",
        str(cue.get("text") or "").strip(),
        "\x1e".join(parents),
    ))
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def normalize_raw_cues(subtitles: list[dict]) -> list[dict]:
    """Return source cues with deterministic immutable identities."""
    normalized: list[dict] = []
    used: set[str] = set()
    for position, item in enumerate(subtitles or []):
        cue = dict(item)
        cue["id"] = position
        cue["start"] = round(_float(cue.get("start")), 3)
        cue["end"] = round(max(cue["start"], _float(cue.get("end"), cue["start"])), 3)
        cue["text"] = str(cue.get("text") or "").strip()
        requested = str(cue.get("cue_id") or "").strip()
        cue_id = requested if requested and requested not in used else _cue_digest("raw", position, cue, ())
        while cue_id in used:
            cue_id = _cue_digest("raw", position + len(used), cue, ())
        cue["cue_id"] = cue_id
        cue["schema_version"] = SCHEMA_VERSION
        used.add(cue_id)
        normalized.append(cue)
    return normalized


def _overlapping_source_ids(cue: dict, raw_cues: list[dict]) -> list[str]:
    start = _float(cue.get("start"))
    end = max(start, _float(cue.get("end"), start))
    overlaps = [
        str(source["cue_id"])
        for source in raw_cues
        if _float(source.get("end")) > start
        and _float(source.get("start")) < end
    ]
    if overlaps or not raw_cues:
        return overlaps
    midpoint = (start + end) / 2
    nearest = min(
        raw_cues,
        key=lambda source: abs(
            (_float(source.get("start")) + _float(source.get("end"))) / 2 - midpoint
        ),
    )
    return [str(nearest["cue_id"])]


def normalize_segmented_cues(
    subtitles: list[dict], raw_subtitles: list[dict],
) -> list[dict]:
    """Attach stable cue IDs and raw-source lineage to segmented cues."""
    raw_cues = normalize_raw_cues(raw_subtitles)
    normalized: list[dict] = []
    used: set[str] = set()
    for position, item in enumerate(subtitles or []):
        cue = dict(item)
        cue["id"] = position
        cue["start"] = round(_float(cue.get("start")), 3)
        cue["end"] = round(max(cue["start"], _float(cue.get("end"), cue["start"])), 3)
        cue["text"] = str(cue.get("text") or "").strip()
        parents = [str(value) for value in (cue.get("source_cue_ids") or []) if value]
        if not parents:
            parents = _overlapping_source_ids(cue, raw_cues)
        cue["source_cue_ids"] = parents
        requested = str(cue.get("cue_id") or "").strip()
        cue_id = requested if requested and requested not in used else _cue_digest("cue", position, cue, parents)
        while cue_id in used:
            cue_id = _cue_digest("cue", position + len(used), cue, parents)
        cue["cue_id"] = cue_id
        cue["schema_version"] = SCHEMA_VERSION
        used.add(cue_id)
        normalized.append(cue)
    return normalized


def normalize_translated_cues(
    subtitles: list[dict], segmented_subtitles: list[dict],
) -> list[dict]:
    """Align translated cues by cue identity and reject positional drift."""
    segmented = [dict(item) for item in segmented_subtitles]
    by_cue_id = {
        str(item.get("cue_id")): item
        for item in subtitles or []
        if item.get("cue_id")
    }
    by_numeric_id = {item.get("id"): item for item in subtitles or []}
    result: list[dict] = []
    for position, source in enumerate(segmented):
        translated = by_cue_id.get(str(source.get("cue_id")))
        if translated is None:
            translated = by_numeric_id.get(source.get("id"))
        if translated is None and position < len(subtitles or []):
            candidate = (subtitles or [])[position]
            if str(candidate.get("text") or "").strip() == str(source.get("text") or "").strip():
                translated = candidate
        cue = dict(source)
        cue["translation"] = str((translated or {}).get("translation") or "").strip()
        for key in ("speaker", "speaker_name"):
            if (translated or {}).get(key) and not cue.get(key):
                cue[key] = translated[key]
        cue.pop("words", None)
        cue.pop("wc", None)
        result.append(cue)
    return result


def build_quality_report(
    raw_subtitles: list[dict],
    segmented_subtitles: list[dict],
    translated_subtitles: list[dict],
) -> dict:
    """Return machine-readable subtitle integrity findings."""
    raw = normalize_raw_cues(raw_subtitles)
    segmented = normalize_segmented_cues(segmented_subtitles, raw)
    translated = normalize_translated_cues(translated_subtitles, segmented)
    issues: list[dict] = []
    seen: set[str] = set()
    previous_start = -1.0
    previous_end = -1.0
    raw_ids = {item["cue_id"] for item in raw}

    for cue in segmented:
        cue_id = cue["cue_id"]
        start = _float(cue.get("start"))
        end = _float(cue.get("end"))
        if cue_id in seen:
            issues.append({"code": "duplicate_cue_id", "cue_id": cue_id, "severity": "error"})
        seen.add(cue_id)
        if end <= start:
            issues.append({"code": "invalid_timing", "cue_id": cue_id, "severity": "error"})
        if start < previous_start:
            issues.append({"code": "out_of_order", "cue_id": cue_id, "severity": "error"})
        if start < previous_end - 0.15:
            issues.append({"code": "timing_overlap", "cue_id": cue_id, "severity": "warning"})
        text = str(cue.get("text") or "").strip()
        if not text:
            issues.append({"code": "missing_source", "cue_id": cue_id, "severity": "error"})
        if len(text.split()) > LONG_CUE_WORDS:
            issues.append({"code": "long_cue", "cue_id": cue_id, "severity": "warning"})
        if text and len(text.split()) > 12 and not _STRONG_END.search(text):
            issues.append({"code": "unfinished_long_cue", "cue_id": cue_id, "severity": "warning"})
        missing_parents = [parent for parent in cue.get("source_cue_ids", []) if parent not in raw_ids]
        if missing_parents:
            issues.append({"code": "missing_source_lineage", "cue_id": cue_id, "severity": "error"})
        previous_start, previous_end = start, max(previous_end, end)

    translated_ids = [str(item.get("cue_id")) for item in translated]
    if translated_ids != [str(item.get("cue_id")) for item in segmented]:
        issues.append({"code": "translation_alignment", "severity": "error"})
    requires_translation = any(
        str(cue.get("translation") or "").strip() for cue in translated
    )
    for cue in translated:
        if requires_translation and cue.get("text") and not str(cue.get("translation") or "").strip():
            issues.append({"code": "missing_translation", "cue_id": cue.get("cue_id"), "severity": "error"})

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "schema_version": SCHEMA_VERSION,
        "counts": {"raw": len(raw), "segmented": len(segmented), "translated": len(translated)},
        "errors": errors,
        "warnings": warnings,
        "passed": errors == 0,
        "issues": issues,
    }
