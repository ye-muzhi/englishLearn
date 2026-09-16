"""Compatibility migration from the retired sentence-favorites surface."""
from __future__ import annotations

import hashlib
import re

from ..storage.collections_store import CollectionStore


def migrate_legacy_favorites(
    favorites_store: CollectionStore,
    notes_store: CollectionStore,
) -> int:
    """Copy each historical sentence favorite into Notes exactly once.

    The original file remains untouched so older builds can still read it.
    ``legacy_favorite_id`` makes repeated Streamlit executions idempotent.
    """
    favorites = favorites_store.load()
    if not isinstance(favorites, list) or not favorites:
        return 0
    notes = notes_store.load()
    if not isinstance(notes, list):
        notes = []
    migrated_ids = {
        str(note.get("legacy_favorite_id"))
        for note in notes
        if note.get("legacy_favorite_id")
    }
    added = 0
    for favorite in favorites:
        if not isinstance(favorite, dict):
            continue
        source = str(favorite.get("text") or "").strip()
        translation = str(favorite.get("translation") or "").strip()
        if not source and not translation:
            continue
        stable_id = str(favorite.get("id") or "").strip()
        if not stable_id:
            digest_source = "\0".join((
                str(favorite.get("project_id") or ""),
                str(favorite.get("subtitle_id") or ""),
                source,
                translation,
            ))
            stable_id = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
        if stable_id in migrated_ids:
            continue
        first_expression = re.split(r"[.!?。？！]+", source, maxsplit=1)[0].strip()
        title_source = first_expression if 1 <= len(first_expression) <= 72 else source
        title = (title_source or translation or "Learning note")[:60]
        body = "\n\n".join(part for part in (source, translation) if part)
        created_at = str(favorite.get("created_at") or "")
        try:
            timestamp = max(0.0, float(favorite.get("time", 0) or 0))
        except (TypeError, ValueError):
            timestamp = 0.0
        notes.append({
            "id": f"legacy-{stable_id}",
            "legacy_favorite_id": stable_id,
            "project_id": favorite.get("project_id") or "",
            "project_title": favorite.get("project_title") or "",
            "time": timestamp,
            "subtitle_id": str(favorite.get("subtitle_id") or ""),
            "source_text": source,
            "source_translation": translation,
            "title": title,
            "body": body,
            "raw_idea": "",
            "summary": translation or source,
            "tags": [],
            "key_points": [],
            "vocabulary": [],
            "refined_by": "legacy_favorite",
            "created_at": created_at,
            "updated_at": created_at,
        })
        migrated_ids.add(stable_id)
        added += 1
    if added:
        notes_store.save(notes)
    return added
