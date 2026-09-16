"""Fast, user-owned local dictionary index with conservative morphology."""
from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Optional


_INDEX_LOCK = threading.RLock()
_MAX_CSV_BYTES = 512 * 1024 * 1024


def word_forms(word: str) -> list[str]:
    """Return conservative English lookup candidates, exact form first."""
    value = str(word or "").strip().lower()
    forms = [value] if value else []

    def add(candidate: str) -> None:
        if len(candidate) >= 2 and candidate not in forms:
            forms.append(candidate)

    if value.endswith("'s"):
        add(value[:-2])
    if value.endswith("ies") and len(value) > 4:
        add(value[:-3] + "y")
    if value.endswith("ves") and len(value) > 4:
        add(value[:-3] + "f")
        add(value[:-3] + "fe")
    if value.endswith("ing") and len(value) > 5:
        stem = value[:-3]
        add(stem)
        add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            add(stem[:-1])
    if value.endswith("ied") and len(value) > 4:
        add(value[:-3] + "y")
    if value.endswith("ed") and len(value) > 4:
        stem = value[:-2]
        add(stem)
        add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            add(stem[:-1])
    if value.endswith("es") and len(value) > 4:
        add(value[:-2])
        add(value[:-1])
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        add(value[:-1])
    return forms[:10]


def _row_value(row: dict, name: str) -> str:
    return str(row.get(name) or "").strip()


def build_csv_index(csv_path: str, index_path: str) -> bool:
    """Build a compact SQLite index beside app data, atomically."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) > _MAX_CSV_BYTES:
        return False
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    source_mtime = os.stat(csv_path).st_mtime_ns
    if os.path.exists(index_path):
        try:
            with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
                stored = connection.execute(
                    "SELECT value FROM metadata WHERE key='source_mtime_ns'"
                ).fetchone()
                if stored and stored[0] == str(source_mtime):
                    return True
        except sqlite3.Error:
            pass

    with _INDEX_LOCK:
        fd, temp_path = tempfile.mkstemp(
            prefix=".dictionary-", suffix=".db", dir=os.path.dirname(index_path),
        )
        os.close(fd)
        try:
            connection = sqlite3.connect(temp_path)
            try:
                connection.executescript(
                    "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
                    "CREATE TABLE stardict ("
                    "word TEXT PRIMARY KEY COLLATE NOCASE, phonetic TEXT, "
                    "definition TEXT, translation TEXT, pos TEXT);"
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);"
                )
                batch: list[tuple[str, str, str, str, str]] = []
                with open(csv_path, encoding="utf-8-sig", newline="") as source:
                    for row in csv.DictReader(source):
                        word = _row_value(row, "word")
                        if not word:
                            continue
                        batch.append((
                            word, _row_value(row, "phonetic"),
                            _row_value(row, "definition"),
                            _row_value(row, "translation"), _row_value(row, "pos"),
                        ))
                        if len(batch) >= 5000:
                            connection.executemany(
                                "INSERT OR REPLACE INTO stardict VALUES (?,?,?,?,?)", batch,
                            )
                            batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT OR REPLACE INTO stardict VALUES (?,?,?,?,?)", batch,
                    )
                connection.execute(
                    "INSERT INTO metadata VALUES ('source_mtime_ns', ?)",
                    (str(source_mtime),),
                )
                connection.commit()
            finally:
                connection.close()
            os.replace(temp_path, index_path)
            return True
        except (OSError, csv.Error, sqlite3.Error, UnicodeError):
            return False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def lookup_sqlite(path: str, word: str) -> Optional[dict]:
    """Look up an exact or inflected form from an ECDICT-compatible DB."""
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.15) as connection:
            connection.row_factory = sqlite3.Row
            for form in word_forms(word):
                row = connection.execute(
                    "SELECT word, phonetic, definition, translation, pos FROM stardict "
                    "WHERE word = ? COLLATE NOCASE LIMIT 1", (form,),
                ).fetchone()
                if row:
                    result = dict(row)
                    result["query"] = word
                    result["matched_form"] = str(row["word"])
                    return result
    except sqlite3.Error:
        return None
    return None


def ensure_csv_index(csv_path: str, dictionary_dir: str) -> Optional[str]:
    fingerprint = Path(csv_path).stem.replace(" ", "_")
    index_path = os.path.join(dictionary_dir, f"{fingerprint}.indexed.db")
    return index_path if build_csv_index(csv_path, index_path) else None
