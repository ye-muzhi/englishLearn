"""CollectionStore: Generic JSON-backed CRUD for user collections.

Used by wordbook, favorites, and any future collection types.
Each entry gets an auto-generated `id` and `created_at` timestamp.
"""
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional


class CollectionStore:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.RLock()

    def _write(self, items: list[dict]) -> None:
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory or None)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.filepath)
            try:
                os.chmod(self.filepath, 0o600)
            except OSError:
                pass
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load(self) -> list[dict]:
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def save(self, items: list[dict]) -> None:
        with self._lock:
            self._write(items)

    def add(self, item: dict) -> dict:
        with self._lock:
            items = self.load()
            entry = dict(item)
            entry["id"] = entry.get("id") or uuid.uuid4().hex[:12]
            entry["created_at"] = datetime.now(timezone.utc).isoformat()
            items.append(entry)
            self._write(items)
            return entry

    def remove(self, item_id: str) -> None:
        with self._lock:
            items = self.load()
            self._write([i for i in items if i.get("id") != item_id])

    def update(self, item_id: str, **fields) -> Optional[dict]:
        with self._lock:
            items = self.load()
            for i in items:
                if i.get("id") == item_id:
                    i.update(fields)
                    self._write(items)
                    return i
        return None

    def find(self, **criteria) -> Optional[dict]:
        for item in self.load():
            if all(item.get(k) == v for k, v in criteria.items()):
                return item
        return None

    def find_all(self, **criteria) -> list[dict]:
        return [
            item for item in self.load()
            if all(item.get(k) == v for k, v in criteria.items())
        ]

    def remove_if(self, **criteria) -> bool:
        with self._lock:
            items = self.load()
            before = len(items)
            items = [i for i in items if not all(i.get(k) == v for k, v in criteria.items())]
            if before != len(items):
                self._write(items)
                return True
        return False
