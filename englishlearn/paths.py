"""Canonical filesystem locations used across the application."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def work_dir() -> Path:
    """Return the configured data directory, defaulting to ``<project>/work``."""
    configured = os.environ.get("ENGLISHLEARN_WORK_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "work"
