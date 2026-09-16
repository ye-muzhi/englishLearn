"""Build a source release that contains no private user data or local models."""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INCLUDE_FILES = [
    ".python-version", ".gitignore", "AGENTS.md", "FEATURES.md", "README.md",
    "requirements.txt", "app.py", "EnglishLearn.command", "EnglishLearn-Windows.bat",
]
INCLUDE_DIRS = [".streamlit", ".github", "desktop", "docs", "englishlearn", "scripts"]
EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", ".venv", ".tools", "work", "dist",
    "node_modules", "target", "binaries",
}


def should_copy(path: Path) -> bool:
    return not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts) and path.suffix not in {".pyc", ".pyo"}


def build(version: str) -> Path:
    dist = ROOT / "dist"
    package_name = f"EnglishLearn-{version}"
    staging = dist / package_name
    archive = dist / f"{package_name}.zip"
    shutil.rmtree(staging, ignore_errors=True)
    dist.mkdir(exist_ok=True)
    archive.unlink(missing_ok=True)
    staging.mkdir()

    for name in INCLUDE_FILES:
        shutil.copy2(ROOT / name, staging / name)
    for directory in INCLUDE_DIRS:
        source = ROOT / directory
        destination = staging / directory
        shutil.copytree(source, destination, ignore=lambda folder, names: [
            name for name in names if not should_copy(Path(folder) / name)
        ])
    (staging / "work").mkdir()
    (staging / "work" / ".gitkeep").touch()

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(dist))
    shutil.rmtree(staging)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.1.0")
    args = parser.parse_args()
    print(build(args.version))
