import zipfile
from pathlib import Path

from scripts.build_release import build


def test_release_contains_double_click_launchers_and_no_private_data():
    archive = build("test")
    try:
        with zipfile.ZipFile(archive) as package:
            names = set(package.namelist())
        root = "EnglishLearn-test/"
        assert root + "EnglishLearn.command" in names
        assert root + "EnglishLearn-Windows.bat" in names
        assert root + "scripts/run_app_windows.ps1" in names
        assert root + "README.md" in names
        assert root + "work/.gitkeep" in names
        assert not any("subtitles_" in name for name in names)
        assert not any(name.endswith((".mp4", ".wav", ".safetensors")) for name in names)
        assert not any("projects.json" in name or "model_presets.json" in name for name in names)
        assert not any("/.venv/" in name or "/.tools/" in name for name in names)
    finally:
        archive.unlink(missing_ok=True)
        try:
            Path(archive).parent.rmdir()
        except OSError:
            pass
