"""Build the Python runtime and name it for Tauri's current target triple."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def target_triple() -> str:
    try:
        return subprocess.check_output(["rustc", "--print", "host-tuple"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        machine = platform.machine().lower()
        if os.name == "nt":
            return "aarch64-pc-windows-msvc" if "arm" in machine else "x86_64-pc-windows-msvc"
        if platform.system() == "Darwin":
            return "aarch64-apple-darwin" if machine in {"arm64", "aarch64"} else "x86_64-apple-darwin"
        return "aarch64-unknown-linux-gnu" if machine in {"arm64", "aarch64"} else "x86_64-unknown-linux-gnu"


def build() -> Path:
    executable = "pyinstaller.exe" if os.name == "nt" else "pyinstaller"
    subprocess.run([
        str(ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / executable),
        "--noconfirm", "--clean", "--onefile", "--name", "englishlearn-server",
        "--collect-all", "streamlit", "--collect-all", "faster_whisper",
        "--collect-submodules", "englishlearn", "--paths", str(ROOT),
        "--add-data", f"{ROOT / 'app.py'}{os.pathsep}.",
        "--add-data", f"{ROOT / '.streamlit'}{os.pathsep}.streamlit",
        str(ROOT / "englishlearn" / "desktop_entry.py"),
    ], cwd=ROOT, check=True)
    extension = ".exe" if os.name == "nt" else ""
    source = ROOT / "dist" / f"englishlearn-server{extension}"
    destination = ROOT / "desktop" / "src-tauri" / "binaries" / f"englishlearn-server-{target_triple()}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


if __name__ == "__main__":
    print(build())
