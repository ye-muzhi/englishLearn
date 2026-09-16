"""Self-contained desktop sidecar entrypoint built with PyInstaller."""
from __future__ import annotations

import argparse
import os
import secrets
import sys
import threading
import time
from pathlib import Path


def _data_home() -> Path:
    configured = os.environ.get("ENGLISHLEARN_WORK_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.uname().sysname == "Darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "EnglishLearn"


def _watch_parent(parent_pid: int) -> None:
    """Avoid leaving the PyInstaller child alive after the desktop shell exits."""
    while True:
        time.sleep(0.5)
        if os.getppid() != parent_pid:
            os._exit(0)


def streamlit_options(port: int) -> dict[str, object]:
    """Options passed to Streamlit's programmatic CLI entrypoint."""
    return {
        "server_address": "127.0.0.1",
        "server_port": port,
        "server_fileWatcherType": "none",
        "server_headless": True,
        "browser_gatherUsageStats": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    data_home = _data_home()
    data_home.mkdir(parents=True, exist_ok=True)
    os.environ["ENGLISHLEARN_WORK_DIR"] = str(data_home)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    parent_pid = os.getppid()
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()

    # The native shell and future integrations use a private loopback-only API.
    # Its bearer token is generated per process and never persisted.
    api_token = secrets.token_urlsafe(32)
    os.environ["ENGLISHLEARN_API_TOKEN"] = api_token
    from englishlearn.api_server import serve

    api_server = serve(
        "127.0.0.1", int(os.environ.get("ENGLISHLEARN_API_PORT", "8511")), api_token,
    )

    from streamlit.web import cli as streamlit_cli

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    app_path = bundle_root / "app.py"
    try:
        streamlit_cli.main_run(
            str(app_path), args=[], **streamlit_options(args.port),
        )
    finally:
        api_server.shutdown()


if __name__ == "__main__":
    main()
