"""Small authenticated local API for the desktop shell and diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .paths import work_dir
from .processing import pipeline_runner
from .storage.project_store import load_project_subtitle_quality, load_projects


def route_get(path: str, supplied_token: str, expected_token: str) -> tuple[int, dict]:
    """Pure routing core, kept independent from sockets for deterministic tests."""
    if path == "/health":
        return 200, {"ok": True, "version": __version__}
    if not (
        expected_token and supplied_token
        and secrets.compare_digest(expected_token, supplied_token)
    ):
        return 401, {"ok": False, "error": "unauthorized"}
    if path == "/api/v1/projects":
        return 200, {"ok": True, "projects": load_projects()}
    if path == "/api/v1/tasks":
        return 200, {"ok": True, "tasks": pipeline_runner.all_active_tasks()}
    if path.startswith("/api/v1/projects/") and path.endswith("/quality"):
        project_id = path.removeprefix("/api/v1/projects/").removesuffix("/quality").strip("/")
        report = load_project_subtitle_quality(project_id)
        return (200 if report else 404), {"ok": bool(report), "quality": report}
    return 404, {"ok": False, "error": "not_found"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "EnglishLearnAPI/1.0"

    def log_message(self, *_args) -> None:
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = getattr(self.server, "api_token", "")
        supplied = self.headers.get("X-EnglishLearn-Token", "")
        return bool(token and supplied and secrets.compare_digest(token, supplied))

    def do_GET(self) -> None:  # noqa: N802
        status, payload = route_get(
            self.path, self.headers.get("X-EnglishLearn-Token", ""),
            getattr(self.server, "api_token", ""),
        )
        self._send(status, payload)


def serve(host: str = "127.0.0.1", port: int = 8511, token: str = "") -> ThreadingHTTPServer:
    pipeline_runner.configure_task_store(str(work_dir()))
    server = ThreadingHTTPServer((host, port), _Handler)
    server.api_token = token or secrets.token_urlsafe(32)  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ENGLISHLEARN_API_PORT", "8511")))
    parser.add_argument("--token", default=os.environ.get("ENGLISHLEARN_API_TOKEN", ""))
    args = parser.parse_args()
    server = serve(args.host, args.port, args.token)
    print(f"EnglishLearn API ready on http://{args.host}:{args.port}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
