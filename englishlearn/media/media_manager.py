"""
Media Manager: Video download & audio extraction.
"""
import functools
import csv
import http.server
import json
import os
import re
import socket
import shutil
import sqlite3
import tempfile
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import ffmpeg
import yt_dlp

from ..paths import work_dir
from .dictionary_index import ensure_csv_index, lookup_sqlite


# Public dictionary/translation services are intentionally accessed from this
# localhost process instead of the player iframe.  Browsers may block those
# keyless APIs because of CORS, even when they are reachable from the machine.
_LOOKUP_CACHE: dict[tuple[str, str, str], dict] = {}
_LOOKUP_CACHE_LOCK = threading.Lock()
_LOOKUP_CACHE_MAX = 512
_DIRECT_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_COLLECTION_API_LOCK = threading.RLock()
_FFMPEG_SETUP_LOCK = threading.Lock()
_FFMPEG_READY = False


def _ensure_ffmpeg() -> None:
    """Make ffmpeg/ffprobe available lazily without delaying application boot."""
    global _FFMPEG_READY
    if _FFMPEG_READY:
        return
    with _FFMPEG_SETUP_LOCK:
        if _FFMPEG_READY:
            return
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            _FFMPEG_READY = True
            return
        try:
            import static_ffmpeg
            # Downloads a platform-matched user-local binary on first media
            # operation; it never needs administrator access.
            static_ffmpeg.add_paths(weak=True)
        except ImportError as exc:
            raise RuntimeError(
                "FFmpeg is unavailable. Run the EnglishLearn installer again."
            ) from exc
        _FFMPEG_READY = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        if not _FFMPEG_READY:
            raise RuntimeError("FFmpeg setup did not complete successfully.")


def _fetch_public_json(url: str, timeout: float = 1.55) -> Optional[dict | list]:
    """Fetch a small public JSON response without leaking a proxy to the UI."""
    request = urllib.request.Request(url, headers={"User-Agent": "englishLearn/1.0"})
    try:
        # The app's optional SOCKS proxy is for video/model downloads. Public
        # dictionary lookup must not inherit it: a stale proxy can turn a fast
        # keyless lookup into a multi-second failure.
        with _DIRECT_HTTP_OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _safe_target_language(value: str) -> str:
    value = str(value or "zh-CN").strip()
    return value if re.fullmatch(r"[a-z]{2}(?:-[A-Z]{2})?", value) else "zh-CN"


def _cache_get(kind: str, text: str, target: str) -> Optional[dict]:
    with _LOOKUP_CACHE_LOCK:
        return _LOOKUP_CACHE.get((kind, text, target))


def _cache_put(kind: str, text: str, target: str, result: dict) -> dict:
    with _LOOKUP_CACHE_LOCK:
        if len(_LOOKUP_CACHE) >= _LOOKUP_CACHE_MAX:
            _LOOKUP_CACHE.pop(next(iter(_LOOKUP_CACHE)))
        _LOOKUP_CACHE[(kind, text, target)] = result
    return result


def _free_translate(text: str, target: str) -> str:
    """Translate through MyMemory, a free no-key fallback with a hard timeout."""
    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode({
        "q": text, "langpair": f"en|{target}",
    })
    response = _fetch_public_json(url)
    translated = response.get("responseData", {}).get("translatedText") if isinstance(response, dict) else ""
    return str(translated).strip() if translated else ""


def _dictionary_work_dir() -> str:
    return str(work_dir())


def _ecdict_candidates() -> list[str]:
    configured = os.environ.get("ENGLISHLEARN_ECDICT_PATH", "").strip()
    dictionary_dir = os.path.join(_dictionary_work_dir(), "dictionaries")
    candidates = [configured] if configured else []
    candidates.extend([
        os.path.join(dictionary_dir, "ecdict.db"),
        os.path.join(dictionary_dir, "stardict.db"),
        os.path.join(dictionary_dir, "ecdict.mini.csv"),
    ])
    return [path for path in candidates if path and os.path.isfile(path)]


def _normalise_ecdict_row(row: dict) -> Optional[dict]:
    if not row:
        return None
    translations = [line.strip() for line in str(row.get("translation") or "").splitlines() if line.strip()]
    definitions = [line.strip() for line in str(row.get("definition") or "").splitlines() if line.strip()]
    if not translations and not definitions:
        return None
    return {
        "word": str(row.get("word") or "").strip(),
        "phonetic": str(row.get("phonetic") or "").strip(),
        "translations": translations[:12],
        "definitions": definitions[:12],
        "pos": str(row.get("pos") or "").strip(),
        "source": "ECDICT",
        "matched_form": str(row.get("matched_form") or row.get("word") or "").strip(),
    }


def _lookup_ecdict(word: str) -> Optional[dict]:
    """Read an optional, user-installed MIT-licensed ECDICT dictionary."""
    key = word.lower()
    for path in _ecdict_candidates():
        try:
            if path.lower().endswith(".db"):
                row = lookup_sqlite(path, word)
                return _normalise_ecdict_row(row) if row else None
            index_path = ensure_csv_index(
                path, os.path.join(_dictionary_work_dir(), "dictionaries"),
            )
            row = lookup_sqlite(index_path, word) if index_path else None
            return _normalise_ecdict_row(row) if row else None
        except (OSError, csv.Error, sqlite3.Error, UnicodeError):
            continue
    return None


def free_translation_lookup(text: str, target: str = "zh-CN") -> dict:
    """Return a cached free translation. Input and total response size are bounded."""
    source = str(text or "").strip()[:900]
    target = _safe_target_language(target)
    if not source:
        return {"translation": ""}
    cached = _cache_get("translation", source.lower(), target)
    if cached is not None:
        return cached
    result = {"translation": _free_translate(source, target)}
    return _cache_put("translation", source.lower(), target, result) if result["translation"] else result


def free_word_lookup(word: str, target: str = "zh-CN") -> dict:
    """Look up dictionary data and a free translation in parallel under 2 s."""
    source = str(word or "").strip()[:48]
    target = _safe_target_language(target)
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{1,40}", source):
        return {"dictionary": None, "translation": ""}
    key = source.lower()
    cached = _cache_get("word", key, target)
    if cached is not None:
        return cached
    local_dictionary = _lookup_ecdict(source)
    if local_dictionary:
        local_translation = "；".join(local_dictionary.get("translations") or [])
        result = {
            "dictionary": None,
            "translation": local_translation,
            "local": local_dictionary,
            "lookup_source": "local",
        }
        return _cache_put("word", key, target, result)
    dictionary_url = "https://api.dictionaryapi.dev/api/v2/entries/en/" + urllib.parse.quote(source)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dictionary_task = executor.submit(_fetch_public_json, dictionary_url)
        translation_task = None if (local_dictionary and target.lower().startswith("zh")) else executor.submit(_free_translate, source, target)
        try:
            dictionary = dictionary_task.result(timeout=1.7)
        except Exception:
            dictionary = None
        try:
            translation = translation_task.result(timeout=1.7) if translation_task else ""
        except Exception:
            translation = ""
    result = {
        "dictionary": dictionary if isinstance(dictionary, list) else None,
        "translation": translation,
        "local": local_dictionary,
        "lookup_source": "public" if dictionary else ("translation" if translation else "none"),
    }
    return _cache_put("word", key, target, result) if result["dictionary"] or result["translation"] else result


def save_subtitle_offset(project_id: str, value: str) -> dict:
    """Persist a bounded player timing preference without reloading Streamlit."""
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", str(project_id or "")):
        return {"saved": False, "subtitle_offset": 0.0}
    try:
        requested = float(value)
    except (TypeError, ValueError):
        return {"saved": False, "subtitle_offset": 0.0}
    # Keep this import lazy: the media server is also used during project-store
    # import and must not introduce a module-level cycle.
    from ..storage.project_store import update_project_subtitle_offset
    saved = update_project_subtitle_offset(project_id, requested)
    return {"saved": saved is not None, "subtitle_offset": saved if saved is not None else 0.0}


def _clean_collection_payload(data: dict) -> dict:
    return {str(key): value for key, value in (data or {}).items() if isinstance(key, str)}


def _valid_collection_project(project_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9-]{3,64}", str(project_id or "")))


def _collection_store(filename: str):
    from ..storage.collections_store import CollectionStore
    return CollectionStore(os.path.join(_dictionary_work_dir(), filename))


def toggle_favorite_api(data: dict) -> dict:
    """Toggle one sentence favorite without navigating or reloading the player."""
    data = _clean_collection_payload(data)
    project_id = str(data.get("project_id") or "")
    subtitle_id = str(data.get("subtitle_id") or "")[:64]
    if not _valid_collection_project(project_id) or not subtitle_id:
        return {"ok": False, "favorited": False, "error": "invalid_request"}
    store = _collection_store("favorites.json")
    with _COLLECTION_API_LOCK:
        existing = next((item for item in store.load()
                         if str(item.get("project_id")) == project_id
                         and str(item.get("subtitle_id")) == subtitle_id), None)
        if existing:
            store.remove(existing["id"])
            return {"ok": True, "favorited": False, "id": existing["id"]}
        try:
            time_value = max(0.0, float(data.get("time") or 0))
        except (TypeError, ValueError):
            time_value = 0.0
        entry = store.add({
            "subtitle_id": subtitle_id,
            "text": str(data.get("text") or "").strip()[:4000],
            "translation": str(data.get("translation") or "").strip()[:4000],
            "time": time_value,
            "project_id": project_id,
        })
        return {"ok": True, "favorited": True, "entry": entry}


def delete_favorite_api(data: dict) -> dict:
    item_id = str((data or {}).get("id") or "")[:64]
    project_id = str((data or {}).get("project_id") or "")[:64]
    subtitle_id = str((data or {}).get("subtitle_id") or "")[:64]
    if not item_id and not (project_id and subtitle_id):
        return {"ok": False, "error": "invalid_request"}
    with _COLLECTION_API_LOCK:
        store = _collection_store("favorites.json")
        if item_id:
            store.remove(item_id)
        else:
            store.remove_if(project_id=project_id, subtitle_id=subtitle_id)
    return {"ok": True, "id": item_id, "project_id": project_id, "subtitle_id": subtitle_id}


def save_word_api(data: dict) -> dict:
    """Add or update a wordbook item directly from the dictionary popup."""
    data = _clean_collection_payload(data)
    project_id = str(data.get("project_id") or "")
    word = str(data.get("word") or "").strip()[:80]
    if not _valid_collection_project(project_id) or not re.fullmatch(r"[A-Za-z][A-Za-z' -]{0,78}", word):
        return {"ok": False, "saved": False, "error": "invalid_request"}
    try:
        time_value = max(0.0, float(data.get("time") or 0))
    except (TypeError, ValueError):
        time_value = 0.0
    meanings = data.get("other_meanings") if isinstance(data.get("other_meanings"), list) else []
    payload = {
        "word": word,
        "translation": str(data.get("translation") or "").strip()[:2000],
        "pos": str(data.get("pos") or "").strip()[:120],
        "pronunciation": str(data.get("pronunciation") or "").strip()[:160],
        "other_meanings": [str(item).strip()[:500] for item in meanings[:12] if str(item).strip()],
        "audio": str(data.get("audio") or "").strip()[:1000],
        "context": str(data.get("context") or "").strip()[:4000],
        "time": time_value,
        "subtitle_id": str(data.get("subtitle_id") or "")[:64],
        "project_id": project_id,
    }
    store = _collection_store("wordbook.json")
    with _COLLECTION_API_LOCK:
        existing = next((item for item in store.load()
                         if str(item.get("project_id")) == project_id
                         and str(item.get("word") or "").lower() == word.lower()), None)
        if existing:
            entry = store.update(existing["id"], **payload)
        else:
            entry = store.add(payload)
    return {"ok": True, "saved": True, "entry": entry}


def delete_word_api(data: dict) -> dict:
    item_id = str((data or {}).get("id") or "")[:64]
    project_id = str((data or {}).get("project_id") or "")[:64]
    word = str((data or {}).get("word") or "").strip().lower()[:100]
    if not item_id and not (project_id and word):
        return {"ok": False, "error": "invalid_request"}
    with _COLLECTION_API_LOCK:
        store = _collection_store("wordbook.json")
        if item_id:
            store.remove(item_id)
        else:
            store.remove_if(project_id=project_id, word=word)
    return {"ok": True, "id": item_id, "project_id": project_id, "word": word}


# ---------------------------------------------------------------------------
# Video file server (serves temp files to the custom HTML5 component)
# ---------------------------------------------------------------------------

class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """Serve only explicitly approved local video files to the player iframe."""

    def __init__(self, *args, allowed_files: set[str], **kwargs):
        self.allowed_files = allowed_files
        self._range_remaining: Optional[int] = None
        super().__init__(*args, **kwargs)

    def _is_allowed(self) -> bool:
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).lstrip("/")
        # Do not allow directory traversal, directory indexes, subtitle data, or settings.
        return bool(path) and "/" not in path and path in self.allowed_files

    def _reject_if_not_allowed(self) -> bool:
        if self._is_allowed():
            return False
        self.send_error(404)
        return True

    def _send_lookup_json(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_collection_bridge(self, data: dict, request_id: str) -> None:
        """Confirm a hidden-form write to the opaque player iframe."""
        message = json.dumps({
            "type": "englishLearn.collection",
            "requestId": str(request_id or "")[:96],
            "data": data,
        }, ensure_ascii=False).replace("</", "<\\/")
        payload = (
            "<!doctype html><meta charset=\"utf-8\">"
            f"<script>parent.postMessage({message}, '*');</script>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_collection_script(self, data: dict, request_id: str) -> None:
        """Return a one-shot JSONP-style callback for the player iframe."""
        safe_request_id = json.dumps(str(request_id or "")[:96], ensure_ascii=False)
        safe_data = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        payload = (
            "window.__englishLearnCollectionBridge("
            f"{safe_request_id},{safe_data});"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_collection_pixel(self, ok: bool) -> None:
        """Return a valid 1×1 GIF; load/error is the iframe write receipt."""
        payload = bytes.fromhex(
            "47494638396101000100800000000000ffffff21f90401000000002c"
            "00000000010001000002024401003b"
        )
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "image/gif")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_lookup_api(self) -> bool:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("target", ["zh-CN"])[0]
        if parsed.path == "/api/free-translate":
            self._send_lookup_json(free_translation_lookup(params.get("text", [""])[0], target))
            return True
        if parsed.path == "/api/word-lookup":
            self._send_lookup_json(free_word_lookup(params.get("word", [""])[0], target))
            return True
        if parsed.path == "/api/subtitle-offset":
            self._send_lookup_json(save_subtitle_offset(
                params.get("project_id", [""])[0], params.get("offset", [""])[0],
            ))
            return True
        if parsed.path == "/api/collection-bridge":
            request_id = params.get("request_id", [""])[0]
            action = params.get("action", [""])[0]
            try:
                data = json.loads(params.get("payload", ["{}"])[0])
            except (TypeError, ValueError, json.JSONDecodeError):
                data = {}
            result = self._dispatch_collection_api(
                action, data if isinstance(data, dict) else {},
            )
            self._send_collection_script(
                result or {"ok": False, "error": "invalid_action"},
                request_id,
            )
            return True
        if parsed.path == "/api/collection-pixel":
            action = params.get("action", [""])[0]
            try:
                data = json.loads(params.get("payload", ["{}"])[0])
            except (TypeError, ValueError, json.JSONDecodeError):
                data = {}
            result = self._dispatch_collection_api(
                action, data if isinstance(data, dict) else {},
            )
            self._send_collection_pixel(bool(result and result.get("ok")))
            return True
        return False

    def _dispatch_collection_api(self, path: str, data: dict) -> Optional[dict]:
        handlers = {
            "/api/favorite/toggle": toggle_favorite_api,
            "/api/favorite/delete": delete_favorite_api,
            "/api/word/save": save_word_api,
            "/api/word/delete": delete_word_api,
        }
        handler = handlers.get(path)
        if not handler:
            return None
        return handler(data)

    def do_GET(self):
        if self._handle_lookup_api():
            return
        if not self._reject_if_not_allowed():
            try:
                super().do_GET()
            except (BrokenPipeError, ConnectionResetError):
                # Browsers routinely cancel range/media requests while seeking,
                # reloading, or closing a tab. This is not an application error.
                pass

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(404)
            return
        try:
            length = min(65536, max(0, int(self.headers.get("Content-Length", "0"))))
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            content_type = self.headers.get("Content-Type", "").lower()
            request_id = ""
            if content_type.startswith("application/x-www-form-urlencoded"):
                fields = urllib.parse.parse_qs(raw, keep_blank_values=True)
                request_id = fields.get("request_id", [""])[0]
                data = json.loads(fields.get("payload", ["{}"])[0])
            else:
                data = json.loads(raw) if raw else {}
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_lookup_json({"ok": False, "error": "invalid_json"})
            return
        result = self._dispatch_collection_api(
            parsed.path, data if isinstance(data, dict) else {},
        )
        if result is None:
            self.send_error(404)
        elif request_id:
            self._send_collection_bridge(result, request_id)
        else:
            self._send_lookup_json(result)

    def do_HEAD(self):
        if not self._reject_if_not_allowed():
            super().do_HEAD()

    def send_head(self):
        """Serve a file with single-range support required by HTML5 seeking."""
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            self.send_error(404)
            return None
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None

        stat = os.fstat(source.fileno())
        size = stat.st_size
        content_type = self.guess_type(path)
        start, end = 0, max(0, size - 1)
        range_header = self.headers.get("Range", "")

        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or not size:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            first, last = match.groups()
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            elif last:
                suffix_length = int(last)
                if suffix_length <= 0:
                    source.close()
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None
                start = max(0, size - suffix_length)
                end = size - 1
            if start >= size or start > end:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self._range_remaining = length
            source.seek(start)
        else:
            length = size
            self.send_response(200)
            self._range_remaining = None

        self.send_header("Content-type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        remaining = self._range_remaining
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        # Streamlit serves ``srcdoc`` iframes with an opaque ``null`` origin.
        # This listener is loopback-only and its file allowlist remains intact,
        # so the player needs a wildcard CORS response for API access.
        if urllib.parse.urlparse(self.path).path.startswith("/api/"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")
            super().end_headers()
            return
        origin = self.headers.get("Origin", "")
        if re.match(r"^https?://(localhost|127\.0\.0\.1):\d+$", origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress noisy logs


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


_VIDEO_SERVERS: dict[str, tuple[http.server.HTTPServer, set[str], str]] = {}
_VIDEO_SERVERS_LOCK = threading.Lock()


def start_video_server(directory: str, filename: Optional[str] = None) -> str:
    """Start a background HTTP server to serve video files from *directory*.
    Only the requested basename is accessible. Existing servers reuse their
    port and add the newly selected video to their allowlist.
    """
    directory = os.path.abspath(directory)
    filename = os.path.basename(filename) if filename else ""
    with _VIDEO_SERVERS_LOCK:
        existing = _VIDEO_SERVERS.get(directory)
        if existing:
            if filename:
                existing[1].add(filename)
            return existing[2]

        allowed_files = {filename} if filename else set()
        port = _find_free_port()
        handler = functools.partial(
            _CORSHandler, directory=directory, allowed_files=allowed_files,
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{port}"
        _VIDEO_SERVERS[directory] = (server, allowed_files, url)
        return url


def start_lookup_server() -> str:
    """Start the localhost lookup bridge without exposing any local file."""
    directory = os.path.join(tempfile.gettempdir(), "englishlearn-lookup")
    os.makedirs(directory, exist_ok=True)
    return start_video_server(directory)


# ---------------------------------------------------------------------------
# Video download / parse
# ---------------------------------------------------------------------------

def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def _is_local_file(s: str) -> bool:
    p = Path(s)
    return p.exists() and p.is_file()


def _guess_filename(info: dict) -> str:
    """Best-effort to produce a normalised filename from yt-dlp info dict."""
    title = info.get("title", "video")
    ext = info.get("ext", "mp4")
    safe = re.sub(r"[^\w\s\-.]", "", title).strip()[:120]
    return f"{safe}.{ext}"


def _is_mp4_container(path: str) -> bool:
    """Detect ISO-BMFF/MP4 by signature instead of trusting the extension."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        return len(header) >= 12 and header[4:8] == b"ftyp"
    except OSError:
        return False


def normalize_video_path(path: str) -> str:
    """Rename an MP4 container with a misleading extension to ``.mp4``.

    yt-dlp may merge H.264 + AAC into MP4 while retaining an information-pass
    ``.webm`` filename. Serving that file as ``video/webm`` prevents browsers
    from loading metadata or playback.
    """
    path = os.path.abspath(path)
    if Path(path).suffix.lower() == ".mp4" or not _is_mp4_container(path):
        return path
    corrected = str(Path(path).with_suffix(".mp4"))
    if os.path.exists(corrected):
        return corrected
    os.replace(path, corrected)
    return corrected


def _download_thumbnail_image(url: str, output_dir: str, stem: str = "cover") -> Optional[str]:
    """Download a bounded platform thumbnail and return its local path."""
    if not url or not re.match(r"^https?://", str(url)):
        return None
    os.makedirs(output_dir, exist_ok=True)
    request = urllib.request.Request(str(url), headers={"User-Agent": "Mozilla/5.0"})
    try:
        with _DIRECT_HTTP_OPENER.open(request, timeout=12) as response:
            content_type = str(response.headers.get_content_type() or "").lower()
            suffix = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }.get(content_type)
            if suffix is None:
                return None
            data = response.read(8 * 1024 * 1024 + 1)
    except OSError:
        return None
    if not data or len(data) > 8 * 1024 * 1024:
        return None
    destination = os.path.join(output_dir, f"{stem}{suffix}")
    temp_path = f"{destination}.tmp"
    try:
        with open(temp_path, "wb") as handle:
            handle.write(data)
        os.replace(temp_path, destination)
        return destination
    except OSError:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return None


def generate_video_thumbnail(
    video_path: str,
    output_dir: str,
    stem: str = "cover",
    force: bool = False,
) -> Optional[str]:
    """Extract a representative 16:9 JPEG frame from a local video."""
    if not video_path or not os.path.isfile(video_path):
        return None
    _ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, f"{stem}.jpg")
    if not force and os.path.isfile(destination) and os.path.getsize(destination) > 0:
        return destination
    try:
        probe = ffmpeg.probe(video_path)
        duration = float((probe.get("format") or {}).get("duration") or 0.0)
    except (ffmpeg.Error, OSError, TypeError, ValueError):
        duration = 0.0
    seek_time = min(60.0, max(0.5, duration * 0.25)) if duration else 1.0
    try:
        (
            ffmpeg
            .input(video_path, ss=seek_time)
            .filter("scale", 640, 360, force_original_aspect_ratio="decrease")
            .filter("pad", 640, 360, "(ow-iw)/2", "(oh-ih)/2", color="black")
            .output(destination, vframes=1, format="image2", qscale=3)
            .run(overwrite_output=True, quiet=True)
        )
        return destination if os.path.getsize(destination) > 0 else None
    except (ffmpeg.Error, OSError):
        try:
            os.unlink(destination)
        except OSError:
            pass
        return None


def ensure_video_thumbnail(
    video_path: Optional[str],
    output_dir: str,
    remote_url: Optional[str] = None,
) -> Optional[str]:
    """Prefer a platform cover; fall back to extracting a local video frame."""
    downloaded = _download_thumbnail_image(remote_url or "", output_dir)
    return downloaded or generate_video_thumbnail(video_path or "", output_dir)


def download_video(
    url: str,
    output_dir: str,
    proxy: str = None,
    metadata_out: Optional[dict] = None,
    thumbnail_dir: Optional[str] = None,
) -> str:
    """Download a video from *url* into *output_dir* using yt-dlp.
    Returns the path to the downloaded file.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # First pass: extract info without downloading to get the real title
    info_opts = {"quiet": True, "no_warnings": True}
    if proxy:
        info_opts["proxy"] = proxy
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if metadata_out is not None:
        metadata_out.update({
            "title": info.get("title") or "",
            "id": info.get("id") or "",
            "extractor": info.get("extractor_key") or info.get("extractor") or "",
            "duration": info.get("duration"),
            "thumbnail_url": info.get("thumbnail") or "",
        })

    guessed_filename = _guess_filename(info)
    # The selected format is merged as MP4, so the output extension must also
    # be MP4. Check and migrate an older misleading filename before downloading.
    filename = f"{Path(guessed_filename).stem}.mp4"
    outtmpl = os.path.join(output_dir, filename)
    legacy_outtmpl = os.path.join(output_dir, guessed_filename)

    # If already downloaded, skip
    if os.path.exists(outtmpl):
        video_path = outtmpl
        if thumbnail_dir and metadata_out is not None:
            metadata_out["thumbnail_path"] = ensure_video_thumbnail(
                video_path, thumbnail_dir, metadata_out.get("thumbnail_url"),
            )
        return video_path
    if legacy_outtmpl != outtmpl and os.path.exists(legacy_outtmpl):
        normalized = normalize_video_path(legacy_outtmpl)
        if normalized != legacy_outtmpl:
            if thumbnail_dir and metadata_out is not None:
                metadata_out["thumbnail_path"] = ensure_video_thumbnail(
                    normalized, thumbnail_dir, metadata_out.get("thumbnail_url"),
                )
            return normalized

    ydl_opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "format_sort": ["res:720", "res:480"],
        # Android client avoids YouTube SABR streaming 403 errors
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    if proxy:
        ydl_opts["proxy"] = proxy
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    video_path = normalize_video_path(outtmpl)
    if thumbnail_dir and metadata_out is not None:
        metadata_out["thumbnail_path"] = ensure_video_thumbnail(
            video_path, thumbnail_dir, metadata_out.get("thumbnail_url"),
        )
    return video_path


def youtube_video_id(url: str) -> Optional[str]:
    """Return a validated YouTube video id for common public URL shapes."""
    from urllib.parse import parse_qs, urlparse

    try:
        parsed = urlparse(url.strip())
    except (TypeError, ValueError):
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
                candidate = parts[1]
    return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate or "") else None


def extract_subtitles(
    url: str,
    output_dir: str,
    proxy: str = None,
    metadata_out: Optional[dict] = None,
    preferred_languages: Optional[list[str]] = None,
    strict_preferred: bool = False,
) -> Optional[list[dict]]:
    """Try to download existing subtitles from *url* via yt-dlp.
    Returns a list of {"id", "start", "end", "text"} or None.
    Supports YouTube (yt), Bilibili automatic captions, and other sites
    that yt-dlp can extract subtitles from.
    """
    import yt_dlp

    requested_languages = list(dict.fromkeys(
        preferred_languages
        or ["en", "en-US", "en-GB", "zh-Hans", "zh-CN", "zh"]
    ))
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": requested_languages,
        "skip_download": True,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
    }
    if proxy:
        ydl_opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None

    if metadata_out is not None:
        metadata_out.update({
            "title": info.get("title") or "",
            "id": info.get("id") or "",
            "extractor": info.get("extractor_key") or info.get("extractor") or "",
            "duration": info.get("duration"),
            "thumbnail_url": info.get("thumbnail") or "",
        })

    # Check for manual subtitles first, then auto-generated
    subtitles_info = info.get("subtitles", {}) or {}
    auto_subtitles = info.get("automatic_captions", {}) or {}

    sub_list = None
    selected_language = ""
    selected_source = ""
    for lang in requested_languages:
        if lang in subtitles_info and subtitles_info[lang]:
            sub_list = subtitles_info[lang]
            selected_language = lang
            selected_source = "manual"
            break
    if sub_list is None:
        for lang in requested_languages:
            if lang in auto_subtitles and auto_subtitles[lang]:
                sub_list = auto_subtitles[lang]
                selected_language = lang
                selected_source = "automatic"
                break

    if sub_list is None and not strict_preferred:
        fallback_languages = [
            lang for lang in ("en", "en-US", "en-GB", "zh-Hans", "zh-CN", "zh")
            if lang not in requested_languages
        ]
        for collection, source_name in (
            (subtitles_info, "manual"),
            (auto_subtitles, "automatic"),
        ):
            for lang in fallback_languages:
                if lang in collection and collection[lang]:
                    sub_list = collection[lang]
                    selected_language = lang
                    selected_source = source_name
                    break
            if sub_list is not None:
                break

    if not sub_list:
        return None

    if metadata_out is not None:
        metadata_out["subtitle_language"] = selected_language
        metadata_out["subtitle_source"] = selected_source

    # Find the JSON3 or SRT format URL
    sub_url = None
    for fmt in sub_list:
        if fmt.get("ext") in ("json3", "srv3"):
            sub_url = fmt["url"]
            break
    if sub_url is None:
        for fmt in sub_list:
            if fmt.get("ext") == "srt":
                sub_url = fmt["url"]
                break
    if sub_url is None and sub_list:
        sub_url = sub_list[0]["url"]

    if not sub_url:
        return None

    # Download and parse the subtitle file
    try:
        resp = _fetch(sub_url)
    except Exception:
        return None

    if sub_url.endswith(".json3") or "json3" in sub_url:
        return _parse_json3(resp)
    else:
        return _parse_srt(resp)


def resolve_video_source(
    url_or_path: str,
    work_dir: str,
    proxy: str = None,
    metadata_out: Optional[dict] = None,
    thumbnail_dir: Optional[str] = None,
) -> str:
    """Given a URL or local path, return the absolute path to a playable video file.
    Downloads the video to *work_dir* if it's a URL.
    """
    if _is_local_file(url_or_path) or not _is_url(url_or_path):
        return os.path.abspath(url_or_path)

    return download_video(
        url_or_path,
        work_dir,
        proxy=proxy,
        metadata_out=metadata_out,
        thumbnail_dir=thumbnail_dir,
    )


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _fetch(url: str) -> str:
    """Download content from *url* and return as string."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _parse_json3(raw: str) -> list[dict]:
    """Parse YouTube JSON3, including per-word/phrase relative offsets."""
    data = json.loads(raw)
    prepared = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        text = re.sub(
            r"\s+", " ", "".join(str(s.get("utf8", "")) for s in segs),
        ).strip()
        if not text:
            continue
        try:
            start_ms = max(0.0, float(ev.get("tStartMs", 0)))
            duration_ms = max(0.0, float(ev.get("dDurationMs", 0) or 0))
        except (TypeError, ValueError):
            continue
        prepared.append({
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "text": text,
            "segs": segs,
        })

    results = []
    for index, item in enumerate(prepared):
        start_ms = item["start_ms"]
        next_start_ms = (
            prepared[index + 1]["start_ms"]
            if index + 1 < len(prepared) else None
        )
        duration_ms = item["duration_ms"]
        if duration_ms <= 0:
            if next_start_ms is not None and next_start_ms > start_ms:
                duration_ms = next_start_ms - start_ms
            else:
                # Final fallback only; normal JSON3 captions include either a
                # duration or a following event boundary.
                duration_ms = min(8000.0, max(700.0, len(item["text"]) * 65.0))
        end_ms = start_ms + max(120.0, duration_ms)
        # Caption events occasionally overlap because a platform emits rolling
        # updates. A learning cue should hand off at the next event boundary.
        if next_start_ms is not None and next_start_ms > start_ms:
            end_ms = min(end_ms, next_start_ms)

        timed_words = []
        segs = item["segs"]
        for seg_index, seg in enumerate(segs):
            seg_text = re.sub(r"\s+", " ", str(seg.get("utf8", ""))).strip()
            tokens = seg_text.split()
            if not tokens:
                continue
            try:
                offset_ms = max(0.0, float(seg.get("tOffsetMs", 0) or 0))
            except (TypeError, ValueError):
                offset_ms = 0.0
            seg_start_ms = min(end_ms, start_ms + offset_ms)

            next_offset_ms = None
            for following in segs[seg_index + 1:]:
                if not str(following.get("utf8", "")).strip():
                    continue
                try:
                    next_offset_ms = max(
                        offset_ms,
                        float(following.get("tOffsetMs", 0) or 0),
                    )
                except (TypeError, ValueError):
                    next_offset_ms = None
                break
            seg_end_ms = (
                min(end_ms, start_ms + next_offset_ms)
                if next_offset_ms is not None and next_offset_ms > offset_ms
                else end_ms
            )
            # tOffsetMs marks the spoken start. Do not stretch a word across a
            # long silence merely because the next word starts much later.
            estimated_spoken_ms = sum(
                min(700.0, max(160.0, len(token.strip(".,!?;:")) * 55.0 + 100.0))
                for token in tokens
            )
            seg_duration_ms = min(
                max(30.0, seg_end_ms - seg_start_ms),
                max(30.0, estimated_spoken_ms),
            )
            for token_index, token in enumerate(tokens):
                token_start_ms = seg_start_ms + seg_duration_ms * token_index / len(tokens)
                token_end_ms = seg_start_ms + seg_duration_ms * (token_index + 1) / len(tokens)
                timed_words.append({
                    "word": token,
                    "start": round(token_start_ms / 1000.0, 3),
                    "end": round(token_end_ms / 1000.0, 3),
                })

        text_tokens = item["text"].split()
        timings_overlap = any(
            timed_words[pos]["start"] < timed_words[pos - 1]["end"] - 0.005
            for pos in range(1, len(timed_words))
        )
        if len(timed_words) != len(text_tokens) or timings_overlap:
            event_duration_ms = max(30.0, end_ms - start_ms)
            timed_words = [
                {
                    "word": token,
                    "start": round(
                        (start_ms + event_duration_ms * pos / len(text_tokens)) / 1000.0,
                        3,
                    ),
                    "end": round(
                        (start_ms + event_duration_ms * (pos + 1) / len(text_tokens)) / 1000.0,
                        3,
                    ),
                }
                for pos, token in enumerate(text_tokens)
            ]
        else:
            for token, timing in zip(text_tokens, timed_words):
                timing["word"] = token

        results.append({
            "id": index,
            "start": round(
                timed_words[0]["start"] if timed_words else start_ms / 1000.0,
                3,
            ),
            "end": round(
                timed_words[-1]["end"] if timed_words else end_ms / 1000.0,
                3,
            ),
            "text": item["text"],
            "words": timed_words,
        })
    return results or None


def _parse_srt(raw: str) -> list[dict]:
    """Parse SRT subtitle format into list of {id, start, end, text}."""
    import html
    import re
    results = []
    blocks = re.split(r"\n\s*\n", raw.strip())
    idx = 0
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Parse timestamp line: "00:01:23,456 --> 00:01:25,789"
        ts_match = re.match(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
            lines[1],
        )
        if not ts_match:
            continue
        start = (int(ts_match[1]) * 3600 + int(ts_match[2]) * 60 +
                 int(ts_match[3]) + int(ts_match[4]) / 1000.0)
        end = (int(ts_match[5]) * 3600 + int(ts_match[6]) * 60 +
               int(ts_match[7]) + int(ts_match[8]) / 1000.0)
        # SRT commonly carries styling tags (<i>, <b>, <font>) that are useful
        # to a native subtitle renderer but are noise for ASR/translation and
        # would otherwise appear literally in the learning player.
        text = html.unescape(re.sub(r"<[^>]+>", "", "\n".join(lines[2:]))).strip()
        if not text:
            continue
        results.append({"id": idx, "start": start, "end": end, "text": text})
        idx += 1
    return _stabilize_cue_boundaries(results) or None


def _stabilize_cue_boundaries(cues: list[dict]) -> list[dict]:
    """Repair invalid/overlapping external cue boundaries for one active row."""
    stable = [dict(cue) for cue in cues]
    for index, cue in enumerate(stable):
        start = max(0.0, float(cue.get("start", 0.0)))
        end = max(start + 0.12, float(cue.get("end", start)))
        if index + 1 < len(stable):
            next_start = max(0.0, float(stable[index + 1].get("start", end)))
            if next_start > start:
                end = min(end, next_start)
        cue["start"] = round(start, 3)
        cue["end"] = round(max(start + 0.05, end), 3)
    return stable


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio from *video_path* as 16 kHz mono WAV.
    Returns the path to the extracted .wav file.
    """
    _ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(video_path).stem
    wav_path = os.path.join(output_dir, f"{stem}.wav")

    if os.path.exists(wav_path):
        return wav_path

    (
        ffmpeg
        .input(video_path)
        .output(wav_path, ac=1, ar=16000, format="wav")
        .run(overwrite_output=True, quiet=True)
    )
    return wav_path
