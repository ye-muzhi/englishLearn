"""
LLM Translator: Batch translation via Ollama or OpenAI API.

Supports multiple translation output formats, auto-detected per batch:

- JSON array: ``[{"id": 0, "translation": "..."}, ...]``
- ``---`` separated: ``翻译1\n---\n翻译2\n---\n翻译3``
- Numbered lines: ``1. 翻译1\n2. 翻译2`` or ``[0] 翻译1\n[1] 翻译2``
- Plain line-by-line: one translation per input line

Unparseable responses are logged for debugging.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable, Optional

from openai import OpenAI

_log = logging.getLogger("llm_translator")


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

# Models that should use line-by-line mode (dedicated MT models)
_LINE_MODE_PATTERNS = ("mt2", "mt-", "translator", "translate")


def _detect_mode(model: str) -> str:
    name = model.lower()
    return "line" if any(p in name for p in _LINE_MODE_PATTERNS) else "json"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _build_json_system_prompt(target_language: str) -> str:
    return (
        f"Translate these subtitles into {target_language}. "
        "Use natural, conversational language. Keep context across sentences. "
        "If already in target language, keep as-is. "
        "Return ONLY a JSON array: [{\"id\": <num>, \"translation\": \"<text>\"}, ...]"
    )


_LANG_NAMES: dict[str, str] = {
    "Chinese (Simplified)": "中文",
    "Chinese (Traditional)": "繁體中文",
    "Japanese": "日本語",
    "Korean": "한국어",
    "French": "法语",
    "German": "德语",
    "Spanish": "西班牙语",
    "Portuguese": "葡萄牙语",
    "Russian": "俄语",
    "Arabic": "阿拉伯语",
    "Thai": "泰语",
    "Vietnamese": "越南语",
    "English": "英文",
}


def _light_punctuate(text: str) -> str:
    """Add minimal punctuation to raw ASR text for MT models.

    MT models are trained on well-formatted text and often refuse or
    garble unpunctuated ASR output.  This heuristic adds the minimum:
    - Capitalise the first letter
    - Append a period if the text lacks sentence-ending punctuation
    - Capitalise after sentence-ending punctuation
    """
    text = text.strip()
    if not text:
        return text
    # Capitalise first letter (skip non-alpha leading chars)
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1:]
            break
    # Ensure sentence-ending punctuation
    if text[-1] not in ".?!。？！…":
        text += "."
    # Capitalise after sentence-ending punctuation
    text = re.sub(
        r"([.?!])\s+([a-z])",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )
    return text


def _build_line_system_prompt(target_language: str) -> str:
    """Build a system prompt for numbered line-by-line MT models."""
    native = _LANG_NAMES.get(target_language, target_language)
    return (
        f"逐条翻译以下编号句子为{native}。"
        f"严格保持编号格式输出：1. 翻译\\n2. 翻译\\n..."
        f"只输出翻译结果。"
    )


BATCH_SIZE = 30       # subtitles per request (JSON-mode LLMs)
LINE_BATCH_SIZE = 10  # smaller batches for dedicated MT models
MAX_WORKERS = 6       # concurrent API calls
MAX_PARSE_RETRIES = 1  # retry one malformed batch before isolating each line


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Optional[list[dict]]:
    """Attempt to parse response as a JSON array of {id, translation}."""
    try:
        data = json.loads(text)
        if isinstance(data, list) and data and "id" in data[0]:
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    # Extract embedded JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list) and data and "id" in data[0]:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    # Regex per-object fallback
    results = []
    for m in re.finditer(
        r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        text,
    ):
        results.append({"id": int(m.group(1)), "translation": m.group(2)})
    return results or None


def _try_parse_separator(text: str, expected_count: int) -> Optional[list[str]]:
    """Parse ``---`` separated translations."""
    if "---" not in text:
        return None
    parts = [p.strip() for p in re.split(r"\n?\s*---+\s*\n?", text) if p.strip()]
    if len(parts) == expected_count:
        return parts
    return None


def _try_parse_numbered(text: str, expected_count: int) -> Optional[list[str]]:
    """Parse numbered lines like ``1. 翻译`` or ``[0] 翻译``."""
    # Match patterns: "1. text", "1) text", "[0] text", "1: text"
    pattern = r'(?:^|\n)\s*(?:\[(\d+)\]|(\d+)[.):]\s*)(.+?)(?=(?:\n\s*(?:\[\d+\]|\d+[.):])\s*|$))'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    numbered = [
        (int(bracketed or plain), translation.strip())
        for bracketed, plain, translation in matches
    ]
    if len(numbered) != expected_count or any(not item[1] for item in numbered):
        return None

    # Models commonly use either 0..N-1 or 1..N.  Respect the labels instead
    # of the physical response order, otherwise one reordered line silently
    # assigns a translation to the wrong subtitle.
    labels = [item[0] for item in numbered]
    if set(labels) == set(range(expected_count)):
        base = 0
    elif set(labels) == set(range(1, expected_count + 1)):
        base = 1
    else:
        return None
    by_position = {label - base: value for label, value in numbered}
    if len(by_position) != expected_count:
        return None
    return [by_position[index] for index in range(expected_count)]


def _try_parse_lines(text: str, expected_count: int) -> Optional[list[str]]:
    """Parse plain line-by-line output."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) == expected_count:
        return lines
    return None


def _normalise_json_result(
    items: list[dict], batch_ids: list[int],
) -> Optional[list[dict]]:
    """Validate and align a JSON response without ever guessing a missing row."""
    if len(items) != len(batch_ids):
        return None

    parsed: list[tuple[int, str]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            return None
        translation = str(item.get("translation") or "").strip()
        if not translation:
            return None
        parsed.append((item_id, translation))

    response_ids = [item[0] for item in parsed]
    if len(set(response_ids)) != len(batch_ids):
        return None

    expected = set(batch_ids)
    if set(response_ids) == expected:
        by_id = dict(parsed)
        return [
            {"id": subtitle_id, "translation": by_id[subtitle_id]}
            for subtitle_id in batch_ids
        ]

    # Some APIs renumber every batch locally.  Rebase only when the complete,
    # unambiguous 0-based or 1-based sequence is present.
    n = len(batch_ids)
    if set(response_ids) == set(range(n)):
        base = 0
    elif set(response_ids) == set(range(1, n + 1)):
        base = 1
    else:
        return None
    by_position = {item_id - base: value for item_id, value in parsed}
    return [
        {"id": subtitle_id, "translation": by_position[index]}
        for index, subtitle_id in enumerate(batch_ids)
    ]


def _try_parse_single_block(text: str) -> Optional[list[str]]:
    """Last resort: treat the entire output as one translation.
    Only used when there's exactly one subtitle in the batch.
    """
    text = text.strip()
    if text:
        return [text]
    return None


def _parse_response(
    raw: str, batch_ids: list[int], model: str = "",
) -> list[dict]:
    """Universal response parser.  Tries multiple strategies in order:

    1. JSON array ``[{"id": ..., "translation": ...}]``
    2. ``---`` separated blocks
    3. Numbered lines (``1. text``, ``[0] text``)
    4. Plain line-by-line (one output line per input line)
    5. Single block (for single-subtitle batches)

    Logs raw output on total failure for debugging.
    """
    n = len(batch_ids)
    sid_map = {i: sid for i, sid in enumerate(batch_ids)}

    # Strategy 1: JSON
    json_result = _try_parse_json(raw)
    if json_result:
        aligned_json = _normalise_json_result(json_result, batch_ids)
        if aligned_json:
            return aligned_json

    # Strategy 2: separator-delimited
    sep_result = _try_parse_separator(raw, n)
    if sep_result:
        return [{"id": sid_map[i], "translation": t}
                for i, t in enumerate(sep_result) if i in sid_map]

    # Strategy 3: numbered lines
    num_result = _try_parse_numbered(raw, n)
    if num_result:
        return [{"id": sid_map[i], "translation": t}
                for i, t in enumerate(num_result) if i in sid_map]

    # Strategy 4: plain lines
    line_result = _try_parse_lines(raw, n)
    if line_result:
        return [{"id": sid_map[i], "translation": t}
                for i, t in enumerate(line_result) if i in sid_map]

    # Strategy 5: single block (only for single-item batches)
    if n == 1:
        single = _try_parse_single_block(raw)
        if single:
            return [{"id": batch_ids[0], "translation": single[0]}]

    # All strategies failed — log for debugging
    _log.warning(
        "[LLM] Failed to parse response from model '%s' "
        "(expected %d translations). Raw output:\n%s",
        model, n, raw[:2000],
    )
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_subtitles(
    subtitles: list[dict],
    engine: str = "ollama",
    model: str = "kaelri/hy-mt2:7b",
    api_base: str = "http://192.168.101.202:11434/v1",
    api_key: str = "ollama",
    target_lang: str = "Chinese (Simplified)",
    max_tokens: int = 32768,
    on_progress: Optional[Callable[[int, int], None]] = None,
    translate_mode: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> list[dict]:
    """Translate *subtitles* in batches with concurrent API calls.

    Parameters
    ----------
    subtitles : list of {"id", "start", "end", "text"}
    engine    : "ollama" or "openai"
    model     : model name
    api_base  : base URL for the API
    api_key   : API key (use "ollama" placeholder for local Ollama)
    target_lang : target language name
    max_tokens : max tokens per API response
    on_progress : optional callback(completed_batches, total_batches)
    translate_mode : "json" or "line" (auto-detected from model name if None)

    Returns
    -------
    Copy of *subtitles* with "translation" field added to each entry.
    """
    if engine == "hy_mt2_local":
        from .hy_mt2_local import translate_subtitles as translate_with_hy_mt2
        return translate_with_hy_mt2(
            subtitles=subtitles,
            target_lang=target_lang,
            model_dir=api_base or None,
            max_tokens=min(max_tokens, 4096),
            on_progress=on_progress,
        )

    import httpx

    _is_local = any(
        api_base.startswith(p) for p in
        ("http://localhost", "http://127.", "http://192.168.", "http://10.")
    )
    http_client = httpx.Client(proxy=None) if _is_local else None

    mode = translate_mode or _detect_mode(model)
    translated = [dict(s) for s in subtitles]
    translated_by_id = {s.get("id"): s for s in translated}

    if mode == "line":
        system_prompt = _build_line_system_prompt(target_lang)
    else:
        system_prompt = _build_json_system_prompt(target_lang)

    # Build batches (MT models get smaller batches)
    batch_size = LINE_BATCH_SIZE if mode == "line" else BATCH_SIZE
    batches = []
    for i in range(0, len(subtitles), batch_size):
        batch = subtitles[i:i + batch_size]
        if mode == "line":
            # Numbered format: "1. text\n2. text\n..."
            # MT models need punctuated text; add light preprocessing
            user_msg = "\n".join(
                f"{i+1}. {_light_punctuate(s['text'])}"
                for i, s in enumerate(batch)
            )
        else:
            user_msg = "\n".join(f"[{s['id']}] {s['text']}" for s in batch)
        batches.append((i // batch_size + 1, batch, user_msg))

    total = len(batches)
    completed = 0
    lock = Lock()

    # Use the model's preferred temperature
    temperature = 0.7 if mode == "line" else 0.1

    client = OpenAI(base_url=api_base, api_key=api_key, http_client=http_client)

    def _request_translation(user_msg: str, prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def _single_user_message(subtitle: dict) -> str:
        if mode == "line":
            return f"1. {_light_punctuate(subtitle['text'])}"
        return f"[{subtitle['id']}] {subtitle['text']}"

    def _translate_batch(batch_num: int, batch: list[dict], user_msg: str):
        """Translate a single batch, return (batch_num, list of {id, translation})."""
        try:
            batch_ids = [s["id"] for s in batch]
            for attempt in range(MAX_PARSE_RETRIES + 1):
                retry_note = (
                    "\nThe previous response was invalid. Return exactly one "
                    "non-empty result for every input item; do not merge, omit, "
                    "split, or reorder items."
                    if attempt else ""
                )
                raw = _request_translation(user_msg, system_prompt + retry_note)
                parsed = _parse_response(raw, batch_ids, model=model)
                if len(parsed) == len(batch):
                    return batch_num, parsed

            # A malformed batch must never be partially zipped back onto the
            # timeline. Isolate each cue so one model formatting mistake cannot
            # shift every translation that follows it.
            isolated: list[dict] = []
            for subtitle in batch:
                raw = _request_translation(
                    _single_user_message(subtitle), system_prompt,
                )
                parsed = _parse_response(raw, [subtitle["id"]], model=model)
                if parsed:
                    isolated.extend(parsed)
            return batch_num, isolated
        except Exception as exc:
            _log.error("[LLM] Batch %d request failed: %s", batch_num, exc)
            return batch_num, []

    with ThreadPoolExecutor(max_workers=max_workers or MAX_WORKERS) as executor:
        futures = {
            executor.submit(_translate_batch, bn, b, msg): bn
            for bn, b, msg in batches
        }
        for future in as_completed(futures):
            batch_num, parsed = future.result()
            for item in parsed:
                sid = item.get("id")
                trans = item.get("translation", "")
                if sid in translated_by_id:
                    translated_by_id[sid]["translation"] = trans
            # Fill missing translations in this batch
            for source in batches[batch_num - 1][1]:
                target = translated_by_id.get(source.get("id"))
                if target is not None and "translation" not in target:
                    target["translation"] = ""

            with lock:
                completed += 1
                if on_progress:
                    on_progress(completed, total)

    return translated


# ---------------------------------------------------------------------------
# Word / phrase translation (with linguistic details)
# ---------------------------------------------------------------------------

_LANG_NAMES_FULL: dict[str, str] = {
    "Chinese (Simplified)": "简体中文",
    "Chinese (Traditional)": "繁體中文",
    "Japanese": "日本語",
    "Korean": "한국어",
    "French": "法语",
    "German": "德语",
    "Spanish": "西班牙语",
    "Portuguese": "葡萄牙语",
    "Russian": "俄语",
    "Arabic": "阿拉伯语",
    "Thai": "泰语",
    "Vietnamese": "越南语",
    "English": "英文",
}


def translate_word(
    word: str,
    context: str,
    target_lang: str,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int = 1024,
    engine: str = "openai",
) -> dict:
    """Translate a word/phrase with context and return linguistic details.

    Returns dict with keys: translation, pos, pronunciation, other_meanings.
    Falls back to simple translation for MT models.
    """
    if engine == "hy_mt2_local":
        from .hy_mt2_local import translate_text
        return {
            "translation": translate_text(word, target_lang, api_base or None, max_tokens),
            "pos": "",
            "pronunciation": "",
            "other_meanings": [],
        }

    import httpx

    _is_local = any(
        api_base.startswith(p)
        for p in ("http://localhost", "http://127.", "http://192.168.", "http://10.")
    )
    http_client = httpx.Client(proxy=None) if _is_local else None
    client = OpenAI(base_url=api_base, api_key=api_key, http_client=http_client)

    native = _LANG_NAMES_FULL.get(target_lang, target_lang)
    mode = _detect_mode(model)

    if mode == "line":
        # MT model — simple line translation, no linguistic details
        prompt = f"{word}"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"逐条翻译为{native}。只输出翻译结果。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()
            return {
                "translation": raw.split("\n")[0].strip(),
                "pos": "",
                "pronunciation": "",
                "other_meanings": [],
            }
        except Exception as exc:
            _log.error("[WordTranslate] MT model failed: %s", exc)
            return {"translation": "", "pos": "", "pronunciation": "", "other_meanings": []}

    # Instruction-following model — full linguistic details
    system_prompt = (
        "You are a dictionary assistant. Given a word/phrase and its context, "
        "return linguistic details as JSON. Output ONLY the JSON object."
    )
    user_prompt = (
        f'Word/phrase: "{word}"\n'
        f'Context sentence: "{context}"\n'
        f"Target language: {native}\n\n"
        "Return a JSON object with:\n"
        '- "translation": most likely translation given the context\n'
        '- "pos": part of speech (noun/verb/adj/adv/phrase/etc.)\n'
        '- "pronunciation": phonetic or romanization if applicable, empty string otherwise\n'
        '- "other_meanings": array of alternative meanings (without context)\n\n'
        "Output ONLY the JSON, no markdown, no explanation."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip code fences if present
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        # Try parsing as JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Extract embedded JSON
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
            else:
                raise

        return {
            "translation": str(data.get("translation", "")).strip(),
            "pos": str(data.get("pos", "")).strip(),
            "pronunciation": str(data.get("pronunciation", "")).strip(),
            "other_meanings": list(data.get("other_meanings", [])),
        }

    except Exception as exc:
        _log.error("[WordTranslate] failed: %s", exc)
        return {"translation": "", "pos": "", "pronunciation": "", "other_meanings": []}
