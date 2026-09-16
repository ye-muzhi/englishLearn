"""AI-based sentence segmentation using LLM.

Sends subtitle fragments to an instruction-following LLM and asks it to merge
them into complete natural sentences. Validates that each sentence ends with
proper punctuation and retries with feedback if not.

Falls back to rule-based segmentation if:
  - The model is an MT model (can't follow instructions)
  - The LLM call fails
  - The response can't be parsed
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from openai import OpenAI

from ..translation.llm_translator import _detect_mode
from .sentence_segmenter import sentence_segment

_log = logging.getLogger("ai_segmenter")

SENTENCE_END_CHARS = frozenset(".?!。？！…)\"'》」』")
SEGMENT_BATCH_SIZE = 80
MAX_WORKERS = 5
MAX_RETRIES = 2


def can_use_ai_segment(model: str) -> bool:
    """Check if the model can follow segmentation instructions."""
    return _detect_mode(model) != "line"


def _build_prompt(fragments: list[dict]) -> str:
    lines = []
    for i, f in enumerate(fragments):
        lines.append(f"{i}. [{f['start']:.1f}-{f['end']:.1f}] {f['text']}")

    return (
        "Merge these subtitle fragments into COMPLETE natural sentences.\n\n"
        "Rules:\n"
        "- Each output sentence MUST end with punctuation (. ! ? or equivalent)\n"
        "- Merge fragments belonging to the same sentence\n"
        "- Split at natural sentence boundaries\n"
        "- Keep sentences SHORT: one idea per sentence, max ~20 words\n"
        "- Preserve original wording exactly — do NOT paraphrase or translate\n"
        "- Each output gets: start = first fragment's start, end = last fragment's end\n\n"
        f"Fragments:\n" + "\n".join(lines) + "\n\n"
        "Output format: one sentence per line as  START|END|text\n"
        "Output ONLY these lines, nothing else."
    )


def _retry_prompt(base_prompt: str, bad_segments: list[dict]) -> str:
    issues = "\n".join(
        f"- \"{s['text'][:60]}\" — missing ending punctuation"
        for s in bad_segments[:5]
    )
    return (
        base_prompt
        + "\n\n---\nYour previous output had problems:\n" + issues
        + "\n\nRe-segment. EVERY sentence must end with . ! ? or equivalent. "
        "Fix the issues above."
    )


def _parse_response(raw: str) -> list[dict]:
    results = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        # Match: START|END|text  (allow extra pipes in text)
        m = re.match(r"^(\d+\.?\d*)\s*\|\s*(\d+\.?\d*)\s*\|\s*(.+)$", line)
        if not m:
            continue
        try:
            start = float(m.group(1))
            end = float(m.group(2))
            text = m.group(3).strip()
            if text:
                results.append({"start": start, "end": end, "text": text})
        except ValueError:
            continue
    return results


def _validate(segments: list[dict]) -> list[dict]:
    """Return segments that don't end with sentence-ending punctuation."""
    bad = []
    for seg in segments:
        t = seg["text"].rstrip("\"'")
        if t and t[-1] not in SENTENCE_END_CHARS:
            bad.append(seg)
    return bad


def _content_tokens(text: str) -> list[str]:
    """Comparable lexical content, excluding spacing and speaker arrows."""
    cleaned = str(text or "").replace(">>", " ").casefold()
    return re.findall(r"[\w]+(?:['’][\w]+)*", cleaned, flags=re.UNICODE)


def _preserves_source(batch: list[dict], segments: list[dict]) -> bool:
    """Reject dropped, duplicated, reordered, or time-shifted AI output."""
    if not batch or not segments:
        return False
    source_tokens = _content_tokens(" ".join(str(item.get("text") or "") for item in batch))
    output_tokens = _content_tokens(" ".join(str(item.get("text") or "") for item in segments))
    if source_tokens != output_tokens:
        return False

    try:
        source_start = float(batch[0]["start"])
        source_end = float(batch[-1]["end"])
        previous_start = source_start
        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            if start < source_start - 0.05 or end > source_end + 0.05:
                return False
            if end < start or start < previous_start - 0.05:
                return False
            previous_start = start
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _process_batch(
    batch_idx: int,
    batch: list[dict],
    model: str,
    api_base: str,
    api_key: str,
    http_client=None,
    client=None,
) -> list[dict]:
    if client is None:
        client = OpenAI(base_url=api_base, api_key=api_key, http_client=http_client)
    prompt = _build_prompt(batch)

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a subtitle segmentation expert. "
                        "Output only the requested format, nothing else.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip code fences if present
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

            segments = _parse_response(raw)
            if not segments:
                _log.warning(
                    "Segmenter batch %d attempt %d: no segments parsed", batch_idx, attempt + 1
                )
                continue

            bad = _validate(segments)
            source_preserved = _preserves_source(batch, segments)
            if not bad and source_preserved:
                return segments

            _log.info(
                "Segmenter batch %d attempt %d rejected: %d/%d missing punctuation, source_preserved=%s",
                batch_idx, attempt + 1, len(bad), len(segments), source_preserved,
            )
            prompt = _retry_prompt(_build_prompt(batch), bad)

        except Exception as exc:
            _log.error("Segmenter batch %d attempt %d failed: %s", batch_idx, attempt + 1, exc)

    # All retries failed — fall back to rule-based for this batch
    _log.warning("Segmenter batch %d: falling back to rule-based", batch_idx)
    return _fallback_batch(batch)


def _fallback_batch(batch: list[dict]) -> list[dict]:
    """Use rule-based segmenter as fallback for a single batch."""
    segs = sentence_segment(batch, max_words=20)
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]


def ai_sentence_segment(
    subtitles: list[dict],
    model: str,
    api_base: str,
    api_key: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    max_workers: Optional[int] = None,
) -> list[dict]:
    """Segment subtitles into natural sentences using LLM.

    Automatically falls back to rule-based segmentation per-batch on failure.
    Returns segments with id, start, end, text.
    """
    import httpx

    if not subtitles:
        return subtitles

    if not can_use_ai_segment(model):
        _log.info("Model '%s' is MT — using rule-based segmentation", model)
        return sentence_segment(subtitles, max_words=20)

    _is_local = any(
        api_base.startswith(p)
        for p in ("http://localhost", "http://127.", "http://192.168.", "http://10.")
    )
    http_client = httpx.Client(proxy=None) if _is_local else None
    shared_client = OpenAI(base_url=api_base, api_key=api_key, http_client=http_client)

    batches = []
    for i in range(0, len(subtitles), SEGMENT_BATCH_SIZE):
        batches.append(subtitles[i : i + SEGMENT_BATCH_SIZE])

    total = len(batches)
    all_segments = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers or MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _process_batch, i, b, model, api_base, api_key, http_client, shared_client
            ): i
            for i, b in enumerate(batches)
        }
        for future in as_completed(futures):
            segments = future.result()
            all_segments.extend(segments)
            completed += 1
            if on_progress:
                on_progress(completed, total)

    all_segments.sort(key=lambda s: s["start"])
    # One final deterministic pass splits any multi-sentence AI row and joins
    # an unfinished phrase that happened to cross an AI batch boundary.
    return sentence_segment(all_segments, min_pause_sec=0.55, max_words=18)
