"""
Sentence Segmenter — Re-segment subtitle fragments into natural sentences.

Whisper and extracted subtitles often produce fragments that split mid-sentence
or are too short to be meaningful for language learning.  This module applies
a multi-signal heuristic to merge adjacent fragments into complete sentences
while splitting any that grow too long.

Signals (in priority order):
  1. Sentence-ending punctuation (. ? ! ？ 。)
  2. Discourse markers starting a new segment ("Right", "Okay", "So")
  3. Phrase continuation cues — trailing function words, number+unit pairs
  4. Pause gap between segments (when segments are not contiguous)
  5. Segment word count — prefer 10–35 words per display segment
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MIN_PAUSE = 0.8   # seconds — sentence break when gap ≥ this
DEFAULT_MAX_WORDS = 24    # target ceiling per segment
# Learning subtitles should normally fit in one or two short visual lines.
# Complete sentences remain the first choice; long sentences fall back to a
# nearby clause boundary instead of an arbitrary word count.
SUBTITLE_MIN_PAUSE = 0.55
SUBTITLE_MAX_WORDS = 18
NATURAL_OVERFLOW_WORDS = 4  # small allowance to finish a real sentence
ABSOLUTE_MAX_WORDS = 60   # hard ceiling, always break before this

# Sentence-ending punctuation (multi-language)
_STRONG_ENDERS = frozenset(". ? ! ？ 。 ！ ． …".split())

# Words that, as the LAST word of a segment, signal the sentence continues
# into the next segment.  Grouped by grammatical role for clarity.
_CONTINUATION_WORDS: frozenset[str] = frozenset("""
    the a an my your his her its our their
    this that these those
    of in on at to for from with by about into over under between
    is are was were be been being am
    has have had
    will would shall should can could may might must
    and but or
    as if
    no nor not only
    very too also just really quite rather even still already yet
    so such
""".split())

# Words/phrases at the START of the next segment that strongly signal a
# new sentence or discourse unit.  Context-sensitive — see
# _is_strong_sentence_starter().
_DISCOURSE_MARKERS: frozenset[str] = frozenset("""
    right okay ok well anyway anyways
    actually basically essentially
    yeah yes yep yea nope nah
    look listen hey
    oh
    so then therefore however meanwhile
    first second third finally lastly
    also additionally furthermore moreover
    now
""".split())

# "Right now", "right here", "right there" — NOT discourse markers.
_RIGHT_NOT_MARKER: frozenset[str] = frozenset(
    "now here there there's away ahead back".split()
)

# "Well known", "well being" etc. — NOT discourse markers.
_WELL_NOT_MARKER: frozenset[str] = frozenset(
    "known being being's established defined documented".split()
)

# Do not treat a period in a common abbreviation as a sentence boundary.
_ABBREVIATIONS: frozenset[str] = frozenset(
    "mr mrs ms dr prof sr jr st vs etc e.g i.e a.m p.m no fig inc ltd u.s"
    .split()
)

# Number-unit continuations: "a hundred" + "times", "one" + "trillion", etc.
# When prev ends with a number word and next starts with one of these,
# they belong together.
_NUMBER_UNITS: frozenset[str] = frozenset("""
    times percent percentage dollars euros pounds yuan yen
    thousand million billion trillion
    meters metres kilometers kilometres miles feet inches
    grams kilograms pounds ounces liters litres gallons
    hours minutes seconds days weeks months years
    people person men women children kids
""".split())

# Number words that can be followed by a unit
_NUMBER_WORDS: frozenset[str] = frozenset("""
    zero one two three four five six seven eight nine ten
    eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
    twenty thirty forty fifty sixty seventy eighty ninety
    hundred thousand million billion trillion
    a an
""".split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sentence_segment(
    subtitles: list[dict],
    min_pause_sec: float = DEFAULT_MIN_PAUSE,
    max_words: int = DEFAULT_MAX_WORDS,
    keep_word_timing: bool = False,
) -> list[dict]:
    """Re-segment subtitle fragments into natural sentences.

    Parameters
    ----------
    subtitles
        ``[{"id", "start", "end", "text"}, ...]``
    min_pause_sec
        Pause gap (seconds) that triggers a sentence boundary.
    max_words
        Target maximum words per output segment.
    keep_word_timing
        Preserve internal word timing for pipeline persistence/retranslation.
        Player-facing callers should keep the default ``False``.

    Returns
    -------
    New list with the same dict shape, ``id`` values renumbered from 0.
    """
    if not subtitles:
        return []

    subs = [s for s in subtitles if s.get("text", "").strip()]
    if not subs:
        return []

    # Stage 0 — source captions often contain several sentences in a single
    # cue. Split those first so later merging cannot hide a real full stop.
    punctuated = _split_source_cues(subs)

    # Stage 1 — stitch very short, tightly-timed fragments
    merged = _merge_fragments(punctuated, max_words)

    # Stage 2 — group merged blocks into sentences
    sentences = _build_sentences(merged, min_pause_sec, max_words)

    # Stage 3 — split any remaining over-long segments
    split = _split_long(sentences, max_words)

    # Keep explicit sentence punctuation as a hard display boundary. Merging
    # several short completed utterances makes speaker changes ambiguous and
    # gives a translator multiple independent sentences under one timestamp.
    readable = split

    for i, s in enumerate(readable):
        s["id"] = i
        if not keep_word_timing:
            # Word timing is working metadata. Keeping it in player subtitle
            # JSON would substantially increase every Streamlit rerender.
            s.pop("words", None)
        s.pop("wc", None)

    return readable


# ---------------------------------------------------------------------------
# Stage 0: Split sentence boundaries inside one source cue
# ---------------------------------------------------------------------------

def _split_source_cues(segments: list[dict]) -> list[dict]:
    """Split complete sentences already present inside a source cue.

    YouTube/ASR captions frequently contain ``"First sentence. Next..."`` in
    one timestamped cue.  Timing is apportioned by token count because these
    formats do not expose an exact timestamp for the internal punctuation.
    """
    result: list[dict] = []
    for seg in segments:
        text = _normalise_text(seg.get("text", ""))
        words = text.split()
        if not words:
            continue
        timed_words = _aligned_timed_words(seg, words)

        break_after: list[int] = []
        for index, word in enumerate(words[:-1], start=1):
            if _word_has_sentence_ender(word) and not _word_is_abbreviation(word):
                break_after.append(index)

        if not break_after:
            start, end = _timed_bounds(
                timed_words, float(seg["start"]), float(seg["end"]),
            )
            result.append({
                "text": text,
                "start": start,
                "end": end,
                "wc": _word_count(text),
                "words": timed_words,
            })
            continue

        boundaries = [0, *break_after, len(words)]
        duration = max(0.0, float(seg["end"]) - float(seg["start"]))
        for left, right in zip(boundaries, boundaries[1:]):
            part = " ".join(words[left:right]).strip()
            if not part:
                continue
            part_timed_words = timed_words[left:right] if timed_words else []
            if part_timed_words:
                part_start = part_timed_words[0]["start"]
                part_end = part_timed_words[-1]["end"]
            else:
                part_start = float(seg["start"]) + duration * (left / len(words))
                part_end = float(seg["start"]) + duration * (right / len(words))
            result.append({
                "text": part,
                "start": round(part_start, 3),
                "end": round(part_end, 3),
                "wc": _word_count(part),
                "words": part_timed_words,
            })
    return result


# ---------------------------------------------------------------------------
# Stage 1: Merge very short adjacent fragments
# ---------------------------------------------------------------------------

def _merge_fragments(segments: list[dict], max_words: int) -> list[dict]:
    """Stitch incomplete fragments under 6 words separated by < 0.4 s."""
    MIN_FRAGMENT_WORDS = 6
    TIGHT_PAUSE = 0.4

    blocks: list[dict] = []
    buf: Optional[dict] = None

    for seg in segments:
        text = seg["text"].strip()
        wc = _word_count(text)

        if buf is None:
            buf = {"text": text, "start": seg["start"], "end": seg["end"],
                   "wc": wc, "words": list(seg.get("words") or [])}
            continue

        gap = seg["start"] - buf["end"]
        combined_wc = buf["wc"] + wc
        buf_has_sentence_end = (
            _is_strong_ender(_trailing_punctuation(buf["text"]))
            and not _word_is_abbreviation(buf["text"].split()[-1])
        )

        if (gap < TIGHT_PAUSE
                and not buf_has_sentence_end
                and (wc < MIN_FRAGMENT_WORDS or buf["wc"] < MIN_FRAGMENT_WORDS)
                and combined_wc <= max_words * 2):
            buf["text"] = _join(buf["text"], text)
            buf["end"] = seg["end"]
            buf["wc"] = _word_count(buf["text"])
            buf["words"].extend(seg.get("words") or [])
        else:
            blocks.append(buf)
            buf = {"text": text, "start": seg["start"], "end": seg["end"],
                   "wc": wc, "words": list(seg.get("words") or [])}

    if buf is not None:
        blocks.append(buf)
    return blocks


# ---------------------------------------------------------------------------
# Stage 2: Group blocks into sentences (core logic)
# ---------------------------------------------------------------------------

def _build_sentences(
    blocks: list[dict],
    min_pause: float,
    max_words: int,
) -> list[dict]:
    """Greedily accumulate blocks into sentence-length segments."""
    if not blocks:
        return []

    result: list[dict] = []
    acc_text = blocks[0]["text"]
    acc_start = blocks[0]["start"]
    acc_end = blocks[0]["end"]
    acc_wc = blocks[0]["wc"]
    acc_words = list(blocks[0].get("words") or [])

    for i in range(1, len(blocks)):
        blk = blocks[i]
        gap = blk["start"] - acc_end
        combined_wc = acc_wc + blk["wc"]

        if _is_boundary(acc_text, blk["text"], gap, combined_wc,
                        min_pause, max_words):
            result.append(_make_seg(acc_text, acc_start, acc_end, acc_words))
            acc_text = blk["text"]
            acc_start = blk["start"]
            acc_end = blk["end"]
            acc_wc = blk["wc"]
            acc_words = list(blk.get("words") or [])
        else:
            acc_text = _join(acc_text, blk["text"])
            acc_end = blk["end"]
            acc_wc = _word_count(acc_text)
            acc_words.extend(blk.get("words") or [])

    result.append(_make_seg(acc_text, acc_start, acc_end, acc_words))
    return result


def _is_boundary(
    prev_text: str,
    next_text: str,
    gap: float,
    combined_wc: int,
    min_pause: float,
    max_words: int,
) -> bool:
    """Decide whether to insert a sentence break between two blocks.

    Returns ``True`` = break here, ``False`` = merge with previous.
    """

    # Hard ceiling — always break before exceeding absolute maximum
    if combined_wc > ABSOLUTE_MAX_WORDS:
        return True

    prev_words = prev_text.split()
    next_words = next_text.split()
    prev_last = (prev_words[-1] if prev_words else "").rstrip(".,;:!?\"'）)】」』")
    next_first = (next_words[0] if next_words else "").rstrip(".,;:!?\"'（(【「『")
    prev_end_ch = _trailing_punctuation(prev_text)
    next_start_ch = next_text.lstrip()[:1] if next_text.lstrip() else ""

    # ------------------------------------------------------------------
    # Signal 1: Punctuation
    # ------------------------------------------------------------------
    if _is_strong_ender(prev_end_ch):
        # Whisper frequently emits cues such as "Dr." + "Smith".  Splitting
        # there creates an unusable subtitle and loses translation context.
        if prev_last.lower().rstrip(".") in _ABBREVIATIONS:
            return False
        if next_start_ch and next_start_ch.isupper():
            return True
        if gap > 0.3:
            return True
        return True  # default to break after strong punctuation

    # ------------------------------------------------------------------
    # Signal 2: Phrase continuation — DO NOT break
    # ------------------------------------------------------------------
    # Trailing function word: "the", "in", "of", etc.
    if prev_last.lower() in _CONTINUATION_WORDS:
        return False

    # Number + unit: "a hundred" → "times larger"
    if prev_last.lower() in _NUMBER_WORDS and next_first.lower() in _NUMBER_UNITS:
        return False

    # Trailing adjective/adverb expecting complement:
    # "larger than", "different from", "similar to"
    if len(prev_words) >= 2:
        bigram = (prev_words[-2] + " " + prev_words[-1]).lower()
        if bigram.endswith(("than", "from", "to", "with", "about")):
            return False

    # ------------------------------------------------------------------
    # Signal 3: Discourse marker at start of next segment — BREAK
    # ------------------------------------------------------------------
    if _is_strong_sentence_starter(next_first.lower(),
                                   next_words[1].lower() if len(next_words) > 1 else ""):
        return True

    # ------------------------------------------------------------------
    # Signal 4: Pause gap
    # ------------------------------------------------------------------
    if gap >= min_pause:
        return True

    # Discourse marker + tiny pause (≥ 0.2s) — probably a new sentence
    if next_first.lower() in _DISCOURSE_MARKERS and gap >= 0.2:
        return True

    # ------------------------------------------------------------------
    # Signal 5: Length-based
    # ------------------------------------------------------------------
    if combined_wc > max_words:
        next_has_sentence_end = (
            _is_strong_ender(_trailing_punctuation(next_text))
            and not _word_is_abbreviation(next_text.split()[-1])
        )
        if next_has_sentence_end and combined_wc <= max_words + NATURAL_OVERFLOW_WORDS:
            return False
        return True

    # ------------------------------------------------------------------
    # No strong signal — merge (continue the sentence)
    # ------------------------------------------------------------------
    return False


def _is_strong_sentence_starter(word: str, next_word: str = "") -> bool:
    """Check if *word* is a discourse marker signalling a new sentence.

    *next_word* is used for context-sensitive markers like "right" / "well".
    """
    if word not in _DISCOURSE_MARKERS:
        return False
    # "right now/here/there" is not a discourse marker
    if word == "right" and next_word in _RIGHT_NOT_MARKER:
        return False
    # "well known/established" is not a discourse marker
    if word == "well" and next_word in _WELL_NOT_MARKER:
        return False
    # "so" is weak — only count it as a marker if it's followed by
    # a clause opener (subject pronoun, discourse marker, etc.)
    if word == "so":
        # "so this", "so we", "so it" → discourse marker
        # "so big", "so much" → intensifier, not a marker
        intensifiers = {"big", "much", "many", "few", "little", "long",
                        "short", "fast", "slow", "hard", "easy", "far",
                        "good", "bad", "great", "small", "large"}
        if next_word in intensifiers:
            return False
    # "now" is weak — "now that", "now we" → marker; "now is" → might not be
    if word == "now":
        if next_word in ("is", "was", "the", "a", "an"):
            return False
    return True


# ---------------------------------------------------------------------------
# Stage 3: Split over-long segments
# ---------------------------------------------------------------------------

def _split_long(segments: list[dict], max_words: int) -> list[dict]:
    result: list[dict] = []
    for seg in segments:
        word_count = _word_count(seg["text"])
        is_complete_sentence = _is_strong_ender(_trailing_punctuation(seg["text"]))
        if (
            word_count <= max_words
            or (
                is_complete_sentence
                and word_count <= max_words + NATURAL_OVERFLOW_WORDS
            )
        ):
            result.append(seg)
        else:
            result.extend(_do_split(seg, max_words))
    return result


def _do_split(seg: dict, max_words: int) -> list[dict]:
    text = seg["text"]
    words = text.split()
    if len(words) <= max_words:
        return [seg]

    target = len(words) // 2
    best = _best_break(words, target)

    left_words = words[:best]
    right_words = words[best:]
    left_text = " ".join(left_words)
    right_text = " ".join(right_words)

    timed_words = _aligned_timed_words(seg, words)
    if timed_words:
        left_timed_words = timed_words[:best]
        right_timed_words = timed_words[best:]
        left_start, left_end = _timed_bounds(
            left_timed_words, seg["start"], seg["end"],
        )
        right_start, right_end = _timed_bounds(
            right_timed_words, seg["start"], seg["end"],
        )
        left_seg = _make_seg(left_text, left_start, left_end, left_timed_words)
        right_seg = _make_seg(right_text, right_start, right_end, right_timed_words)
    else:
        ratio = len(left_words) / len(words)
        duration = seg["end"] - seg["start"]
        split_t = seg["start"] + duration * ratio
        left_seg = _make_seg(left_text, seg["start"], split_t)
        right_seg = _make_seg(right_text, split_t, seg["end"])

    parts: list[dict] = []
    parts.extend(
        _do_split(left_seg, max_words)
        if _word_count(left_text) > max_words else [left_seg]
    )
    parts.extend(
        _do_split(right_seg, max_words)
        if _word_count(right_text) > max_words else [right_seg]
    )
    return parts


def _best_break(words: list[str], target: int) -> int:
    """Find the best split position near *target*."""
    n = len(words)
    lo = max(3, target - n // 4)
    hi = min(n - 3, target + n // 4)

    best_pos = target
    best_score = -999.0

    for pos in range(lo, hi + 1):
        raw_prev = words[pos - 1]
        prev_w = raw_prev.rstrip(".,;:!?\"'）)】」』")
        punctuation = _trailing_punctuation(raw_prev)
        score = 0.0
        if punctuation in ".?!。！？…":
            score = 10.0
        elif punctuation in ",;:，；：、":
            score = 5.0
        elif prev_w.lower() in ("and", "but", "or", "so", "yet"):
            score = 3.0
        if prev_w.lower() in _CONTINUATION_WORDS:
            score -= 4.0
        score -= abs(pos - target) * 0.1
        if score > best_score:
            best_score = score
            best_pos = pos

    return best_pos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg(
    text: str,
    start: float,
    end: float,
    words: Optional[list[dict]] = None,
) -> dict:
    segment = {
        "id": 0,
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text.strip(),
    }
    if words:
        segment["words"] = [dict(item) for item in words]
    return segment


def _join(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    if _cjk(a[-1]) or _cjk(b[0]):
        return a + b
    return a + " " + b


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\xa0", " ")).strip()


def _aligned_timed_words(seg: dict, text_words: list[str]) -> list[dict]:
    """Return timing metadata only when it maps one-to-one to text tokens."""
    raw_words = seg.get("words")
    if not isinstance(raw_words, list) or len(raw_words) != len(text_words):
        return []
    aligned: list[dict] = []
    previous_start = -1.0
    for token, item in zip(text_words, raw_words):
        if not isinstance(item, dict):
            return []
        try:
            start = max(0.0, float(item.get("start")))
            end = max(start, float(item.get("end")))
        except (TypeError, ValueError):
            return []
        if start < previous_start:
            return []
        aligned.append({
            "word": token,
            "start": round(start, 3),
            "end": round(end, 3),
        })
        previous_start = start
    return aligned


def _timed_bounds(
    words: list[dict],
    fallback_start: float,
    fallback_end: float,
) -> tuple[float, float]:
    if words:
        return round(float(words[0]["start"]), 3), round(float(words[-1]["end"]), 3)
    return round(float(fallback_start), 3), round(float(fallback_end), 3)


def _word_has_sentence_ender(word: str) -> bool:
    return _is_strong_ender(_trailing_punctuation(word))


def _word_is_abbreviation(word: str) -> bool:
    value = word.strip("\"'“”‘’()[]{}").lower().rstrip(".")
    if value in _ABBREVIATIONS:
        return True
    # Initialisms such as U.S. / U.K. are abbreviations, not cue boundaries.
    return bool(re.fullmatch(r"(?:[a-z]\.){2,}", word.strip("\"'“”‘’()[]{}").lower()))


def _word_count(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if _cjk(c))
    if cjk > len(text) * 0.25:
        return len(text.replace(" ", ""))
    return len(text.split())


def _cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3000 <= cp <= 0x303F
        or 0x3040 <= cp <= 0x30FF
    )


def _is_strong_ender(ch: str) -> bool:
    return ch in _STRONG_ENDERS


def _trailing_punctuation(text: str) -> str:
    """Return the semantic final punctuation, ignoring closing quotes/brackets."""
    value = text.rstrip()
    while value and value[-1] in "\"'”’）)]】」』»":
        value = value[:-1].rstrip()
    return value[-1:] if value else ""
