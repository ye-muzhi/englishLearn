"""
ASR Engine: Speech-to-text with timestamps via faster-whisper.
"""
import os
from typing import Optional

from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Model size → display name (for UI)
# ---------------------------------------------------------------------------

MODEL_SIZES = {
    "tiny":   "tiny   (≈1 GB VRAM, fastest)",
    "base":   "base   (≈1 GB VRAM, fast)",
    "small":  "small  (≈2 GB VRAM)",
    "medium": "medium (≈5 GB VRAM)",
    "large-v3": "large-v3 (≈10 GB VRAM, most accurate)",
}


# ---------------------------------------------------------------------------
# ASR Engine
# ---------------------------------------------------------------------------

class ASREngine:
    """Wraps a faster-whisper model for transcription with word-level timestamps."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """
        Parameters
        ----------
        model_size : one of 'tiny', 'base', 'small', 'medium', 'large-v3'
        device     : 'cpu', 'cuda', or 'auto'
        compute_type : 'int8', 'float16', or 'auto'
            'int8' for CPU, 'float16' for GPU, 'auto' lets faster-whisper decide.
        """
        self.model_size = model_size
        self.model: Optional[WhisperModel] = None
        self._device = device
        self._compute_type = compute_type

    def load_model(self):
        if self.model is not None:
            return
        self.model = WhisperModel(
            self.model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

    def transcribe(self, audio_path: str) -> list[dict]:
        """Transcribe *audio_path* (16 kHz mono WAV) and return a list of
        subtitle segments:

            [
                {"id": 0, "start": 1.25, "end": 4.50, "text": "..."},
                ...
            ]
        """
        self.load_model()
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            # Shorter silence threshold → more granular segments; the
            # sentence_segmenter will merge them back into natural sentences.
            vad_parameters={"min_silence_duration_ms": 300},
            # Gentle nudge for Whisper to produce punctuated, capitalised
            # output — which dramatically helps the sentence segmenter.
            initial_prompt="Hello, welcome to today's presentation.",
        )

        results = []
        for i, seg in enumerate(segments):
            timed_words = []
            for word in (getattr(seg, "words", None) or []):
                text = str(getattr(word, "word", "") or "").strip()
                if not text:
                    continue
                try:
                    start = max(0.0, float(word.start))
                    end = max(start, float(word.end))
                except (TypeError, ValueError):
                    continue
                timed_words.append({
                    "word": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                })
            results.append({
                "id": i,
                "start": round(timed_words[0]["start"] if timed_words else seg.start, 3),
                "end": round(timed_words[-1]["end"] if timed_words else seg.end, 3),
                "text": seg.text.strip(),
                "words": timed_words,
            })
        return results
