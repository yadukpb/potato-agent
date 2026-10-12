"""Text-to-speech via Piper (CPU-only, real-time).

speak() splits the reply into sentences and pipelines them: while sentence N
is playing, sentence N+1 is already being synthesized on a background
thread, so there's no audio gap waiting for the next chunk. This doesn't cut
total synthesis time, but it cuts time-to-first-audio for multi-sentence
replies instead of waiting for the whole reply to render before speaking.

Full token-level streaming (starting to speak before the LLM has even
finished generating the answer) isn't implemented here — the agent's answers
are generated inside grammar-constrained JSON (see agent.py), and reliably
extracting a finished sentence from a partially-streamed JSON string mid-
generation is a real parsing problem, not just plumbing. Worth revisiting if
answers get long enough for it to matter.
"""
from __future__ import annotations

import queue
import re
import threading

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

from . import config

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class TextToSpeech:
    def __init__(self, voice_path=None):
        voice_path = str(voice_path or config.PIPER_VOICE_PATH)
        if not config.PIPER_VOICE_PATH.exists() and voice_path == str(config.PIPER_VOICE_PATH):
            raise FileNotFoundError(
                f"Piper voice not found at {voice_path}. Run scripts/setup_models.sh first."
            )
        self._voice = PiperVoice.load(voice_path)

    def _synthesize(self, text: str) -> np.ndarray:
        pcm_chunks = list(self._voice.synthesize_stream_raw(text))
        if not pcm_chunks:
            return np.zeros(0, dtype=np.int16)
        return np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)

    def speak(self, text: str) -> None:
        sentences = split_sentences(text)
        if not sentences:
            return

        audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)

        def producer():
            for sentence in sentences:
                audio_queue.put(self._synthesize(sentence))
            audio_queue.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            if audio.size:
                sd.play(audio, samplerate=self._voice.config.sample_rate)
                sd.wait()
