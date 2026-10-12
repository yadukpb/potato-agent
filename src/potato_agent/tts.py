"""Text-to-speech via Piper (CPU-only, real-time).

speak()/speak_stream() pipeline synthesis and playback sentence-by-sentence:
while sentence N is playing, sentence N+1 is already being synthesized on a
background thread, so there's no audio gap waiting for the next chunk.

speak_stream() takes that one step further: the sentences themselves can
still be arriving (e.g. agent.py's on_sentence callback, called as the LLM's
grammar-constrained JSON streams in -- see its _decode_partial_json_string).
That cuts time-to-first-audio for the whole reply, not just for sentences
2+, since playback of sentence 1 can start the moment the LLM finishes it
rather than after the full response (which may still involve tool calls)
has been generated.
"""
from __future__ import annotations

import queue
import re
import threading
from typing import Iterable

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
        pcm_chunks = [chunk.audio_int16_bytes for chunk in self._voice.synthesize(text)]
        if not pcm_chunks:
            return np.zeros(0, dtype=np.int16)
        return np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        """Speak a complete, already-finished piece of text."""
        self.speak_stream(split_sentences(text), stop_event=stop_event)

    def speak_stream(self, sentences: Iterable[str], stop_event: threading.Event | None = None) -> None:
        """Speak sentences as they arrive from `sentences` (a list, or a
        generator still being fed from elsewhere -- iterating blocks until
        the next sentence is available or the iterable ends).

        If `stop_event` is given and gets set (e.g. by VAD detecting the
        user has started talking over it -- see voice_loop.py's barge-in
        wiring), playback is cut immediately rather than finishing the
        current sentence or waiting for more to arrive.
        """
        audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)

        def producer():
            for sentence in sentences:
                if stop_event is not None and stop_event.is_set():
                    break
                if sentence.strip():
                    audio_queue.put(self._synthesize(sentence))
            audio_queue.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            if stop_event is not None and stop_event.is_set():
                sd.stop()
                break
            audio = audio_queue.get()
            if audio is None:
                break
            if audio.size:
                sd.play(audio, samplerate=self._voice.config.sample_rate)
                stream = sd.get_stream()
                while stream.active:
                    if stop_event is not None and stop_event.is_set():
                        sd.stop()
                        break
                    sd.sleep(30)
