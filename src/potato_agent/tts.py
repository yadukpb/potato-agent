"""Text-to-speech via Piper (CPU-only, real-time)."""
from __future__ import annotations

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

from . import config


class TextToSpeech:
    def __init__(self, voice_path=None):
        voice_path = str(voice_path or config.PIPER_VOICE_PATH)
        if not config.PIPER_VOICE_PATH.exists() and voice_path == str(config.PIPER_VOICE_PATH):
            raise FileNotFoundError(
                f"Piper voice not found at {voice_path}. Run scripts/setup_models.sh first."
            )
        self._voice = PiperVoice.load(voice_path)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        pcm_chunks = list(self._voice.synthesize_stream_raw(text))
        if not pcm_chunks:
            return
        pcm = b"".join(pcm_chunks)
        audio = np.frombuffer(pcm, dtype=np.int16)
        sd.play(audio, samplerate=self._voice.config.sample_rate)
        sd.wait()
