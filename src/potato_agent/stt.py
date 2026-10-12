"""Speech-to-text: push-to-talk mic capture + faster-whisper transcription."""
from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import config

SAMPLE_RATE = 16000


class SpeechToText:
    def __init__(self, model_size: str | None = None, compute_type: str | None = None):
        self._model = WhisperModel(
            model_size or config.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=compute_type or config.WHISPER_COMPUTE_TYPE,
        )

    def transcribe_array(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def record_push_to_talk(self) -> np.ndarray:
        """Blocks: press Enter to start, speak, press Enter again to stop."""
        input("Press Enter to start speaking...")
        print("Recording... press Enter to stop.")

        chunks: queue.Queue[np.ndarray] = queue.Queue()
        stop_event = threading.Event()

        def callback(indata, frames, time_info, status):
            chunks.put(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        )

        def wait_for_stop():
            input()
            stop_event.set()

        waiter = threading.Thread(target=wait_for_stop, daemon=True)
        with stream:
            waiter.start()
            while not stop_event.is_set():
                stop_event.wait(0.05)

        frames = []
        while not chunks.empty():
            frames.append(chunks.get())
        if not frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(frames).flatten()
        print(f"Captured {len(audio) / SAMPLE_RATE:.1f}s of audio.")
        return audio

    def listen_and_transcribe(self) -> str:
        audio = self.record_push_to_talk()
        if audio.size == 0:
            return ""
        return self.transcribe_array(audio)
