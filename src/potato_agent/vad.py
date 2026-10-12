"""Continuous listening via WebRTC VAD — no push-to-talk, automatic turn detection.

Chosen over Silero VAD deliberately: webrtcvad is a thin C-extension with no
torch dependency, which matters for a project that's supposed to stay light
on a CPU-only laptop. It's less accurate than Silero but fast and proven
(it's WebRTC's own VAD, used in real-time calls for over a decade).
"""
from __future__ import annotations

import collections
import queue
from typing import Callable

import numpy as np
import sounddevice as sd
import webrtcvad

from . import config

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples/frame

SPEECH_START_FRAMES = 3  # ~90ms of speech to trigger the start of an utterance
SILENCE_END_FRAMES = 20  # ~600ms of trailing silence to end an utterance
MIN_UTTERANCE_FRAMES = 6  # ~180ms minimum; filters out clicks/noise blips
PREROLL_FRAMES = 5  # ~150ms of audio kept from before speech onset, so words aren't clipped


class VADListener:
    def __init__(self, aggressiveness: int | None = None):
        self._vad = webrtcvad.Vad(aggressiveness if aggressiveness is not None else config.VAD_AGGRESSIVENESS)

    def listen_for_utterance(
        self,
        on_speech_start: Callable[[], None] | None = None,
        required_speech_frames: Callable[[], int] | int | None = None,
    ) -> np.ndarray:
        """Blocks until speech is detected and then trails off into silence.

        Returns the captured utterance as float32 PCM at 16kHz, ready for STT.
        Opens the mic only for the duration of one utterance — closing it
        afterward (the caller does TTS playback outside this call) is what
        keeps the agent from hearing its own voice, for callers that aren't
        doing barge-in.

        on_speech_start, if given, fires exactly once, the moment speech
        onset is detected (not when the utterance finishes). voice_loop.py
        uses this to interrupt the agent's own TTS playback the instant the
        user starts talking over it.

        required_speech_frames overrides SPEECH_START_FRAMES, as a plain
        int or as a zero-arg callable evaluated on every frame (so a caller
        can demand a stricter, slower-to-trigger threshold -- e.g. while
        its own TTS is still playing and might bleed into the mic -- then
        relax back to the normal threshold once that's no longer a risk).
        """
        frame_queue: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, time_info, status):
            pcm16 = (indata[:, 0] * 32767.0).astype(np.int16).tobytes()
            frame_queue.put(pcm16)

        preroll: collections.deque[bytes] = collections.deque(maxlen=PREROLL_FRAMES)
        voiced_frames: list[bytes] = []
        speech_run = 0
        silence_run = 0
        triggered = False

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=callback,
        )
        with stream:
            while True:
                frame_bytes = frame_queue.get()
                is_speech = self._vad.is_speech(frame_bytes, SAMPLE_RATE)

                if not triggered:
                    preroll.append(frame_bytes)
                    if is_speech:
                        speech_run += 1
                        needed = (
                            required_speech_frames()
                            if callable(required_speech_frames)
                            else required_speech_frames or SPEECH_START_FRAMES
                        )
                        if speech_run >= needed:
                            triggered = True
                            voiced_frames.extend(preroll)
                            preroll.clear()
                            silence_run = 0
                            if on_speech_start is not None:
                                on_speech_start()
                    else:
                        speech_run = 0
                else:
                    voiced_frames.append(frame_bytes)
                    if is_speech:
                        silence_run = 0
                    else:
                        silence_run += 1
                        if silence_run >= SILENCE_END_FRAMES:
                            break

        if len(voiced_frames) < MIN_UTTERANCE_FRAMES:
            return np.zeros(0, dtype=np.float32)

        pcm = b"".join(voiced_frames)
        pcm16 = np.frombuffer(pcm, dtype=np.int16)
        return pcm16.astype(np.float32) / 32768.0
