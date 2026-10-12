"""Conversational voice loop: continuous VAD-driven listening -> agent -> speech.

No push-to-talk by default — the agent listens continuously and auto-detects
when you start and stop talking.

Three things make this feel closer to a real conversation than a strict
record -> think -> speak cascade would:

1. Warmup: agent.warm_up() below pays llama.cpp's one-time first-call cost
   (~7-9s on a modest CPU, mostly backend/thread-pool init) up front during
   "Loading models...", instead of it landing on your first real turn. It
   specifically pre-fills the KV cache with SYSTEM_PROMPT, which only helps
   because every run() call starts with that same prefix and llama-cpp-
   python skips re-evaluating a matching prefix -- warming up with any
   other prompt buys nothing. See Agent.warm_up()'s docstring.

2. Streaming: agent.run()'s on_sentence callback fires on each sentence of
   the final answer as soon as the LLM finishes generating it, so TTS
   starts speaking sentence 1 while the model may still be generating
   sentence 2+ (or, for tool-using turns, while it was still running the
   tool). See _speak_streaming below and agent.py's streaming JSON parser.

3. Barge-in: while the agent is speaking, we're already listening for the
   user's *next* utterance via VAD (see vad.listen_for_utterance's
   on_speech_start) instead of waiting for TTS to finish first. Speech
   onset during playback both interrupts the agent and is the start of
   that next utterance, so the same listen call just keeps capturing it --
   no separate "cancel and restart listening" step needed. Set
   POTATO_BARGE_IN=0 if this misfires on your setup (no echo cancellation
   means the mic can hear the agent's own voice from the speakers;
   headphones make it reliable).

Set POTATO_PUSH_TO_TALK=1 to fall back to the old manual press-Enter mode,
e.g. if VAD doesn't suit your mic/room (this also disables barge-in, since
there's no continuous mic stream to listen on while speaking).
"""
import queue
import threading
from typing import Iterator

from . import config
from .agent import Agent
from .stt import SpeechToText
from .tts import TextToSpeech
from .vad import SPEECH_START_FRAMES, VADListener

_SENTINEL = object()


def _drain(sentence_queue: queue.Queue) -> Iterator[str]:
    """Turns a queue fed by agent.run()'s on_sentence callback into an
    iterator TextToSpeech.speak_stream can consume, blocking for the next
    sentence until one arrives or the producer signals it's done.
    """
    while True:
        item = sentence_queue.get()
        if item is _SENTINEL:
            return
        yield item


def _speak_streaming(
    agent: Agent, tts: TextToSpeech, text: str, stop_event: threading.Event
) -> tuple[threading.Thread, threading.Thread]:
    """Starts generating the agent's reply and speaking it as sentences
    become available, both in background threads. Returns (agent_thread,
    speak_thread); the caller should join both (joining speak_thread alone
    very likely means agent_thread already finished too, since it's the one
    that unblocks speak_thread, but joining it explicitly removes any doubt
    and keeps the one-statement-after-the-put race in generate() harmless).
    """
    sentence_queue: queue.Queue = queue.Queue()

    def generate():
        answer = agent.run(text, on_sentence=sentence_queue.put)
        sentence_queue.put(_SENTINEL)
        print(f"agent: {answer}")

    agent_thread = threading.Thread(target=generate, daemon=True)
    agent_thread.start()

    speak_thread = threading.Thread(
        target=tts.speak_stream, args=(_drain(sentence_queue),), kwargs={"stop_event": stop_event}, daemon=True
    )
    speak_thread.start()
    return agent_thread, speak_thread


def main():
    print("potato-agent (voice mode). Loading models...")
    stt = SpeechToText()
    tts = TextToSpeech()
    agent = Agent()
    agent.warm_up()  # pays the LLM's one-time ~7s first-call cost now, not on your first turn
    vad = None if config.PUSH_TO_TALK else VADListener()
    barge_in = vad is not None and config.BARGE_IN_ENABLED

    if config.PUSH_TO_TALK:
        print("Ready (push-to-talk mode). Ctrl+C to exit.\n")
    elif barge_in:
        print("Ready — just start talking. Talk over me any time to interrupt. Ctrl+C to exit.\n")
    else:
        print("Ready — just start talking. Ctrl+C to exit.\n")

    pending_audio = None  # an utterance already captured while the agent was still speaking

    while True:
        try:
            if pending_audio is not None:
                audio, pending_audio = pending_audio, None
            elif vad is None:
                audio = stt.record_push_to_talk()
            else:
                print("[listening...]")
                audio = vad.listen_for_utterance()

            if audio.size == 0:
                print("(heard nothing, try again)")
                continue

            text = stt.transcribe_array(audio)
            if not text:
                print("(couldn't make out any words, try again)")
                continue
            print(f"you said: {text}")

            interrupted = threading.Event()
            agent_thread, speak_thread = _speak_streaming(agent, tts, text, interrupted)

            if barge_in:
                pending_audio = vad.listen_for_utterance(
                    on_speech_start=interrupted.set,
                    required_speech_frames=lambda: (
                        config.VAD_BARGE_IN_FRAMES if speak_thread.is_alive() else SPEECH_START_FRAMES
                    ),
                )
                if pending_audio.size == 0:
                    pending_audio = None

            agent_thread.join()
            speak_thread.join()
        except KeyboardInterrupt:
            print("\nbye.")
            break


if __name__ == "__main__":
    main()
