"""Conversational voice loop: continuous VAD-driven listening -> agent -> speech.

No push-to-talk by default — the agent listens continuously and auto-detects
when you start and stop talking. The mic is only open while listening; it's
closed during the agent's own speech, so it never hears itself (no echo
cancellation needed for that part). What this does NOT do yet is barge-in:
you can't interrupt the agent mid-sentence by talking over it. That needs
echo cancellation or headphones to be reliable and is a reasonable v2, not
bundled into this loop.

Set POTATO_PUSH_TO_TALK=1 to fall back to the old manual press-Enter mode,
e.g. if VAD misfires on a noisy mic/room.
"""
from . import config
from .agent import Agent
from .stt import SpeechToText
from .tts import TextToSpeech
from .vad import VADListener


def main():
    print("potato-agent (voice mode). Loading models...")
    stt = SpeechToText()
    tts = TextToSpeech()
    agent = Agent()
    vad = None if config.PUSH_TO_TALK else VADListener()

    if config.PUSH_TO_TALK:
        print("Ready (push-to-talk mode). Ctrl+C to exit.\n")
    else:
        print("Ready — just start talking. Ctrl+C to exit.\n")

    while True:
        try:
            if vad is None:
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

            answer = agent.run(text)
            print(f"agent: {answer}")
            tts.speak(answer)
        except KeyboardInterrupt:
            print("\nbye.")
            break


if __name__ == "__main__":
    main()
