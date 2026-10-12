"""Push-to-talk voice loop: mic -> STT -> agent -> TTS -> speaker."""
from .agent import Agent
from .stt import SpeechToText
from .tts import TextToSpeech


def main():
    print("potato-agent (voice mode). Loading models...")
    stt = SpeechToText()
    tts = TextToSpeech()
    agent = Agent()
    print("Ready. Ctrl+C to exit.\n")

    while True:
        try:
            text = stt.listen_and_transcribe()
            if not text:
                print("(heard nothing, try again)")
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
