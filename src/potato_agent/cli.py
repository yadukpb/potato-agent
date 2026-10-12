"""Text-only REPL — for testing the agent without the audio pipeline."""
from .agent import Agent


def main():
    print("potato-agent (text mode). Ctrl+C to exit.\n")
    agent = Agent()
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user_input:
            continue
        answer = agent.run(user_input)
        print(f"agent> {answer}\n")


if __name__ == "__main__":
    main()
