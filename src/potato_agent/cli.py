"""Text-only REPL — for testing the agent without the audio pipeline."""
from .agent import Agent


def main():
    print("potato-agent (text mode). /reset clears conversation history. Ctrl+C to exit.\n")
    agent = Agent()
    agent.warm_up()  # pays the LLM's one-time ~7s first-call cost now, not on your first message
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user_input:
            continue
        if user_input == "/reset":
            agent.clear_history()
            print("(conversation history cleared)\n")
            continue
        answer = agent.run(user_input)
        print(f"agent> {answer}\n")


if __name__ == "__main__":
    main()
