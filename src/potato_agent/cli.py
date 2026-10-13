"""Text-only REPL — for testing the agent without the audio pipeline."""
import logging

from . import logging_setup
from .agent import Agent

log = logging.getLogger(__name__)


def main():
    logging_setup.configure()
    print("potato-agent (text mode). /reset clears conversation history. Ctrl+C to exit.\n")
    try:
        agent = Agent()
        agent.warm_up()  # pays the LLM's one-time ~7s first-call cost now, not on your first message
    except FileNotFoundError as e:
        print(f"\n{e}")
        raise SystemExit(1)

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
        try:
            answer = agent.run(user_input)
        except Exception:
            log.exception("agent.run() failed")
            print("(that one failed -- see the error above, try again)\n")
            continue
        print(f"agent> {answer}\n")


if __name__ == "__main__":
    main()
