"""Central logging config for internal diagnostics.

Deliberately separate from the conversational output: what the agent heard
and said is the actual UI and is always print()-ed (voice_loop.py, cli.py),
never gated behind a log level. This is only for things an operator might
want to filter/suppress/elevate -- a recovered mid-conversation error, a
tool call that failed -- via POTATO_LOG_LEVEL (default INFO).
"""
import logging

from . import config


def configure() -> None:
    # Root (and so every third-party library -- ddgs's HTTP request
    # logging via its "primp" logger, in particular) stays at WARNING
    # regardless of POTATO_LOG_LEVEL; only potato_agent's own loggers (via
    # the parent "potato_agent" logger every module logger here is a child
    # of) follow it. Without this split, POTATO_LOG_LEVEL=DEBUG/INFO would
    # also unmute every dependency's own internal logging.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("potato_agent").setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
