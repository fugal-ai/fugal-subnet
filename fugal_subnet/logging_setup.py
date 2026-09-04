"""Logging configuration that survives importing bittensor.

Importing `bittensor` initialises its logging machine, which runs a
`logging.config.dictConfig(...)` with `disable_existing_loggers` left at its
default of **True**. That sets `.disabled = True` on every logger that already
exists — which includes any module-level `logging.getLogger(__name__)` in this
package, because our modules are imported first.

The effect is total and silent: `logger.info(...)`, `logger.error(...)` and
`logger.exception(...)` from the neurons all return without emitting anything.
A miner failing every epoch looks exactly like a miner sitting idle, and the
traceback that would explain it is written to nowhere. That is not a
hypothetical — a miner in this state logged its backbone load and its on-chain
commitment (modules imported lazily, *after* bittensor, so their loggers were
created too late to be disabled) while every line from `fugal.miner` itself
vanished.

Call `configure_logging()` AFTER importing bittensor.
"""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def configure_logging(level: str = "INFO", root_name: str = "fugal") -> None:
    """Install a handler and re-enable loggers bittensor disabled.

    Args:
        level: Level name for our own loggers.
        root_name: Logger prefix to re-enable, alongside `fugal_subnet`.
    """
    numeric = getattr(logging, str(level).upper(), logging.INFO)

    root = logging.getLogger()
    # Only add a handler if nothing will already print our records; otherwise
    # every line appears twice (basicConfig's handler plus ours).
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    if root.level > numeric or root.level == logging.NOTSET:
        root.setLevel(numeric)

    # Undo dictConfig(disable_existing_loggers=True) for our own loggers.
    prefixes = (root_name, "fugal_subnet")
    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(obj, logging.Logger):
            continue
        if name.split(".", 1)[0] in prefixes:
            obj.disabled = False
            obj.setLevel(numeric)
