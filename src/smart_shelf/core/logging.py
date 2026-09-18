"""Structured logging setup built on ``structlog``.

Local development gets colourised key/value output; every other environment
gets newline-delimited JSON so logs are directly ingestible by Loki, Datadog or
CloudWatch. :func:`configure_logging` is idempotent and safe to call from both
the API entrypoint and test fixtures.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from smart_shelf.core.config import Settings, get_settings

_CONFIGURED = False


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Install structlog processors and align the stdlib root logger with them."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = settings or get_settings()
    level = getattr(logging, settings.log_level)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.use_json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level, force=True)
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Call sites pass ``__name__``.

    The module name is bound as an event key rather than resolved by
    ``add_logger_name``, because the print-based logger factory used here has no
    stdlib logger behind it to read a name from.
    """
    configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger
