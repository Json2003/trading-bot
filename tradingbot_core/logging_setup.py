"""Centralised logging helpers used across services and scripts."""
from __future__ import annotations

import json
import logging
import sys
from logging import Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import IO, Any, Mapping

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class JsonFormatter(Formatter):
    """Serialize log records as JSON objects."""

    #: Fields that are part of the :class:`logging.LogRecord` protocol that we
    #: do not want to expose in the JSON payload.
    _RESERVED_FIELDS = {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "stacklevel",
        "thread",
        "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack_info"] = record.stack_info

        payload.update(self._extra_payload(record))
        return json.dumps(payload, default=str)

    def _extra_payload(self, record: logging.LogRecord) -> Mapping[str, Any]:
        """Extract custom attributes added via ``LoggerAdapter`` or ``extra``."""

        custom_fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._RESERVED_FIELDS:
                continue
            custom_fields[key] = value
        return custom_fields


def setup_logging(
    *,
    name: str = "tradingbot",
    level: int | str = logging.INFO,
    stream: IO[str] | None = None,
    log_path: str | Path | None = None,
    formatter: Formatter | str | None = None,
    propagate: bool = False,
) -> Logger:
    """Configure and return a logger instance.

    The helper is intentionally opinionated but still provides a handful of
    extension points so that tests can exercise behaviour without touching the
    global logging configuration from multiple locations.
    """

    logger = getLogger(name)

    if isinstance(level, str):
        level = level.upper()

    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = propagate

    chosen_formatter = _choose_formatter(formatter)

    stream_handler = StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(chosen_formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(Path(log_path))
        file_handler.setFormatter(chosen_formatter)
        logger.addHandler(file_handler)

    return logger


def _choose_formatter(formatter: Formatter | str | None) -> Formatter:
    if formatter is None:
        return Formatter(DEFAULT_LOG_FORMAT)

    if isinstance(formatter, Formatter):
        return formatter

    if formatter.lower() == "json":
        return JsonFormatter()

    return Formatter(formatter)


__all__ = ["setup_logging", "DEFAULT_LOG_FORMAT", "JsonFormatter"]
