"""Centralised logging helpers used across services and scripts."""
from __future__ import annotations

from logging import Formatter, Logger, StreamHandler, getLogger
import logging
from pathlib import Path
from typing import IO
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(
    *,
    name: str = "tradingbot",
    level: int | str = logging.INFO,
    stream: IO[str] | None = None,
    log_path: str | Path | None = None,
    formatter: Formatter | None = None,
    propagate: bool = False,
) -> Logger:
    """Configure and return a logger instance.

    The helper is intentionally opinionated but still provides a handful of
    extension points so that tests can exercise behaviour without touching the
    global logging configuration from multiple locations.
    """

    logger = getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = propagate

    chosen_formatter = formatter or Formatter(DEFAULT_LOG_FORMAT)

    stream_handler = StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(chosen_formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(Path(log_path))
        file_handler.setFormatter(chosen_formatter)
        logger.addHandler(file_handler)

    return logger


__all__ = ["setup_logging", "DEFAULT_LOG_FORMAT"]
