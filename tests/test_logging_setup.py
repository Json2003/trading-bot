"""Tests for the logging helper utilities."""
from __future__ import annotations

import io
import json
import logging

import pytest

from tradingbot_core.logging_setup import JsonFormatter, setup_logging


def test_json_formatter_serializes_basic_fields() -> None:
    stream = io.StringIO()
    logger = setup_logging(name="test.json", stream=stream, formatter="json")

    logger.info("hello world")

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "INFO"
    assert payload["name"] == "test.json"
    assert payload["msg"] == "hello world"


def test_json_formatter_serializes_exceptions() -> None:
    stream = io.StringIO()
    logger = setup_logging(name="test.exc", stream=stream, formatter="json")

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("captured")

    payload = json.loads(stream.getvalue().strip())
    assert payload["msg"] == "captured"
    assert "RuntimeError: boom" in payload["exc_info"]


def test_json_formatter_preserves_custom_attributes() -> None:
    stream = io.StringIO()
    logger = setup_logging(name="test.extra", stream=stream, formatter="json")

    logger.info("hello", extra={"context": "value"})

    payload = json.loads(stream.getvalue().strip())
    assert payload["context"] == "value"


def test_setup_logging_accepts_formatter_instances() -> None:
    stream = io.StringIO()
    formatter = JsonFormatter()
    logger = setup_logging(name="test.instance", stream=stream, formatter=formatter)

    logger.warning("warn")

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "WARNING"


def test_setup_logging_accepts_string_format_specifier() -> None:
    stream = io.StringIO()
    logger = setup_logging(name="test.format", stream=stream, formatter="%(message)s")

    logger.info("plain text")

    assert stream.getvalue().strip() == "plain text"


@pytest.mark.parametrize("level", ["info", "INFO", logging.INFO])
def test_setup_logging_accepts_various_level_types(level: int | str) -> None:
    stream = io.StringIO()
    logger = setup_logging(name="test.level", stream=stream, level=level)

    logger.info("hi")

    assert "hi" in stream.getvalue()
