"""Offline-friendly subset of the :mod:`requests` API used in tests.

The production code depends on ``requests`` but the execution environment blocks
outbound networking.  We expose just enough of the public surface to let callers
construct sessions, tweak headers, mount adapters, and then deterministically
fail when an actual HTTP call is attempted.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class RequestsWarning(Warning):
    """Base warning matching :mod:`requests.exceptions`."""


class RequestsDependencyWarning(RequestsWarning):
    """Warns about incompatible optional dependencies."""


class FileModeWarning(RequestsWarning, DeprecationWarning):
    """Warns when files are opened in text mode but length is inspected."""


class RequestException(Exception):
    """Base exception matching ``requests.exceptions.RequestException``."""


class HTTPError(RequestException):
    """Raised whenever the shim prevents a real HTTP call."""


class ConnectionError(RequestException):
    """Placeholder for :class:`requests.exceptions.ConnectionError`."""


class Timeout(RequestException):
    """Raised when callers expect network timeouts."""


class TooManyRedirects(RequestException):
    """Raised when redirect handling would exceed limits."""


class URLRequired(RequestException):
    """Raised when a URL is required but missing."""


class InvalidURL(RequestException):
    """Raised when the provided URL is invalid."""


class InvalidHeader(RequestException):
    """Raised when a header value is invalid."""


class InvalidSchema(RequestException):
    """Raised when the URL schema is unsupported."""


class MissingSchema(RequestException):
    """Raised when the URL schema is missing."""


class InvalidProxyURL(InvalidURL):
    """Raised when the proxy URL is invalid."""


class ProxyError(ConnectionError):
    """Raised when a proxy error occurs."""


class SSLError(ConnectionError):
    """Raised when SSL negotiation fails."""


class ConnectTimeout(ConnectionError, Timeout):
    """Raised when a connection attempt times out."""


class ReadTimeout(Timeout):
    """Raised when a response is not received in time."""


class ChunkedEncodingError(RequestException):
    """Raised when chunked transfer encoding is invalid."""


class ContentDecodingError(RequestException):
    """Raised when response decoding fails."""


class StreamConsumedError(RequestException):
    """Raised when streamed content has already been consumed."""


class RetryError(RequestException):
    """Raised when retries are exhausted."""


class UnrewindableBodyError(RequestException):
    """Raised when request body cannot be rewound."""


class InvalidJSONError(RequestException):
    """Raised when JSON decoding fails."""


class JSONDecodeError(InvalidJSONError):
    """Compatibility shim for ``requests.exceptions.JSONDecodeError``."""


def _raise_unavailable(operation: str) -> None:  # pragma: no cover - network disabled
    raise HTTPError(f"requests.{operation} unavailable in the offline test environment")


def request(method: str, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    _raise_unavailable(method.lower())


def get(*args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    request("GET", *args, **kwargs)


def post(*args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    request("POST", *args, **kwargs)


def put(*args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    request("PUT", *args, **kwargs)


def delete(*args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    request("DELETE", *args, **kwargs)


def head(*args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    request("HEAD", *args, **kwargs)


def options(*args: Any, **kwargs: Any):  # pragma: no cover - network disabled
    request("OPTIONS", *args, **kwargs)


class Session:
    """Minimal ``requests.Session`` replacement."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        self.headers: dict[str, str] = {}
        self._mounts: dict[str, Any] = {}

    def request(self, method: str, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        _raise_unavailable(f"Session.{method.lower()}")

    def get(self, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        return self.request("GET", *args, **kwargs)

    def post(self, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        return self.request("POST", *args, **kwargs)

    def put(self, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        return self.request("PUT", *args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        return self.request("DELETE", *args, **kwargs)

    def head(self, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        return self.request("HEAD", *args, **kwargs)

    def options(self, *args: Any, **kwargs: Any):  # pragma: no cover - network disabled
        return self.request("OPTIONS", *args, **kwargs)

    def mount(self, prefix: str, adapter: Any) -> None:  # pragma: no cover - trivial
        self._mounts[prefix] = adapter

    def close(self) -> None:  # pragma: no cover - trivial
        self._mounts.clear()

    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        self.close()
        return False


class Response:
    """Small ``requests.Response`` placeholder for tests inspecting attributes."""

    def __init__(self, status_code: int = 599, text: str = "") -> None:  # pragma: no cover - trivial
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - network disabled
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# Submodule shims
# ---------------------------------------------------------------------------

_adapters_module = types.ModuleType(f"{__name__}.adapters")


class HTTPAdapter:  # pragma: no cover - trivial configuration helper
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


_adapters_module.HTTPAdapter = HTTPAdapter
sys.modules[f"{__name__}.adapters"] = _adapters_module


_exceptions_module = types.ModuleType(f"{__name__}.exceptions")
for _cls in (
    RequestException,
    HTTPError,
    ConnectionError,
    Timeout,
    TooManyRedirects,
    URLRequired,
    InvalidURL,
    InvalidHeader,
    InvalidSchema,
    MissingSchema,
    InvalidProxyURL,
    ProxyError,
    SSLError,
    ConnectTimeout,
    ReadTimeout,
    ChunkedEncodingError,
    ContentDecodingError,
    StreamConsumedError,
    RetryError,
    UnrewindableBodyError,
    InvalidJSONError,
    JSONDecodeError,
):
    setattr(_exceptions_module, _cls.__name__, _cls)
_exceptions_module.RequestsWarning = RequestsWarning
_exceptions_module.RequestsDependencyWarning = RequestsDependencyWarning
_exceptions_module.FileModeWarning = FileModeWarning
sys.modules[f"{__name__}.exceptions"] = _exceptions_module


__all__ = [
    "RequestsWarning",
    "RequestsDependencyWarning",
    "FileModeWarning",
    "RequestException",
    "HTTPError",
    "ConnectionError",
    "Timeout",
    "TooManyRedirects",
    "URLRequired",
    "InvalidURL",
    "InvalidHeader",
    "InvalidSchema",
    "MissingSchema",
    "InvalidProxyURL",
    "ProxyError",
    "SSLError",
    "ConnectTimeout",
    "ReadTimeout",
    "ChunkedEncodingError",
    "ContentDecodingError",
    "StreamConsumedError",
    "RetryError",
    "UnrewindableBodyError",
    "InvalidJSONError",
    "JSONDecodeError",
    "Response",
    "Session",
    "HTTPAdapter",
    "request",
    "get",
    "post",
    "put",
    "delete",
    "head",
    "options",
]
