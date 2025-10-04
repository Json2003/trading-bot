"""Tiny ``requests`` shim used inside the offline test environment.

The real project depends on :mod:`requests`, but the execution sandbox does not
allow outbound HTTP calls.  Historically we exposed two helpers (:func:`get`
and :func:`post`) that simply raised :class:`HTTPError`.  Recent changes started
instantiating :class:`requests.Session`, which meant ``import requests`` picked
up this shim and then failed with ``AttributeError: module 'requests' has no
attribute 'Session'``.  The calling code expected the network call to fail – it
handles :class:`HTTPError` and falls back to deterministic sample data – but the
missing attribute caused the pipeline to abort before the error handling kicked
in.

To preserve the no-network guarantee while keeping the public surface area in
line with the real library we provide a minimal :class:`Session` with ``get``
and ``post`` methods that both raise :class:`HTTPError`.  The session also
implements the context-manager protocol so ``with requests.Session()`` keeps
working in tests.
"""


class HTTPError(Exception):
    """Minimal ``requests.HTTPError`` replacement."""


def _raise_unavailable(operation: str) -> "None":  # pragma: no cover - network disabled
    raise HTTPError(f"requests.{operation} unavailable in test environment")


def get(*args, **kwargs):  # pragma: no cover - network disabled
    _raise_unavailable("get")


def post(*args, **kwargs):  # pragma: no cover - network disabled
    _raise_unavailable("post")


class Session:
    """Minimal Session implementation that always fails with ``HTTPError``."""

    def request(self, method: str, *args, **kwargs):  # pragma: no cover - network disabled
        _raise_unavailable(f"Session.{method.lower()}")

    def get(self, *args, **kwargs):  # pragma: no cover - network disabled
        self.request("GET", *args, **kwargs)

    def post(self, *args, **kwargs):  # pragma: no cover - network disabled
        self.request("POST", *args, **kwargs)

    # ``requests.Session`` implements the context manager protocol; mirroring it
    # keeps ``with requests.Session() as session`` working in tests.
    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False
