class HTTPError(Exception):
    """Minimal HTTPError used for retry logic tests."""
    pass

def get(*args, **kwargs):  # pragma: no cover - network disabled
    raise HTTPError("requests.get unavailable in test environment")

def post(*args, **kwargs):  # pragma: no cover - network disabled
    raise HTTPError("requests.post unavailable in test environment")
