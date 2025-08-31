import pytest
from unittest.mock import patch, Mock
import requests

from tradingbot_ibkr import economic_fetch as ef


def test_get_retries_on_failure():
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        m = Mock()
        m.status_code = 500
        def raise_for_status():
            raise requests.HTTPError('server error')
        m.raise_for_status = raise_for_status
        return m

    with patch('requests.get', side_effect=fake_get):
        with pytest.raises(Exception):
            ef._get('http://example.com', {}, retries=2, backoff_base=1, timeout=1, jitter_max=0)
    assert len(calls) >= 2


def test_get_handles_429_and_retry(monkeypatch):
    sequence = []

    def fake_get(url, params=None, timeout=None):
        # first call returns 429, second call returns 200
        if len(sequence) == 0:
            sequence.append('429')
            m = Mock()
            m.status_code = 429
            m.headers = {'Retry-After': '1'}
            def raise_for_status():
                raise requests.HTTPError('429')
            m.raise_for_status = raise_for_status
            return m
        else:
            m = Mock()
            m.status_code = 200
            m.json = lambda: {'observations': []}
            def raise_for_status():
                return None
            m.raise_for_status = raise_for_status
            return m

    with patch('requests.get', side_effect=fake_get):
        res = ef._get('http://example.com', {}, retries=3, backoff_base=1, timeout=1, jitter_max=0)
        assert isinstance(res, dict)
