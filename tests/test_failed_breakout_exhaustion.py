from types import SimpleNamespace

from scripts.run_failed_breakout_exhaustion import candidate, features


def _bars(count=26):
    return [
        SimpleNamespace(
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=100.0,
        )
        for _ in range(count)
    ]


def test_upside_rejection_is_short():
    bars = _bars()
    bars[24] = SimpleNamespace(
        open=109.0,
        high=120.0,
        low=108.0,
        close=109.0,
        volume=300.0,
    )
    btc = features(bars)
    eth = features(_bars())
    picked = candidate(24, btc, eth)
    assert picked is not None
    assert picked["symbol"] == "BTC"
    assert picked["direction"] == -1
    assert picked["rejection"] == "upside"


def test_downside_rejection_is_long():
    bars = _bars()
    bars[24] = SimpleNamespace(
        open=91.0,
        high=92.0,
        low=80.0,
        close=91.0,
        volume=300.0,
    )
    btc = features(bars)
    eth = features(_bars())
    picked = candidate(24, btc, eth)
    assert picked is not None
    assert picked["direction"] == 1
    assert picked["rejection"] == "downside"


def test_two_sided_breach_is_excluded():
    bars = _bars()
    bars[24] = SimpleNamespace(
        open=100.0,
        high=120.0,
        low=80.0,
        close=100.0,
        volume=300.0,
    )
    btc = features(bars)
    eth = features(_bars())
    assert candidate(24, btc, eth) is None
