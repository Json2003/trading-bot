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
        low=99.0,
        close=100.0,
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
        high=101.0,
        low=80.0,
        close=100.0,
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


def _pair(count):
    return [SimpleNamespace(btc=bar, eth=bar) for bar in _bars(count)]


def test_next_open_four_hour_exit_and_unchanged_stress_costs():
    import pytest
    from scripts import run_failed_breakout_exhaustion as research

    pair = _pair(30)
    pair[24].btc.close = 999.0  # Signal close must never be the fill.
    pair[25].btc.open = 100.0
    pair[26].btc.open = 200.0  # Detect an extra latency bar.
    pair[28].btc.close = 110.0  # Fourth held candle's close.
    pair[29].btc.close = 300.0  # Detect a fifth held candle.
    for symbol in ('BTC', 'ETH'):
        for direction in (1, -1):
            row = research.trade_return(pair, symbol, direction, 24)
            assert row['gross_return'] == pytest.approx(direction * 0.10)
            # $6000 * 80% fill * 98% accepted, 86 bps + 4 * 0.5 bps.
            assert row['execution_cost'] == pytest.approx(6000 * .8 * .98 * .0088)
            assert row['net_pnl'] == pytest.approx(
                6000 * .8 * .98 * (direction * .10 - .0088)
            )
    assert research.trade_return(pair[:29], 'BTC', 1, 24) is not None
    assert research.trade_return(pair[:28], 'BTC', 1, 24) is None


def test_segment_excludes_holdout_prices_and_keeps_windows_separate(monkeypatch):
    from scripts import run_failed_breakout_exhaustion as research

    monkeypatch.setattr(research, 'candidate', lambda *args: {
        'symbol': 'BTC', 'direction': 1, 'rejection': 'downside',
        'volume_ratio': 3.0, 'range_ratio': 2.0,
    })
    pair = _pair(240)
    # Last valid signal 225 exits at 229; signal 226 would use holdout bar 230.
    rows = research.collect_segment(pair, {}, {}, 220, 230)
    assert [row['signal_index'] for row in rows] == [220, 225]
    assert research.collect_segment(pair, {}, {}, 225, 230)
    assert research.collect_segment(pair, {}, {}, 226, 230) == []
    for bar in pair[230:]:
        bar.btc.open = 9000.0
        bar.btc.close = 1.0
    assert research.collect_segment(pair, {}, {}, 220, 230) == rows
    assert research.collect_segment(pair[:230], {}, {}, 220, 230) == rows
