from datetime import datetime, timedelta, timezone

from scripts.run_large_trader_positioning import _coverage, _signal, _summary
from scripts.run_momentum_volatility_research import Bar

UTC = timezone.utc


def _bars(count=30):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Bar(
            timestamp=start + timedelta(hours=i),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=10.0,
        )
        for i in range(count)
    ]


def test_signal_requires_agreeing_top_trader_side_and_volume():
    bars = _bars()
    bars[-1] = Bar(
        timestamp=bars[-1].timestamp,
        open=100.0,
        high=101.0,
        low=100.0,
        close=100.6,
        volume=20.0,
    )
    positioning = {
        bars[-1].timestamp: {
            "account_long": 0.56,
            "account_short": 0.44,
            "position_long": 0.61,
            "position_short": 0.39,
            "account_long_short_ratio": 1.27,
            "position_long_short_ratio": 1.56,
        }
    }
    signal = _signal(bars, positioning, len(bars) - 1)
    assert signal is not None
    side, price_move, volume_ratio, cohort = signal
    assert side == 1
    assert price_move >= 0.005
    assert volume_ratio >= 1.5
    assert cohort["position_long"] == 0.61


def test_signal_rejects_disagreeing_cohort():
    bars = _bars()
    bars[-1] = Bar(
        timestamp=bars[-1].timestamp,
        open=100.0,
        high=100.0,
        low=99.0,
        close=99.4,
        volume=20.0,
    )
    positioning = {
        bars[-1].timestamp: {
            "account_long": 0.56,
            "account_short": 0.44,
            "position_long": 0.61,
            "position_short": 0.39,
            "account_long_short_ratio": 1.27,
            "position_long_short_ratio": 1.56,
        }
    }
    assert _signal(bars, positioning, len(bars) - 1) is None


def test_summary_reports_costs_and_drawdown():
    rows = [
        {
            "signal_timestamp": "2026-01-01T00:00:00+00:00",
            "net_pnl": 50.0,
            "net_return": 50.0 / 3000.0,
            "execution_cost": 10.0,
        },
        {
            "signal_timestamp": "2026-01-02T00:00:00+00:00",
            "net_pnl": -20.0,
            "net_return": -20.0 / 3000.0,
            "execution_cost": 10.0,
        },
    ]
    result = _summary(
        rows,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 7, tzinfo=UTC),
    )
    assert result["trade_count"] == 2
    assert result["net_pnl"] == 30.0
    assert result["execution_cost"] == 20.0
    assert result["profit_factor"] == 2.5
    assert result["max_drawdown_pct_of_notional"] > 0


def test_coverage_rejects_missing_evaluation_hour():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    positioning = {start: {}}
    result = _coverage(positioning, start, start + timedelta(hours=2))
    assert result["expected_hour_count"] == 2
    assert result["observed_row_count"] == 1
    assert result["missing_hour_count"] == 1
    assert result["complete"] is False


def test_coverage_accepts_complete_evaluation_hours():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    positioning = {
        start: {},
        start + timedelta(hours=1): {},
    }
    result = _coverage(positioning, start, start + timedelta(hours=2))
    assert result["missing_hour_count"] == 0
    assert result["complete"] is True
