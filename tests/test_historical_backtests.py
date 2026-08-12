from __future__ import annotations

from pathlib import Path

from backtest.engine import ExecConfig
from scripts.run_historical_backtests import run_historical_matrix


def test_historical_matrix_marks_missing_long_horizons_without_filling_them() -> None:
    dataset = Path("tradingbot_ibkr/datafiles/BTC_USDT_bars_annotated.csv")
    report = run_historical_matrix(
        {"BTCUSDT": dataset},
        interval="1h",
        requested_end=None,
        exec_config=ExecConfig(
            fees_bps=10,
            slip_bps=8,
            tp_atr_mult=3,
            sl_atr_mult=1.5,
            atr_period=14,
            risk_per_trade=0.005,
            max_notional_frac=0.9,
            allow_short=True,
            max_bars=24,
        ),
    )
    results = report["results"]
    assert len(results) == 4 * 4
    assert {result["horizon"] for result in results} == {"1d", "1w", "1m", "1y"}
    long_horizon = [result for result in results if result["horizon"] == "1y"]
    assert long_horizon
    assert all(result["status"] == "insufficient_data" for result in long_horizon)
    assert all(result["rows"] < result["expected_rows"] for result in long_horizon)

