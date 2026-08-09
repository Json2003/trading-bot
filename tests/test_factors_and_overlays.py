"""Unit tests for factor and overlay helpers."""

from __future__ import annotations

import math

import pandas as pd

from factors.beta import compute_rolling_beta
from factors.comp_m import compute_comp_m_factor
from overlays.beta_hedge import size_btc_beta_hedge


def test_compute_rolling_beta_matches_market() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    asset = pd.Series([0.01, 0.02, 0.015, 0.0175, 0.0225, 0.03], index=index)
    market = asset.copy()

    beta = compute_rolling_beta(asset, market, window=3)

    assert list(beta.index) == list(asset.index)
    last_values = beta.iloc[-3:]
    for value in last_values:
        if pd.isna(value):
            continue
        assert abs(value - 1.0) < 1e-9


def test_compute_comp_m_factor_returns_zscores() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    prices = pd.DataFrame(
        {
            "A": [10, 10.2, 10.5, 10.7, 11.0, 11.4],
            "B": [10, 10.05, 10.1, 10.15, 10.2, 10.3],
            "C": [10, 9.9, 9.7, 9.6, 9.4, 9.1],
        },
        index=index,
    )

    zscores = compute_comp_m_factor(prices, lookback=2, lag=1)

    # Ensure demeaned cross-section each timestamp
    columns = list(zscores.columns)
    index = list(zscores.index)
    for pos in range(len(index)):
        row_vals = []
        for col in columns:
            series = zscores[col]
            value = series[pos]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            row_vals.append(value)
        if not row_vals:
            continue
        mean_val = sum(row_vals) / len(row_vals)
        assert abs(mean_val) < 1e-9

    # Asset A has the strongest momentum and should have positive score.
    last_pos = len(index) - 1
    last_row_values = {col: zscores[col].iloc[last_pos] for col in columns}
    assert last_row_values["A"] > 0
    assert last_row_values["C"] < 0


def test_size_btc_beta_hedge_uses_band_and_buffer() -> None:
    positions = pd.Series({"ETH": 1.0, "SOL": 1.0})
    prices = pd.Series({"ETH": 2000.0, "SOL": 1000.0})
    betas = pd.Series({"ETH": 1.2, "SOL": 0.8})

    contracts = size_btc_beta_hedge(
        positions,
        betas,
        prices,
        target_beta=(0.1, 0.2),
        btc_price=30000.0,
        contract_size=1.0,
        rebalance_buffer=0.02,
    )

    assert contracts < 0  # Should short BTC to bring beta down

    # Now bring beta inside the buffer to confirm no trade is suggested.
    betas_close = pd.Series({"ETH": 0.15})
    prices_close = pd.Series({"ETH": 1000.0})
    positions_close = pd.Series({"ETH": 1.0})
    contracts_close = size_btc_beta_hedge(
        positions_close,
        betas_close,
        prices_close,
        target_beta=(0.1, 0.2),
        btc_price=30000.0,
        contract_size=1.0,
        rebalance_buffer=0.05,
    )

    assert math.isclose(contracts_close, 0.0)
