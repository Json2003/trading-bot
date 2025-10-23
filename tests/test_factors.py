"""Unit tests for factor and overlay utilities."""

from __future__ import annotations

import pandas as pd

from factors.beta import compute_rolling_beta
from factors.comp_m import compute_comp_m_factor
from overlays.beta_hedge import size_btc_beta_hedge


def test_compute_rolling_beta_tracks_market_relationship() -> None:
    idx = pd.date_range("2023-01-01", periods=6, freq="T")
    market = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, 0.0], index=idx)
    asset = market * 2

    beta = compute_rolling_beta(asset, market, window=3)

    assert list(beta.index) == list(asset.index)
    assert beta.iloc[-1] == 2.0
    assert pd.isna(beta.iloc[0])


def test_compute_comp_m_factor_returns_cross_sectional_zscores() -> None:
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "A": [100, 110, 121],
            "B": [100, 105, 110.25],
        },
        index=idx,
    )

    factor = compute_comp_m_factor(prices, lookback=1, lag=0)
    latest = factor.iloc[-1]

    assert abs(latest["A"] - 1.0) < 1e-8
    assert abs(latest["B"] + 1.0) < 1e-8


def test_size_btc_beta_hedge_supports_target_range() -> None:
    positions = pd.Series({"ETHUSDT": 2.0, "SOLUSDT": 1.0})
    betas = pd.Series({"ETHUSDT": 1.0, "SOLUSDT": 0.5})
    prices = pd.Series({"ETHUSDT": 1000.0, "SOLUSDT": 500.0})

    contracts = size_btc_beta_hedge(
        positions,
        betas,
        prices,
        target_beta=(0.1, 0.2),
        btc_price=25000.0,
        contract_size=1.0,
    )

    assert contracts == -0.07

    within_band = size_btc_beta_hedge(
        positions,
        betas,
        prices,
        target_beta=(0.1, 1.0),
        btc_price=25000.0,
        contract_size=1.0,
    )

    assert within_band == 0.0
