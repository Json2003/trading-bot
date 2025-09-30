"""Simple SMA crossover backtest example.

This script demonstrates a moving-average crossover strategy using
synthetic closing price data. It is intentionally lightweight so that
it can be used as a quick sanity check for the data pipeline or during
interactive exploration sessions. For production usage, replace the
synthetic data generator with a real data source and integrate the
signal generation logic into the broader trading framework.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for the SMA crossover backtest."""

    short_window: int = 50
    long_window: int = 200
    trading_days: int = 252
    risk_free_rate: float = 0.0


def generate_synthetic_prices(days: int = 1000, seed: int | None = 42) -> pd.DataFrame:
    """Create a synthetic closing price series.

    The prices follow a random walk with a small positive drift so that the
    example has a non-zero trend. Deterministic randomness via ``seed`` keeps
    the demo reproducible.
    """

    if days <= 0:
        raise ValueError("'days' must be a positive integer")

    rng = np.random.default_rng(seed)
    price_changes = rng.normal(loc=0.1, scale=2.0, size=days)
    prices = np.cumsum(price_changes) + 100
    return pd.DataFrame({"Close": prices})


def compute_sma_crossover(
    data: pd.DataFrame, *, short_window: int, long_window: int
) -> pd.DataFrame:
    """Compute SMA crossover signals and returns.

    Parameters
    ----------
    data:
        DataFrame that must contain a ``Close`` column.
    short_window:
        Lookback period for the fast moving average.
    long_window:
        Lookback period for the slow moving average.
    """

    if short_window <= 0 or long_window <= 0:
        raise ValueError("Moving-average windows must be positive integers")
    if short_window >= long_window:
        raise ValueError("The short window must be smaller than the long window")

    df = data.copy()
    df["SMA_short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_long"] = df["Close"].rolling(window=long_window).mean()

    df["Signal"] = 0
    df.loc[long_window:, "Signal"] = (
        df.loc[long_window:, "SMA_short"] > df.loc[long_window:, "SMA_long"]
    ).astype(int)
    df["Position"] = df["Signal"].diff().fillna(0)

    df["Return"] = df["Close"].pct_change().fillna(0)
    df["StrategyReturn"] = df["Return"] * df["Signal"].shift(1).fillna(0)

    df["CumulativeMarketReturn"] = (1 + df["Return"]).cumprod()
    df["CumulativeStrategyReturn"] = (1 + df["StrategyReturn"]).cumprod()

    return df


def sharpe_ratio(
    returns: Iterable[float], *, trading_days: int, risk_free_rate: float
) -> float:
    """Calculate the annualised Sharpe ratio for a return series."""

    series = pd.Series(returns)
    excess_return = series - risk_free_rate / trading_days
    std = excess_return.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess_return.mean() / std) * np.sqrt(trading_days)


def plot_cumulative_returns(df: pd.DataFrame) -> None:
    """Plot cumulative returns for the market and the strategy."""

    plt.figure(figsize=(10, 6))
    plt.plot(df["CumulativeMarketReturn"], label="Buy & Hold (Market)")
    plt.plot(df["CumulativeStrategyReturn"], label="Strategy")
    plt.title("Strategy vs. Market Performance")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1000, help="Number of synthetic trading days")
    parser.add_argument("--short-window", type=int, default=50, help="Length of the fast SMA window")
    parser.add_argument("--long-window", type=int, default=200, help="Length of the slow SMA window")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a matplotlib chart of cumulative returns",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the synthetic data generator (use to reproduce results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(short_window=args.short_window, long_window=args.long_window)

    data = generate_synthetic_prices(days=args.days, seed=args.seed)
    result = compute_sma_crossover(
        data,
        short_window=config.short_window,
        long_window=config.long_window,
    )

    ratio = sharpe_ratio(
        result["StrategyReturn"],
        trading_days=config.trading_days,
        risk_free_rate=config.risk_free_rate,
    )
    print(f"Sharpe Ratio: {ratio:.2f}")

    if args.plot:
        plot_cumulative_returns(result)


if __name__ == "__main__":
    main()
