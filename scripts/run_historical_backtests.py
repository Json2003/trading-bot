#!/usr/bin/env python3
"""Run reproducible multi-horizon paper backtests on historical OHLCV CSVs.

The runner scores the last 1 day, 1 week, 30 days, and 365 days available in
each dataset. Indicators are calculated with the preceding history, while all
positions are forced flat before the scored window. This prevents warm-up
history from leaking PnL into a horizon result.

The primary family is the causal crypto momentum/volatility-regime strategy.
The ADX/ATR trend strategy is included as a diagnostic comparator. Results are
reported as ``insufficient_data`` when a horizon lacks coverage or indicator
warm-up; no synthetic rows are created to fill a missing history.

Examples::

    python scripts/run_historical_backtests.py \
      --data-root data/historical/binance --interval 1h \
      --symbols BTCUSDT ETHUSDT SOLUSDT --output var/backtests/historical.json

For a checked-in CSV with a non-standard filename, use ``--dataset``::

    python scripts/run_historical_backtests.py \
      --dataset BTCUSDT=tradingbot_ibkr/datafiles/BTC_USDT_bars_annotated.csv
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _third_party(name: str):
    original = sys.path.copy()
    try:
        repo_paths = {path for path in original if str(REPO_ROOT) in os.path.abspath(path)}
        site_paths = [
            path
            for path in original
            if "site-packages" in (path or "") or "dist-packages" in (path or "")
        ]
        remaining = [path for path in original if path not in repo_paths and path not in site_paths]
        sys.path[:] = site_paths + remaining
        if name in sys.modules:
            del sys.modules[name]
        return importlib.import_module(name)
    finally:
        sys.path[:] = original


pd = _third_party("pandas")

from backtest.engine import ExecConfig, run_backtest
from backtest.metrics import summarize
from backtest.strategies.regime_momentum import generate_signals as regime_momentum_signals
from backtest.strategies.trend_adx_atr import generate_signals as trend_adx_signals


HORIZONS = {
    "1d": timedelta(days=1),
    "1w": timedelta(days=7),
    "1m": timedelta(days=30),
    "1y": timedelta(days=365),
}
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "momentum_fast_regime100",
        "family": "regime_momentum",
        "kind": "regime_momentum",
        "params": {"fast": 8, "slow": 21, "regime": 100, "slope_bars": 24},
        "warmup_bars": 124,
    },
    {
        "name": "momentum_primary_regime200",
        "family": "regime_momentum",
        "kind": "regime_momentum",
        "params": {"fast": 13, "slow": 34, "regime": 200, "slope_bars": 24},
        "warmup_bars": 224,
    },
    {
        "name": "momentum_slow_regime200",
        "family": "regime_momentum",
        "kind": "regime_momentum",
        "params": {"fast": 21, "slow": 55, "regime": 200, "slope_bars": 24},
        "warmup_bars": 224,
    },
    {
        "name": "trend_adx_atr_diagnostic",
        "family": "trend_adx_atr",
        "kind": "trend_adx_atr",
        "params": {
            "fast": 8,
            "slow": 21,
            "trend_ma": 200,
            "adx_period": 14,
            "slope_window": 50,
            "atr_window": 200,
            "enable_shorts": True,
        },
        "warmup_bars": 250,
    },
]


def _parse_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load and validate one historical OHLCV CSV."""

    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns and "ts" in frame.columns:
        frame = frame.rename(columns={"ts": "timestamp"})
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    timestamp_values = frame["timestamp"]
    if pd.api.types.is_numeric_dtype(timestamp_values):
        numeric = pd.to_numeric(timestamp_values, errors="coerce")
        sample = abs(float(numeric.dropna().iloc[0]))
        unit = "us" if sample >= 1e15 else "ms" if sample >= 1e12 else "s"
        frame["timestamp"] = pd.to_datetime(numeric, unit=unit, utc=True)
    else:
        frame["timestamp"] = pd.to_datetime(timestamp_values, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=REQUIRED_COLUMNS)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError(f"{path} contains no valid OHLCV rows")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{path} contains non-positive prices")
    return frame


def discover_datasets(
    *,
    data_root: Path | None,
    symbols: list[str],
    interval: str,
    explicit: list[str],
) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for item in explicit:
        if "=" not in item:
            raise ValueError(f"--dataset must use SYMBOL=PATH: {item}")
        symbol, raw_path = item.split("=", 1)
        datasets[symbol.upper()] = Path(raw_path)
    if data_root is None:
        return datasets
    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        candidates = [
            data_root / symbol / f"{interval}.csv",
            data_root / symbol / f"{symbol}-{interval}.csv",
            data_root / f"{symbol}-{interval}.csv",
        ]
        candidates.extend(sorted(data_root.rglob(f"{symbol}*{interval}*.csv")))
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                datasets.setdefault(symbol, candidate)
                break
    return datasets


def _bar_seconds(frame: pd.DataFrame) -> float:
    differences = frame["timestamp"].diff().dt.total_seconds().dropna()
    positive = differences[differences > 0]
    if positive.empty:
        return 3600.0
    return float(positive.median())


def _signal_builder(strategy: dict[str, Any]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    params = dict(strategy["params"])
    if strategy["kind"] == "regime_momentum":
        return lambda frame: regime_momentum_signals(frame, **params)
    if strategy["kind"] == "trend_adx_atr":
        return lambda frame: trend_adx_signals(frame, **params)
    raise ValueError(f"unknown strategy kind: {strategy['kind']}")


def _make_exec_config(args: argparse.Namespace) -> ExecConfig:
    return ExecConfig(
        fees_bps=args.fees_bps,
        slip_bps=args.slippage_bps,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        atr_period=args.atr_period,
        risk_per_trade=args.risk_per_trade,
        max_notional_frac=args.max_notional_frac,
        allow_short=True,
        max_bars=args.max_bars,
    )


def _empty_result(
    *,
    symbol: str,
    horizon: str,
    strategy: dict[str, Any],
    status: str,
    reason: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "horizon": horizon,
        "strategy": strategy["name"],
        "family": strategy["family"],
        "status": status,
        "reason": reason,
        "params": strategy["params"],
        **metadata,
    }


def _run_window(
    frame: pd.DataFrame,
    *,
    symbol: str,
    horizon: str,
    duration: timedelta,
    strategy: dict[str, Any],
    exec_config: ExecConfig,
    evaluation_end: pd.Timestamp,
    minimum_coverage: float,
) -> dict[str, Any]:
    bar_seconds = _bar_seconds(frame)
    window_start = evaluation_end - duration
    usable = frame[frame["timestamp"] <= evaluation_end].copy()
    window = usable[(usable["timestamp"] >= window_start) & (usable["timestamp"] <= evaluation_end)]
    warmup_rows = int((usable["timestamp"] < window_start).sum())
    expected_rows = max(1, int(round(duration.total_seconds() / bar_seconds)))
    coverage_ratio = len(window) / expected_rows
    metadata = {
        "window_start": window_start.isoformat(),
        "window_end": evaluation_end.isoformat(),
        "rows": int(len(window)),
        "expected_rows": expected_rows,
        "coverage_ratio": round(float(coverage_ratio), 6),
        "warmup_rows": warmup_rows,
        "bar_seconds": bar_seconds,
    }
    if len(window) < 3 or coverage_ratio < minimum_coverage:
        return _empty_result(
            symbol=symbol,
            horizon=horizon,
            strategy=strategy,
            status="insufficient_data",
            reason="historical window does not have the requested bar coverage",
            metadata=metadata,
        )
    if warmup_rows < int(strategy["warmup_bars"]):
        return _empty_result(
            symbol=symbol,
            horizon=horizon,
            strategy=strategy,
            status="insufficient_data",
            reason=f"only {warmup_rows} warm-up bars; need {strategy['warmup_bars']}",
            metadata=metadata,
        )

    builder = _signal_builder(strategy)
    signal_frame = builder(usable)
    signals = signal_frame["signals"].astype(int).copy()
    signals.loc[usable["timestamp"] < window_start] = 0
    backtest_frame = usable.copy()

    def window_signals(_: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signals": signals.to_numpy()})

    trades, equity, bar_returns = run_backtest(backtest_frame, window_signals, exec_config)
    equity = equity[
        (pd.to_datetime(equity["timestamp"], utc=True) >= window_start)
        & (pd.to_datetime(equity["timestamp"], utc=True) <= evaluation_end)
    ].copy()
    if equity.empty:
        return _empty_result(
            symbol=symbol,
            horizon=horizon,
            strategy=strategy,
            status="no_scored_bars",
            reason="backtest produced no equity rows in the requested window",
            metadata=metadata,
        )
    initial_equity = float(equity["equity"].iloc[0])
    if initial_equity <= 0:
        raise ValueError("backtest produced non-positive starting equity")
    equity["equity"] = equity["equity"] / initial_equity
    returns = equity["equity"].pct_change().fillna(0.0)
    if len(trades):
        exit_ts = pd.to_datetime(trades["exit_ts"], utc=True)
        trades = trades[(exit_ts >= window_start) & (exit_ts <= evaluation_end)].copy()
    periods_per_year = max(1, int(round((365.25 * 24 * 3600) / bar_seconds)))
    summary = summarize(trades, equity, returns, periods_per_year=periods_per_year)
    profit_factor = summary.get("profit_factor")
    if isinstance(profit_factor, float) and math.isinf(profit_factor):
        profit_factor = None
    return {
        "symbol": symbol,
        "horizon": horizon,
        "strategy": strategy["name"],
        "family": strategy["family"],
        "status": "ok",
        "reason": None,
        "params": strategy["params"],
        "total_return": float(summary["total_return"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "sharpe": float(summary["sharpe"]),
        "sortino": float(summary["sortino"]),
        "profit_factor": profit_factor,
        "win_rate": float(summary["win_rate"]),
        "avg_trade": float(summary["avg_trade"]),
        "trades": int(summary["num_trades"]),
        **metadata,
    }


def run_historical_matrix(
    datasets: dict[str, Path],
    *,
    interval: str,
    requested_end: pd.Timestamp | None,
    exec_config: ExecConfig,
    minimum_coverage: float = 0.90,
    primary_only: bool = False,
) -> dict[str, Any]:
    loaded = {symbol: load_ohlcv(path) for symbol, path in datasets.items()}
    if not loaded:
        raise ValueError("no datasets supplied")
    common_latest = min(frame["timestamp"].max() for frame in loaded.values())
    evaluation_end = min(common_latest, requested_end) if requested_end is not None else common_latest
    selected = STRATEGIES[:3] if primary_only else STRATEGIES
    results: list[dict[str, Any]] = []
    for symbol, frame in loaded.items():
        for horizon, duration in HORIZONS.items():
            for strategy in selected:
                results.append(
                    _run_window(
                        frame,
                        symbol=symbol,
                        horizon=horizon,
                        duration=duration,
                        strategy=strategy,
                        exec_config=exec_config,
                        evaluation_end=evaluation_end,
                        minimum_coverage=minimum_coverage,
                    )
                )
    return {
        "metadata": {
            "interval": interval,
            "evaluation_end": evaluation_end.isoformat(),
            "symbols": sorted(loaded),
            "horizons": list(HORIZONS),
            "execution": {
                "fees_bps_per_fill": exec_config.fees_bps,
                "slippage_bps_per_fill": exec_config.slip_bps,
                "tp_atr_mult": exec_config.tp_atr_mult,
                "sl_atr_mult": exec_config.sl_atr_mult,
                "atr_period": exec_config.atr_period,
                "risk_per_trade": exec_config.risk_per_trade,
                "max_notional_frac": exec_config.max_notional_frac,
                "allow_short": exec_config.allow_short,
                "max_bars": exec_config.max_bars,
            },
            "data_policy": "real historical OHLCV only; no synthetic fill rows",
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", action="append", default=[], help="SYMBOL=PATH; repeatable")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--end", type=_parse_timestamp)
    parser.add_argument("--fees-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=8.0)
    parser.add_argument("--tp-atr-mult", type=float, default=3.0)
    parser.add_argument("--sl-atr-mult", type=float, default=1.5)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--risk-per-trade", type=float, default=0.005)
    parser.add_argument("--max-notional-frac", type=float, default=0.90)
    parser.add_argument("--max-bars", type=int, default=24)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = discover_datasets(
        data_root=args.data_root,
        symbols=args.symbols,
        interval=args.interval,
        explicit=args.dataset,
    )
    if not datasets:
        raise SystemExit("no datasets found; use --data-root or --dataset SYMBOL=PATH")
    report = run_historical_matrix(
        datasets,
        interval=args.interval,
        requested_end=args.end,
        exec_config=_make_exec_config(args),
        minimum_coverage=args.minimum_coverage,
        primary_only=args.primary_only,
    )
    payload = json.dumps(report, indent=2, default=str, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"historical backtest report: {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
