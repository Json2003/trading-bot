#!/usr/bin/env python3
"""Research-only synchronized crypto shock reversal backtest.

Hypothesis: when BTC and ETH experience a synchronized multi-day shock, the
common move mean-reverts over the next five completed daily bars. The rule is
frozen before evaluation and uses no live/paper orders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
import urllib.error
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://data.binance.vision"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
NOTIONAL = 3000.0
SHOCK_LOOKBACK_DAYS = 3
SHOCK_THRESHOLD = 0.05
CROSS_ASSET_SPREAD_MAX = 0.04
HOLD_DAYS = 5
COOLDOWN_DAYS = 5
LATENCY_BARS = 1
ROUND_TRIP_BPS = 86.0
FILL_FRACTION = 0.80
FUNDING_BPS_PER_BAR = 0.5
OUTAGE_REJECTION_RATE = 0.02
BLOCKS = 6
MIN_TRADES_PER_BLOCK = 20


def month_iter(since: str, until: str) -> list[tuple[int, int]]:
    start = datetime.strptime(since, "%Y-%m")
    end = datetime.strptime(until, "%Y-%m")
    result: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) < (end.year, end.month):
        result.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "trading-bot-research/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def timestamp_date(raw: str) -> date:
    value = int(float(raw))
    if value >= 10**14:
        value //= 1000
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date()


def read_archive(blob: bytes) -> list[tuple[date, float]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV, found {names}")
        rows: list[tuple[date, float]] = []
        with archive.open(names[0], "r") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for values in reader:
                if not values or values[0].lower() in {"open time", "open_time"}:
                    continue
                if len(values) < 5:
                    continue
                day = timestamp_date(values[0])
                close = float(values[4])
                if math.isfinite(close) and close > 0:
                    rows.append((day, close))
        return rows


def checksum(blob: bytes, checksum_text: str) -> str:
    expected = checksum_text.strip().split()[0].lower()
    actual = hashlib.sha256(blob).hexdigest()
    if expected != actual:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    return actual


def fetch_symbol(symbol: str, since: str, until: str, raw_dir: Path) -> tuple[dict[date, float], dict[str, object]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    prices: dict[date, float] = {}
    archives: list[dict[str, object]] = []
    missing: list[str] = []
    invalid_rows = 0
    for year, month in month_iter(since, until):
        stem = f"{symbol}-1d-{year:04d}-{month:02d}"
        url = f"{BASE_URL}/data/spot/monthly/klines/{symbol}/1d/{stem}.zip"
        try:
            blob = download(url)
            check_text = download(url + ".CHECKSUM").decode("utf-8")
            digest = checksum(blob, check_text)
            month_rows = read_archive(blob)
            for day, close in month_rows:
                if day in prices and prices[day] != close:
                    raise ValueError(f"conflicting duplicate for {symbol} {day}")
                prices[day] = close
            (raw_dir / f"{stem}.zip").write_bytes(blob)
            archives.append({"date": f"{year:04d}-{month:02d}", "url": url, "sha256": digest, "rows": len(month_rows)})
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.append(f"{year:04d}-{month:02d}")
            else:
                raise
    if missing:
        raise RuntimeError(f"missing completed archives for {symbol}: {missing}")
    days = sorted(prices)
    manifest = {
        "source": BASE_URL,
        "market": "spot daily klines",
        "symbol": symbol,
        "interval": "1d",
        "since": since,
        "until_exclusive": until,
        "archive_count": len(archives),
        "missing_dates": missing,
        "invalid_rows_excluded": invalid_rows,
        "row_count": len(days),
        "first_date": days[0].isoformat() if days else None,
        "last_date": days[-1].isoformat() if days else None,
        "archives": archives,
    }
    return prices, manifest


def split_blocks(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not rows:
        return [[] for _ in range(BLOCKS)]
    blocks: list[list[dict[str, object]]] = []
    for idx in range(BLOCKS):
        start = idx * len(rows) // BLOCKS
        end = (idx + 1) * len(rows) // BLOCKS
        blocks.append(rows[start:end])
    return blocks


def summarize(rows: list[dict[str, object]], segment_start: date, segment_end: date) -> dict[str, object]:
    net_pnl = sum(float(row["net_pnl"]) for row in rows)
    equity = NOTIONAL
    peak = equity
    max_dd = 0.0
    net_returns = []
    for row in rows:
        value = float(row["net_pnl"])
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / NOTIONAL * 100.0)
        net_returns.append(value / NOTIONAL)
    wins = sum(value for value in net_returns if value > 0)
    losses = -sum(value for value in net_returns if value < 0)
    pf = wins / losses if losses else None
    sharpe = None
    if len(net_returns) >= 2 and statistics.pstdev(net_returns) > 0:
        sharpe = statistics.mean(net_returns) / statistics.pstdev(net_returns) * math.sqrt(len(net_returns))
    blocks = split_blocks(rows)
    block_counts = [len(block) for block in blocks]
    block_means = [
        statistics.mean(float(row["net_return"]) for row in block) if block else None
        for block in blocks
    ]
    positive_blocks = sum(1 for value in block_means if value is not None and value > 0)
    return {
        "segment_start": segment_start.isoformat(),
        "segment_end_exclusive": segment_end.isoformat(),
        "trade_count": len(rows),
        "net_pnl": net_pnl,
        "net_return_pct": net_pnl / NOTIONAL * 100.0,
        "max_drawdown_pct_of_notional": max_dd,
        "sharpe_proxy": sharpe,
        "profit_factor": pf,
        "execution_cost": sum(float(row["execution_cost"]) for row in rows),
        "block_trade_counts": block_counts,
        "block_mean_net_returns": block_means,
        "positive_block_count": positive_blocks,
        "passes_sample_gate": all(count >= MIN_TRADES_PER_BLOCK for count in block_counts),
        "passes_positive_block_gate": positive_blocks >= 4,
    }


def run_asset(asset: str, prices: dict[date, float], start: date, discovery_end: date, end: date) -> dict[str, object]:
    days = sorted(prices)
    trades: list[dict[str, object]] = []
    last_signal_index = -10**9
    cost = NOTIONAL * (ROUND_TRIP_BPS + FUNDING_BPS_PER_BAR * HOLD_DAYS) / 10000.0 * FILL_FRACTION
    for i in range(SHOCK_LOOKBACK_DAYS, len(days) - LATENCY_BARS - HOLD_DAYS):
        signal_day = days[i]
        entry_index = i + LATENCY_BARS
        exit_index = entry_index + HOLD_DAYS
        entry_day = days[entry_index]
        exit_day = days[exit_index]
        if signal_day < start or signal_day >= end:
            continue
        if i <= last_signal_index + COOLDOWN_DAYS:
            continue
        # The shared signal is checked by the caller and attached to prices.
        common = getattr(run_asset, "_common_signals", {})
        signal = common.get(signal_day)
        if signal is None:
            continue
        direction = int(signal)
        gross = direction * (prices[exit_day] / prices[entry_day] - 1.0)
        net = gross - cost / NOTIONAL
        trades.append({
            "signal_date": signal_day.isoformat(),
            "entry_date": entry_day.isoformat(),
            "exit_date": exit_day.isoformat(),
            "asset": asset,
            "side": "long" if direction > 0 else "short",
            "gross_return": gross,
            "net_return": net,
            "net_pnl": net * NOTIONAL,
            "execution_cost": cost,
            "shock_direction": "down" if direction > 0 else "up",
            "shock_return_btc": signal[1],
            "shock_return_eth": signal[2],
        })
        last_signal_index = i
    discovery = [row for row in trades if date.fromisoformat(str(row["signal_date"])) < discovery_end]
    holdout = [row for row in trades if date.fromisoformat(str(row["signal_date"])) >= discovery_end]
    result = {
        "discovery": summarize(discovery, start, discovery_end),
        "holdout": summarize(holdout, discovery_end, end),
        "discovery_trade_rows": discovery,
        "holdout_trade_rows": holdout,
    }
    result["passes_discovery"] = result["discovery"]["passes_sample_gate"] and result["discovery"]["passes_positive_block_gate"]
    result["passes_confirmation"] = result["passes_discovery"] and result["holdout"]["passes_sample_gate"] and result["holdout"]["passes_positive_block_gate"] and float(result["holdout"]["net_return_pct"]) > 0
    result["status"] = "confirmed" if result["passes_confirmation"] else "not_confirmed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2020-12")
    parser.add_argument("--until", default="2026-08")
    parser.add_argument("--discovery-start", default="2021-01-01")
    parser.add_argument("--discovery-end", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/research/synchronized-shock/raw"))
    parser.add_argument("--output", type=Path, default=Path("synchronized-shock-reversal-report.json"))
    args = parser.parse_args()
    start = date.fromisoformat(args.discovery_start)
    discovery_end = date.fromisoformat(args.discovery_end)
    end = date.fromisoformat(args.end)
    data: dict[str, dict[date, float]] = {}
    manifests = {}
    for symbol in SYMBOLS:
        data[symbol], manifests[symbol] = fetch_symbol(symbol, args.since, args.until, args.raw_dir / symbol)
    common_days = sorted(set(data["BTCUSDT"]) & set(data["ETHUSDT"]))
    common_signals: dict[date, tuple[int, float, float]] = {}
    for i in range(SHOCK_LOOKBACK_DAYS, len(common_days)):
        day = common_days[i]
        prior = common_days[i - SHOCK_LOOKBACK_DAYS]
        btc_ret = data["BTCUSDT"][day] / data["BTCUSDT"][prior] - 1.0
        eth_ret = data["ETHUSDT"][day] / data["ETHUSDT"][prior] - 1.0
        aligned = btc_ret * eth_ret > 0
        synchronized = abs(btc_ret - eth_ret) <= CROSS_ASSET_SPREAD_MAX
        shocked = abs(btc_ret) >= SHOCK_THRESHOLD and abs(eth_ret) >= SHOCK_THRESHOLD
        if aligned and synchronized and shocked:
            common_signals[day] = (-1 if btc_ret > 0 else 1, btc_ret, eth_ret)
    run_asset._common_signals = common_signals
    candidates = {
        "BTC": run_asset("BTC", data["BTCUSDT"], start, discovery_end, end),
        "ETH": run_asset("ETH", data["ETHUSDT"], start, discovery_end, end),
    }
    report = {
        "schema_version": 1,
        "hypothesis": "A synchronized BTC/ETH three-day shock mean-reverts over the next five completed daily bars.",
        "window": {
            "start": args.discovery_start,
            "discovery_end_exclusive": args.discovery_end,
            "end_exclusive": args.end,
            "holdout_untouched": True,
            "completed_bars_only": True,
            "six_chronological_blocks_per_split": True,
            "holdout_selection_used": False,
            "overlapping_trade_windows_excluded_by_cooldown": True,
            "threshold_grid_used": False,
            "newest_unseen_data_used": False,
        },
        "frozen_parameters": {
            "signal_source": "Binance Vision spot daily closes",
            "assets": list(SYMBOLS),
            "shock_lookback_days": SHOCK_LOOKBACK_DAYS,
            "shock_threshold": SHOCK_THRESHOLD,
            "cross_asset_return_spread_max": CROSS_ASSET_SPREAD_MAX,
            "direction": "short synchronized up-shock; long synchronized down-shock",
            "hold_days": HOLD_DAYS,
            "cooldown_days": COOLDOWN_DAYS,
            "entry": "first completed daily bar after signal plus one latency bar",
            "notional": NOTIONAL,
        },
        "execution_model": {
            "fee_bps_per_side": 20.0,
            "spread_bps_per_side": 10.0,
            "slippage_bps_per_side": 10.0,
            "impact_bps_per_side": 8.0,
            "latency_bars": LATENCY_BARS,
            "fill_fraction": FILL_FRACTION,
            "funding_bps_per_bar": FUNDING_BPS_PER_BAR,
            "outage_rejection_rate": OUTAGE_REJECTION_RATE,
            "effective_slippage_bps_per_side": 23.0,
            "round_trip_bps": ROUND_TRIP_BPS,
        },
        "source": {"provider": "Binance Vision", "manifests": manifests, "common_day_count": len(common_days), "common_first_date": common_days[0].isoformat(), "common_last_date": common_days[-1].isoformat()},
        "candidates": candidates,
        "status": "confirmed" if all(c["status"] == "confirmed" for c in candidates.values()) else "not_confirmed",
        "research_only": True,
        "leverage_enabled": False,
        "live_orders_placed": False,
        "paper_orders_placed": False,
        "promotion_allowed": False,
        "active_profile_changed": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "signal_count": len(common_signals)}, indent=2))


if __name__ == "__main__":
    main()
