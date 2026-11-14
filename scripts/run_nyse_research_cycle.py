#!/usr/bin/env python3
"""One frozen NYSE-only historical paper replay; no broker or order API imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fetch_history(symbol: str, spec: dict, out: Path) -> tuple[pd.DataFrame, dict]:
    start = int(pd.Timestamp(spec["history_start"], tz="UTC").timestamp())
    end = int((pd.Timestamp(spec["completed_through"], tz="UTC") + pd.Timedelta(days=1)).timestamp())
    query = urlencode({"period1": start, "period2": end, "interval": "1d", "events": "div,splits"})
    failures = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{symbol}?{query}"
        for attempt in range(2):
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                with urlopen(req, timeout=30) as response:
                    raw = response.read()
                received_at = datetime.now(timezone.utc).isoformat()
                payload = json.loads(raw)
                result = payload["chart"]["result"][0]
                frame, meta = parse_history(result, symbol, spec)
                raw_path = out / "raw" / f"{symbol}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(raw)
                frame.to_csv(out / f"{symbol}_daily.csv", index=False)
                return frame, {**meta, "source_url": url, "received_at": received_at,
                               "raw_sha256": digest(raw), "rows": len(frame),
                               "first_session": frame.session.iloc[0], "last_session": frame.session.iloc[-1]}
            except Exception as exc:
                failures.append(f"{host} attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt == 0:
                    time.sleep(3)
    raise RuntimeError("; ".join(failures))


def parse_history(result: dict, symbol: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    meta = result["meta"]
    expected = {"symbol": symbol, "exchangeName": spec["required_exchange"],
                "instrumentType": spec["required_instrument_type"], "currency": spec["required_currency"]}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"rejected {symbol}: {key}={meta.get(key)!r}, required {value!r}")
    if result.get("events", {}).get("splits"):
        raise ValueError("split-bearing history requires a separate audited corporate-action adapter")
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({key: quote[key] for key in ("open", "high", "low", "close", "volume")})
    frame["timestamp"] = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    frame["session"] = frame.timestamp.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    frame["dividend"] = 0.0
    for event in result.get("events", {}).get("dividends", {}).values():
        day = datetime.fromtimestamp(event["date"], NY).date().isoformat()
        frame.loc[frame.session == day, "dividend"] += float(event["amount"])
    if frame.session.duplicated().any() or not frame.session.is_monotonic_increasing:
        raise ValueError("duplicate or out-of-order sessions")
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] <= 0).any():
        raise ValueError("missing, nonfinite or nonpositive market observation")
    if ((frame.high < frame[["open", "close"]].max(axis=1)) |
        (frame.low > frame[["open", "close"]].min(axis=1)) | (frame.low > frame.high)).any():
        raise ValueError("inconsistent OHLC prices")
    if len(frame) < 750 or frame.session.iloc[-1] != spec["completed_through"]:
        raise ValueError("insufficient or stale completed-session history")
    return frame, {key: meta[key] for key in expected}


def add_features(frame: pd.DataFrame, strategy, params: dict, atr_period: int) -> pd.DataFrame:
    result = frame.copy()
    result["signal"] = strategy(frame, **params)["signals"].to_numpy()
    prev = frame.close.shift(1)
    tr = pd.concat([frame.high - frame.low, (frame.high - prev).abs(), (frame.low - prev).abs()], axis=1).max(axis=1)
    result["atr"] = tr.rolling(atr_period, min_periods=atr_period).mean()
    return result


def fill_cash(price: float, quantity: int, side: str, cost_bps: float) -> float:
    """Signed cash movement; costs reduce both purchase and sale outcomes."""
    notional = price * quantity
    fee = notional * cost_bps / 10000.0
    return -(notional + fee) if side == "buy" else notional - fee


def independent_count(trades: list[dict]) -> int:
    count, last_exit = 0, ""
    for trade in sorted(trades, key=lambda t: (t["exit_session"], t["entry_session"])):
        if trade["entry_session"] > last_exit:
            count += 1
            last_exit = trade["exit_session"]
    return count


def replay(frames: dict[str, pd.DataFrame], spec: dict, block: dict, *, benchmark: bool = False) -> dict:
    sessions = next(iter(frames.values())).session.tolist()
    if any(f.session.tolist() != sessions for f in frames.values()):
        raise ValueError("NYSE symbol calendars do not align; do not silently discard gaps")
    indices = [i for i, day in enumerate(sessions) if block["start"] <= day <= block["end"]]
    if not indices or indices[0] == 0:
        raise ValueError("block is empty or lacks prior-session warm-up")
    cash, receivables = float(spec["initial_cash"]), 0.0
    peak, latched, latch_at = cash, False, None
    positions, trades, curve = {}, [], []
    cost = spec["round_trip_stress_bps"] / 2

    def close(symbol, index, price, reason):
        nonlocal cash
        pos = positions.pop(symbol)
        proceeds = fill_cash(price, pos["quantity"], "sell", cost)
        cash += proceeds
        trades.append({"symbol": symbol, "entry_session": pos["entry_session"],
                       "exit_session": sessions[index], "entry_price": pos["entry_price"],
                       "exit_price": float(price), "quantity": pos["quantity"], "reason": reason,
                       "execution_costs": pos["entry_cost"] + price * pos["quantity"] * cost / 10000,
                       "dividend_accrual": pos["dividends"],
                       "net_pnl": proceeds - pos["cash_used"] + pos["dividends"]})

    for i in indices:
        exited_today = set()
        for symbol, pos in list(positions.items()):
            current, previous = frames[symbol].iloc[i], frames[symbol].iloc[i - 1]
            dividend = float(current.dividend) * pos["quantity"]
            receivables += dividend
            pos["dividends"] += dividend
            if not benchmark:
                if latched or i - pos["entry_index"] >= spec["max_holding_sessions"] or not previous.signal:
                    close(symbol, i, float(current.open), "drawdown_latch" if latched else "signal_or_time")
                    exited_today.add(symbol)
                elif float(current.open) <= pos["stop"]:
                    close(symbol, i, float(current.open), "gap_stop")
                    exited_today.add(symbol)
        equity_open = cash + receivables + sum(p["quantity"] * float(frames[s].iloc[i].open) for s, p in positions.items())
        for symbol in spec["symbols"]:
            current, previous = frames[symbol].iloc[i], frames[symbol].iloc[i - 1]
            if symbol in positions or symbol in exited_today or (latched and not benchmark):
                continue
            if benchmark and i != indices[0]:
                continue
            if not benchmark and (not previous.signal or not math.isfinite(float(previous.atr))):
                continue
            price = float(current.open)
            budget = min(spec["max_position_notional"], cash / (1 + cost / 10000))
            quantity = math.floor(budget / price)
            if not benchmark:
                stop_distance = float(previous.atr) * spec["stop_atr_multiple"]
                if stop_distance <= 0:
                    continue
                quantity = min(quantity, math.floor(equity_open * spec["risk_fraction"] / stop_distance))
            quantity = min(quantity, math.floor(float(previous.volume) * spec["prior_session_volume_participation"]))
            if quantity <= 0:
                continue
            cash_used = -fill_cash(price, quantity, "buy", cost)
            cash -= cash_used
            positions[symbol] = {"quantity": quantity, "entry_price": price, "entry_session": sessions[i],
                                 "entry_index": i, "stop": price - float(previous.atr) * spec["stop_atr_multiple"],
                                 "cash_used": cash_used, "entry_cost": price * quantity * cost / 10000,
                                 "dividends": 0.0}
        if not benchmark:
            for symbol, pos in list(positions.items()):
                current = frames[symbol].iloc[i]
                if float(current.low) <= pos["stop"]:
                    close(symbol, i, min(float(current.open), pos["stop"]), "intraday_stop")
        if i == indices[-1]:
            for symbol in list(positions):
                close(symbol, i, float(frames[symbol].iloc[i].close), "terminal_close")
        equity = cash + receivables + sum(p["quantity"] * float(frames[s].iloc[i].close) for s, p in positions.items())
        if cash < -1e-8 or equity < 0:
            raise ValueError("cash-only capital constraint violated")
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if not benchmark and not latched and drawdown <= -spec["drawdown_latch_fraction"]:
            latched, latch_at = True, sessions[i]
        curve.append({"session": sessions[i], "equity": equity, "cash": cash,
                      "dividend_receivable": receivables, "drawdown": drawdown, "latched": latched})
    pnl = [t["net_pnl"] for t in trades]
    wins, losses = sum(max(p, 0) for p in pnl), -sum(min(p, 0) for p in pnl)
    returns = pd.Series([spec["initial_cash"]] + [row["equity"] for row in curve]).pct_change().dropna()
    independent = independent_count(trades)
    return {"block": block["name"], "start_session": sessions[indices[0]], "end_session": sessions[indices[-1]],
            "initial_cash": spec["initial_cash"], "ending_equity": curve[-1]["equity"],
            "net_return": curve[-1]["equity"] / spec["initial_cash"] - 1,
            "net_pnl": curve[-1]["equity"] - spec["initial_cash"],
            "max_drawdown": min(row["drawdown"] for row in curve),
            "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else None,
            "profit_factor": wins / losses if losses > 0 else None,
            "trade_count": len(trades), "nonoverlapping_observations": independent,
            "minimum_sample_met": independent >= spec["minimum_independent_observations_per_block"],
            "execution_costs": sum(t["execution_costs"] for t in trades),
            "drawdown_latched_at": latch_at, "trades": trades, "equity_curve": curve}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "research/experiments/nyse_stock_cycle_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/nyse-paper-cycle")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_spec = args.spec.read_bytes()
    spec = json.loads(raw_spec)
    (args.output_dir / "frozen_spec.json").write_bytes(raw_spec)
    strategy_path = ROOT / spec["strategy_file"]
    if digest(strategy_path.read_bytes()) != spec["strategy_sha256"]:
        raise ValueError("frozen strategy source changed")
    module_spec = importlib.util.spec_from_file_location("nyse_frozen_strategy", strategy_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    report = {"experiment_id": spec["experiment_id"], "generated_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": os.environ.get("RESEARCH_SOURCE_COMMIT", "local-verification"),
              "workflow_event_sha": os.environ.get("GITHUB_SHA"),
              "executed_script_sha256": digest(Path(__file__).read_bytes()), "spec_sha256": digest(raw_spec),
              "strategy_sha256": spec["strategy_sha256"], "research_only": True, "crypto_evaluated": False,
              "orders_placed": False, "paper_orders_placed": False, "leverage_enabled": False,
              "promotion_allowed": False, "forward_confirmation": False, "historical_asof_availability_verified": False,
              "data": {}, "errors": {}, "results": [], "benchmarks": [], "limitations": spec["notes"] + [
                  "Historical download receipt is recorded now; historical quote availability and revisions were not reconstructed.",
                  "No NBBO, opening-auction size, broker fills, payment-date dividend ledger or delisted stock panel.",
                  "Nonoverlapping intervals are a sample-size safeguard, not proof of statistical independence."]}
    frames = {}
    for symbol in spec["symbols"]:
        try:
            frame, metadata = fetch_history(symbol, spec, args.output_dir)
            frames[symbol] = add_features(frame, module.generate_signals, spec["strategy_parameters"], spec["atr_period"])
            report["data"][symbol] = metadata
        except Exception as exc:
            report["errors"][symbol] = f"{type(exc).__name__}: {exc}"
    if report["errors"]:
        report["status"] = "blocked_market_data"
    else:
        try:
            for block in spec["blocks"]:
                report["results"].append(replay(frames, spec, block))
                report["benchmarks"].append(replay(frames, spec, block, benchmark=True))
            report["status"] = "historical_diagnostic_complete"
            report["passes_historical_screen"] = all(
                r["net_return"] > 0 and r["minimum_sample_met"] and r["drawdown_latched_at"] is None
                and r["net_return"] > b["net_return"] for r, b in zip(report["results"], report["benchmarks"]))
        except Exception as exc:
            report["status"] = "blocked_validation"
            report["errors"]["replay"] = f"{type(exc).__name__}: {exc}"
    write_json(args.output_dir / "report.json", report)
    summary = {k: v for k, v in report.items() if k not in ("results", "benchmarks")}
    summary["results"] = [{k: v for k, v in row.items() if k not in ("trades", "equity_curve")} for row in report["results"]]
    summary["benchmarks"] = [{k: v for k, v in row.items() if k not in ("trades", "equity_curve")} for row in report["benchmarks"]]
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if report["status"] == "historical_diagnostic_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
