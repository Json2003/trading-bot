#!/usr/bin/env python3
"""Observe public Binance futures trade flow without placing orders.

The observer records raw normalized market events and completed one-minute
summaries. It contains no broker, account, order, leverage, or risk-control
path. Use the resulting files as research data only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

UTC = timezone.utc
WS_BASE = "wss://fstream.binance.com/market/stream"
SUMMARY_FIELDS = (
    "bucket",
    "symbol",
    "completed",
    "trade_count",
    "buy_trade_count",
    "sell_trade_count",
    "buy_notional",
    "sell_notional",
    "net_aggressive_notional",
    "large_buy_notional",
    "large_sell_notional",
    "max_trade_notional",
    "liquidation_count",
    "liquidation_buy_notional",
    "liquidation_sell_notional",
    "last_trade_price",
    "best_bid",
    "best_ask",
    "spread_bps",
    "book_imbalance",
)


def _finite_number(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _event_time(data: dict[str, Any]) -> int:
    value = data.get("T", data.get("E"))
    if value is None:
        raise ValueError("event has no transaction or event time")
    return int(value)


def _trade_event(data: dict[str, Any]) -> dict[str, Any]:
    symbol = str(data["s"]).upper()
    price = _finite_number(data["p"], "price")
    quantity = _finite_number(data["q"], "quantity")
    if price <= 0 or quantity <= 0:
        raise ValueError("trade price and quantity must be positive")
    # Binance m=true means the buyer was the market maker, so the aggressor
    # was a seller. m=false means the buyer was the taker/aggressor.
    aggressor_side = "SELL" if bool(data.get("m")) else "BUY"
    return {
        "kind": "aggTrade",
        "event_time_ms": _event_time(data),
        "symbol": symbol,
        "price": price,
        "quantity": quantity,
        "notional": price * quantity,
        "aggressor_side": aggressor_side,
        "aggregate_trade_id": data.get("a"),
    }


def _book_event(data: dict[str, Any]) -> dict[str, Any]:
    symbol = str(data["s"]).upper()
    bid = _finite_number(data["b"], "best bid")
    ask = _finite_number(data["a"], "best ask")
    bid_quantity = _finite_number(data["B"], "bid quantity")
    ask_quantity = _finite_number(data["A"], "ask quantity")
    if bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError("invalid best bid/ask")
    return {
        "kind": "bookTicker",
        "event_time_ms": _event_time(data),
        "symbol": symbol,
        "best_bid": bid,
        "best_ask": ask,
        "bid_quantity": bid_quantity,
        "ask_quantity": ask_quantity,
    }


def _liquidation_event(data: dict[str, Any]) -> dict[str, Any]:
    order = data.get("o", data)
    symbol = str(order["s"]).upper()
    side = str(order["S"]).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("liquidation side must be BUY or SELL")
    price = _finite_number(order.get("ap", order.get("p")), "liquidation price")
    quantity = _finite_number(order.get("z", order.get("q")), "liquidation quantity")
    if price <= 0 or quantity <= 0:
        raise ValueError("liquidation price and quantity must be positive")
    return {
        "kind": "forceOrder",
        "event_time_ms": _event_time(order),
        "symbol": symbol,
        "liquidation_side": side,
        "price": price,
        "quantity": quantity,
        "notional": price * quantity,
    }


def parse_message(raw: str) -> list[dict[str, Any]]:
    """Parse a combined-stream payload into normalized observer events."""
    payload = json.loads(raw)
    data = payload.get("data", payload)
    if isinstance(data, list):
        items = data
    else:
        items = [data]
    events: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        event_type = item.get("e")
        try:
            if event_type == "aggTrade":
                events.append(_trade_event(item))
            elif event_type == "bookTicker":
                events.append(_book_event(item))
            elif event_type == "forceOrder":
                events.append(_liquidation_event(item))
        except (KeyError, TypeError, ValueError):
            # A malformed public message is excluded from the derived stream;
            # the raw message remains available for diagnosis.
            continue
    return events


def stream_url(symbols: list[str]) -> str:
    streams = [
        *(f"{symbol.lower()}@aggTrade" for symbol in symbols),
        *(f"{symbol.lower()}@bookTicker" for symbol in symbols),
        "!forceOrder@arr",
    ]
    return WS_BASE + "?streams=" + "/".join(streams)


def _bucket(event_time_ms: int) -> str:
    timestamp = datetime.fromtimestamp(event_time_ms / 1000, tz=UTC)
    return timestamp.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z")


class FlowAggregator:
    """Aggregate events into minute buckets and finalize only closed minutes."""

    def __init__(self, symbols: list[str], large_trade_notional: float) -> None:
        if large_trade_notional <= 0:
            raise ValueError("large_trade_notional must be positive")
        self.symbols = {symbol.upper() for symbol in symbols}
        self.large_trade_notional = large_trade_notional
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}

    def _row(self, bucket: str, symbol: str) -> dict[str, Any]:
        key = (bucket, symbol)
        if key not in self._rows:
            self._rows[key] = {
                "bucket": bucket,
                "symbol": symbol,
                "completed": True,
                "trade_count": 0,
                "buy_trade_count": 0,
                "sell_trade_count": 0,
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "net_aggressive_notional": 0.0,
                "large_buy_notional": 0.0,
                "large_sell_notional": 0.0,
                "max_trade_notional": 0.0,
                "liquidation_count": 0,
                "liquidation_buy_notional": 0.0,
                "liquidation_sell_notional": 0.0,
                "last_trade_price": None,
                "best_bid": None,
                "best_ask": None,
                "spread_bps": None,
                "book_imbalance": None,
            }
        return self._rows[key]

    def ingest(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        symbol = str(event["symbol"]).upper()
        if symbol not in self.symbols:
            return []
        bucket = _bucket(int(event["event_time_ms"]))
        row = self._row(bucket, symbol)
        kind = event["kind"]
        if kind == "aggTrade":
            notional = float(event["notional"])
            row["trade_count"] += 1
            row["max_trade_notional"] = max(row["max_trade_notional"], notional)
            row["last_trade_price"] = float(event["price"])
            if event["aggressor_side"] == "BUY":
                row["buy_trade_count"] += 1
                row["buy_notional"] += notional
                row["net_aggressive_notional"] += notional
                if notional >= self.large_trade_notional:
                    row["large_buy_notional"] += notional
            else:
                row["sell_trade_count"] += 1
                row["sell_notional"] += notional
                row["net_aggressive_notional"] -= notional
                if notional >= self.large_trade_notional:
                    row["large_sell_notional"] += notional
        elif kind == "forceOrder":
            notional = float(event["notional"])
            row["liquidation_count"] += 1
            key = (
                "liquidation_buy_notional"
                if event["liquidation_side"] == "BUY"
                else "liquidation_sell_notional"
            )
            row[key] += notional
        elif kind == "bookTicker":
            bid_quantity = float(event["bid_quantity"])
            ask_quantity = float(event["ask_quantity"])
            row["best_bid"] = float(event["best_bid"])
            row["best_ask"] = float(event["best_ask"])
            midpoint = (row["best_bid"] + row["best_ask"]) / 2.0
            row["spread_bps"] = (
                (row["best_ask"] - row["best_bid"]) / midpoint * 10_000.0
            )
            denominator = bid_quantity + ask_quantity
            row["book_imbalance"] = (
                (bid_quantity - ask_quantity) / denominator
                if denominator > 0
                else None
            )

        current_bucket = bucket
        completed: list[dict[str, Any]] = []
        for key in sorted(self._rows):
            if key[0] < current_bucket:
                completed.append(self._rows.pop(key))
        return completed

    def finalize(self) -> list[dict[str, Any]]:
        rows = [self._rows[key] for key in sorted(self._rows)]
        for row in rows:
            row["completed"] = False
        self._rows.clear()
        return rows


def _write_summary(handle: TextIO, row: dict[str, Any]) -> None:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    if handle.tell() == 0:
        writer.writeheader()
    writer.writerow(row)
    handle.flush()


def run_monitor(
    symbols: list[str],
    output_dir: Path,
    duration_seconds: float,
    large_trade_notional: float,
    reconnect_attempts: int,
) -> dict[str, Any]:
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_events.jsonl"
    summary_path = output_dir / "completed_minute_flow.csv"
    aggregator = FlowAggregator(symbols, large_trade_notional)
    errors: list[str] = []
    event_count = 0
    summary_count = 0
    connected = False
    started_at = datetime.now(UTC)

    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "websocket-client is required; install requirements_research.txt"
        ) from exc

    with raw_path.open("a", encoding="utf-8") as raw_handle, summary_path.open(
        "a", newline="", encoding="utf-8"
    ) as summary_handle:
        def on_message(ws: Any, message: str) -> None:
            nonlocal event_count, summary_count, connected
            connected = True
            received_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            try:
                raw_payload = json.loads(message)
                events = parse_message(message)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"parse error: {exc}")
                raw_payload = {"_raw_message": str(message)}
                events = []
            raw_handle.write(
                json.dumps(
                    {
                        "received_at": received_at,
                        "message": raw_payload,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            raw_handle.flush()
            for event in events:
                event["received_at"] = received_at
                event_count += 1
                for completed in aggregator.ingest(event):
                    _write_summary(summary_handle, completed)
                    summary_count += 1

        def on_error(ws: Any, error: Any) -> None:
            errors.append(str(error))

        def on_close(ws: Any, status: Any, message: Any) -> None:
            return None

        deadline = (
            time.monotonic() + duration_seconds
            if duration_seconds > 0
            else None
        )
        for attempt in range(max(0, reconnect_attempts) + 1):
            if deadline is not None and time.monotonic() >= deadline:
                break
            socket = websocket.WebSocketApp(
                stream_url(symbols),
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            timeout = max(1.0, deadline - time.monotonic()) if deadline else None
            timer = None
            if timeout is not None:
                timer = __import__("threading").Timer(timeout, socket.close)
                timer.daemon = True
                timer.start()
            socket.run_forever(ping_interval=20, ping_timeout=10)
            if timer is not None:
                timer.cancel()
            if deadline is None:
                if attempt >= max(0, reconnect_attempts):
                    break
                time.sleep(2.0)
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))

        for partial in aggregator.finalize():
            _write_summary(summary_handle, partial)
            summary_count += 1

    finished_at = datetime.now(UTC)
    manifest = {
        "research_only": True,
        "paper_orders_placed": False,
        "live_orders_placed": False,
        "leverage_enabled": False,
        "active_profile_changed": False,
        "promotion_allowed": False,
        "source": "Binance USD-M public WebSocket market streams",
        "stream_url": stream_url(symbols).split("?")[0],
        "streams": [
            "symbol@aggTrade",
            "symbol@bookTicker",
            "!forceOrder@arr",
        ],
        "symbols": [symbol.upper() for symbol in symbols],
        "large_trade_notional": large_trade_notional,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "connected": connected,
        "normalized_event_count": event_count,
        "summary_row_count": summary_count,
        "errors": errors[-20:],
        "completed_minute_rows_only": True,
        "final_partial_minute_marked_completed_false": True,
    }
    (output_dir / "monitor_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/observations/binance_trade_flow"),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="0 runs until interrupted; positive values stop automatically",
    )
    parser.add_argument(
        "--large-trade-notional",
        type=float,
        default=100_000.0,
        help="fixed USD notional used only for large-trade tagging",
    )
    parser.add_argument("--reconnect-attempts", type=int, default=3)
    args = parser.parse_args()
    manifest = run_monitor(
        args.symbols,
        args.output_dir,
        args.duration_seconds,
        args.large_trade_notional,
        args.reconnect_attempts,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
