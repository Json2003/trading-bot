#!/usr/bin/env python3
"""Fetch daily equity and crypto data into Parquet files.

The script reads a lightweight YAML/JSON configuration describing the
requested timezone, asset universes, lookback horizon, and output
location.  Equities are sourced from Yahoo Finance via ``yfinance``
while crypto prices come from the public CoinGecko API.  Results are
stored in timezone-aware parquet files grouped by asset class.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests
import yfinance as yf
from zoneinfo import ZoneInfo

CONFIG_DEFAULT = Path("configs/daily_data.yaml")


class ConfigError(RuntimeError):
    """Raised when the configuration file cannot be parsed."""


@dataclass
class FetchConfig:
    timezone: str
    equities: List[str]
    crypto: List[str]
    lookback_days: int
    out_dir: Path

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def _coerce_scalar(value: str) -> Any:
    """Convert a scalar value from the minimal YAML parser."""
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse a tiny subset of YAML used by our configuration file.

    The parser understands top-level keys mapping to scalars or lists of
    scalars.  Comment lines (starting with ``#``) are ignored.  It is not
    a general-purpose YAML parser, but it is sufficient for small
    configuration blobs checked into the repository without requiring an
    additional dependency like PyYAML.
    """

    data: Dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Remove inline comments
        line, *_comment = stripped.split("#", 1)
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("- "):
            if current_key is None:
                raise ConfigError("List item encountered before a key definition")
            item = line[2:].strip()
            data.setdefault(current_key, []).append(_coerce_scalar(item))
            continue
        if ":" not in line:
            raise ConfigError(f"Cannot parse line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = _coerce_scalar(value)
            current_key = None
        else:
            data[key] = []
            current_key = key
    return data


def load_config(path: Path) -> FetchConfig:
    """Load configuration data from YAML or JSON."""

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = path.read_text()
    try:
        parsed: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception:
            parsed = _parse_simple_yaml(raw)
        else:
            parsed = yaml.safe_load(raw)
            if not isinstance(parsed, dict):
                raise ConfigError("Configuration must be a mapping")

    timezone = parsed.get("timezone", "UTC")
    equities = [s for s in parsed.get("equities", []) if s]
    crypto = [s for s in parsed.get("crypto", []) if s]
    lookback = int(parsed.get("lookback_days", 200))
    out_dir = Path(parsed.get("out_dir", "data/daily"))
    return FetchConfig(
        timezone=timezone, equities=equities, crypto=crypto, lookback_days=lookback, out_dir=out_dir
    )


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _localize_index(df: pd.DataFrame, tz: ZoneInfo) -> pd.DataFrame:
    """Return a dataframe with a timezone-aware DatetimeIndex."""

    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.tz_convert(tz)
    df = df.copy()
    df.index = idx
    return df


def fetch_equity_history(symbol: str, lookback_days: int, tz: ZoneInfo) -> pd.DataFrame:
    """Download daily equity data from Yahoo Finance."""

    history = yf.download(
        symbol,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if history.empty:
        return history
    history = history.rename(columns=str.lower)
    history = _localize_index(history, tz)
    history["symbol"] = symbol
    return history


def fetch_crypto_history(asset_id: str, lookback_days: int, tz: ZoneInfo) -> pd.DataFrame:
    """Download daily crypto OHLCV data from CoinGecko."""

    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
    params = {"vs_currency": "usd", "days": lookback_days}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    prices = payload.get("prices", [])
    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(prices, columns=["timestamp", "price"])
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True)
    price_df = price_df.set_index("timestamp")

    ohlc = price_df["price"].resample("1D").agg(["first", "max", "min", "last"])
    ohlc = ohlc.rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"})

    volumes = payload.get("total_volumes") or []
    if volumes:
        vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms", utc=True)
        vol_df = vol_df.set_index("timestamp").resample("1D").sum()
        ohlc = ohlc.join(vol_df, how="left")
    else:
        ohlc["volume"] = pd.NA

    ohlc = ohlc.dropna(how="all")
    ohlc.index = ohlc.index.tz_convert(tz)
    ohlc["symbol"] = asset_id
    return ohlc


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    df.to_parquet(path)


def summarize_assets(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch daily equity and crypto data into Parquet files"
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DEFAULT, help="Path to YAML/JSON configuration file"
    )
    parser.add_argument(
        "--days", type=int, default=None, help="Override lookback days from configuration"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Override output directory from configuration"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.days is not None:
        cfg.lookback_days = args.days
    if args.out is not None:
        cfg.out_dir = args.out

    tz = cfg.tz
    base_dir = ensure_directory(cfg.out_dir)
    equity_dir = ensure_directory(base_dir / "equities")
    crypto_dir = ensure_directory(base_dir / "crypto")

    equity_summary: List[Dict[str, Any]] = []
    for symbol in cfg.equities:
        try:
            history = fetch_equity_history(symbol, cfg.lookback_days, tz)
        except Exception as exc:  # pragma: no cover - network failure logging
            print(f"⚠️  Failed to fetch equity {symbol}: {exc}")
            continue
        if history.empty:
            print(f"ℹ️  No data returned for equity {symbol}")
            continue
        out_path = equity_dir / f"{symbol}.parquet"
        write_parquet(history, out_path)
        equity_summary.append({"symbol": symbol, "rows": int(len(history)), "path": str(out_path)})
        print(f"✅ Wrote {len(history)} equity rows for {symbol} -> {out_path}")

    crypto_summary: List[Dict[str, Any]] = []
    for asset in cfg.crypto:
        try:
            history = fetch_crypto_history(asset, cfg.lookback_days, tz)
        except Exception as exc:  # pragma: no cover - network failure logging
            print(f"⚠️  Failed to fetch crypto {asset}: {exc}")
            continue
        if history.empty:
            print(f"ℹ️  No data returned for crypto {asset}")
            continue
        out_path = crypto_dir / f"{asset}.parquet"
        write_parquet(history, out_path)
        crypto_summary.append({"symbol": asset, "rows": int(len(history)), "path": str(out_path)})
        print(f"✅ Wrote {len(history)} crypto rows for {asset} -> {out_path}")

    manifest = {
        "generated_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "timezone": cfg.timezone,
        "lookback_days": cfg.lookback_days,
        "equities": summarize_assets(equity_summary),
        "crypto": summarize_assets(crypto_summary),
    }
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"📄 Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
