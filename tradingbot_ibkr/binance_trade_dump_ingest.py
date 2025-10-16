#!/usr/bin/env python3
"""Ingest Binance trade dump files (tick-level) and optionally aggregate to OHLCV.

Usage examples (PowerShell):
  # parse all CSV/JSON files in a directory, save ticks and 1m OHLCV
  python binance_trade_dump_ingest.py --input-dir ./downloads --symbol BTC/USDT --out-dir ./datafiles --to-ohlcv 1m

  # just normalize ticks and append
  python binance_trade_dump_ingest.py --input-dir ./downloads --symbol BTC/USDT

The script supports CSV or JSON-lines where each record is a trade with at least
timestamp, price, and quantity fields. It will attempt to auto-detect common
Binance field names (e.g., 'tradeTime','time','T' for timestamp; 'price','p' for
price; 'qty','q','quantity' for quantity).
"""
import argparse
from pathlib import Path
import csv
import json
import sys
from typing import Any, List, Optional, Callable
import hashlib
import os
import logging
import concurrent.futures
import threading
import importlib

# Prefer site-packages pandas; fall back to package-local stub tradingbot_ibkr._pandas_stub
try:
    import pandas as pd  # type: ignore
except Exception:
    try:
        # Try to import site-packages pandas while avoiding a repo-local shadow on sys.path
        removed: List[str] = []
        while sys.path and (sys.path[0] == "" or Path(sys.path[0]).resolve() == Path.cwd()):
            removed.append(sys.path.pop(0))
        pd = importlib.import_module("pandas")
        # restore sys.path
        while removed:
            sys.path.insert(0, removed.pop())
    except Exception:
        # last resort: load the lightweight package stub module if present
        try:
            pd = importlib.import_module("tradingbot_ibkr._pandas_stub")
        except Exception:
            # create minimal placeholder so code can import the module in tests,
            # real pandas is required for resampling/Parquet paths.
            pd = None  # type: ignore


def find_files(input_dir: Path, pattern: str = "*") -> List[Path]:
    return sorted([p for p in input_dir.rglob(pattern) if p.is_file()])


def _file_id(path: Path) -> str:
    # small, stable id for processed-file tracking
    h = hashlib.sha1()
    h.update(str(path.name).encode("utf-8"))
    h.update(str(path.stat().st_size).encode("utf-8"))
    return h.hexdigest()


def read_trade_file(path: Path) -> Any:
    """Read a trade dump file and return a very small DataFrame.

    Only the behaviour required by the unit tests is implemented: the function
    understands CSV and JSON-lines files that contain timestamp, price and
    quantity fields. The return value is a DataFrame (from the local stub) with
    columns ``ts``, ``price`` and ``qty``.
    """
    rows: List[dict] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    norm: List[dict] = []
    for r in rows:
        ts = r.get("tradeTime") or r.get("time") or r.get("T") or r.get("t") or r.get("timestamp")
        price = r.get("price") or r.get("p")
        qty = r.get("qty") or r.get("q") or r.get("quantity") or r.get("amount")
        if ts is None or price is None or qty is None:
            continue
        norm.append({
            "ts": int(ts),
            "price": float(price),
            "qty": float(qty),
            "side": r.get("side"),
        })

    norm.sort(key=lambda x: x["ts"])
    if pd is None:
        raise ImportError("pandas is required to create DataFrame; could not import pandas")
    return pd.DataFrame(norm)


def append_ticks(ticks, out_path) -> int:
    """
    Append ticks (pandas.DataFrame or list-of-dicts) to CSV at out_path.
    Deduplicates by (ts, price, qty). Returns number of rows newly appended.
    """
    from pathlib import Path
    import pandas as _pd

    out = Path(out_path)
    # normalize to DataFrame
    if not hasattr(ticks, "copy"):
        df_new = _pd.DataFrame(ticks)
    else:
        df_new = ticks.copy()

    # ensure ts/price/qty columns exist
    for c in ("ts", "price", "qty"):
        if c not in df_new.columns:
            df_new[c] = _pd.NA

    # read existing file if present
    if out.exists():
        try:
            df_old = _pd.read_csv(out)
        except Exception:
            df_old = _pd.DataFrame(columns=["ts", "price", "qty"])
    else:
        df_old = _pd.DataFrame(columns=["ts", "price", "qty"])

    # concat + dedupe
    df_combined = _pd.concat([df_old, df_new], ignore_index=True, sort=False)
    # normalize types to enable exact dedupe
    df_combined["ts"] = df_combined["ts"].astype("Int64", errors="ignore")
    df_combined["price"] = _pd.to_numeric(df_combined["price"], errors="coerce")
    df_combined["qty"] = _pd.to_numeric(df_combined["qty"], errors="coerce").fillna(0.0)
    before = len(df_old)
    df_combined = df_combined.drop_duplicates(subset=["ts", "price", "qty"], keep="first")
    appended = max(0, len(df_combined) - before)
    # write atomically
    tmp = out.with_suffix(out.suffix + ".tmp")
    df_combined.to_csv(tmp, index=False)
    try:
        os.replace(tmp, out)
    except Exception:
        try:
            tmp.rename(out)
        except Exception:
            # last resort overwrite
            df_combined.to_csv(out, index=False)
    return int(appended)


# Atomic state write helper (use when updating processed state)
def write_state_atomic(state_path: Path, processed_set: set) -> None:
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(list(processed_set)))
    try:
        os.replace(tmp, state_path)
    except Exception:
        tmp.rename(state_path)


def ticks_to_ohlcv(ticks: Any, timeframe: str = "1m") -> Any:  # pragma: no cover - unused in tests
    """
    Resample tick records (DataFrame-like or list-of-dicts with columns
    ['ts','price','qty']) to OHLCV bars. timeframe examples: '1m', '5m', '1h'.
    Returns a pandas.DataFrame with columns ['ts','open','high','low','close','volume'].
    """
    # Require real pandas APIs
    if not hasattr(pd, "to_datetime") or not hasattr(pd, "DataFrame"):
        raise NotImplementedError("OHLCV resampling requires real pandas")

    # normalize timeframe: '1m' -> '1min' for pandas
    tf = timeframe.lower()
    if tf.endswith("m") and not tf.endswith("mo"):
        tf = tf[:-1] + "min"
    tf = tf.replace("H", "h")

    # Accept list-of-dicts or DataFrame-like
    if pd is None or not hasattr(pd, "DataFrame"):
        raise ImportError("pandas is required to create DataFrame; could not import pandas")
    try:
        if not hasattr(ticks, "copy"):
            df = pd.DataFrame(ticks)
        else:
            df = ticks.copy()
    except Exception:
        df = pd.DataFrame(ticks)

    # Accept alternate timestamp names
    if "ts" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "ts"})
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ts"})
        else:
            raise ValueError("ticks DataFrame must include a 'ts' (timestamp) column")

    # Normalize types
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    if df["ts"].isna().all():
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "price"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    df = df.set_index("ts").sort_index()

    ohlc = df["price"].resample(tf).ohlc()
    vol = df["qty"].resample(tf).sum().rename("volume")
    out = ohlc.join(vol, how="outer").reset_index()

    expected = ["ts", "open", "high", "low", "close", "volume"]
    for c in expected:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[expected]
    return out


def main():
    parser = argparse.ArgumentParser(description="Ingest Binance trade dump files (tick-level) and optionally aggregate to OHLCV")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out-dir", default="datafiles")
    parser.add_argument("--to-ohlcv", default=None, help="timeframe to resample ticks to (e.g. 1m, 1h)")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--state-file", default=".ingest_state.json", help="path to state file to store processed file ids")
    parser.add_argument("--workers", type=int, default=None, help="number of worker threads to use (overrides INGEST_WORKERS env var)")
    parser.add_argument("--force", action="store_true", help="reprocess files even if present in state file")
    parser.add_argument("--log-file", default=None, help="path to logfile (if omitted, logs to stdout)")
    parser.add_argument("--progress", action="store_true", help="show progress bar if tqdm is installed")
    args = parser.parse_args()

    # logging
    logger = logging.getLogger("binance_ingest")
    logger.setLevel(logging.INFO)
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(sh)

    # optional progress bar
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    inp = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    files = find_files(inp, args.pattern)
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        print("No files found in", inp)
        return

    # concurrency and state
    max_workers = args.workers or int(os.getenv("INGEST_WORKERS", "4"))
    state_path = Path(args.state_file)
    try:
        processed = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    except Exception:
        processed = set()
    processed_ids = []

    def ingest_dir(input_dir: Path,
                   out_dir: Path,
                   *,
                   progress_callback: Optional[Callable[[float], None]] = None,
                   max_workers: int = 4,
                   **kwargs) -> None:
        total_bytes = sum(p.stat().st_size for p in files)
        processed_bytes = 0
        processed_lock = threading.Lock()

        def _worker(path: Path):
            nonlocal processed_bytes
            fid = _file_id(path)
            if fid in processed and not args.force:
                logger.info(f"Skipping already processed file {path.name}")
                with processed_lock:
                    processed_bytes += path.stat().st_size
                return None, None
            try:
                t = read_trade_file(path)
                with processed_lock:
                    processed_bytes += path.stat().st_size
                    pct = (processed_bytes / total_bytes) * 100 if total_bytes else 100.0
                logger.info(f"Read {len(t)} ticks from {path.name}  [{pct:.1f}%]")
                # update tqdm if enabled
                if args.progress and tqdm is not None:
                    try:
                        # update by bytes to reflect progress
                        if bar is not None:
                            bar.update(path.stat().st_size)
                    except Exception:
                        pass
                return fid, t
            except Exception as e:
                with processed_lock:
                    processed_bytes += path.stat().st_size
                logger.warning(f"Failed to read {path}: {e}")
                return None, None

        # streaming: process each parsed file as it completes to avoid accumulating all ticks
        out_ticks = out_dir / f"{args.symbol.replace('/', '_')}_trades.csv"
        out_bars = out_dir / f"{args.symbol.replace('/', '_')}_bars.csv" if args.to_ohlcv else None

        bar = None
        try:
            if args.progress and tqdm is not None:
                bar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="ingest")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(_worker, p): p for p in files}
                for fut in concurrent.futures.as_completed(futures):
                    fid, t = fut.result()
                    if fid and t is not None:
                        # append ticks from this file immediately
                        try:
                            appended = append_ticks(t, out_ticks)
                            logger.info(f"Appended {appended} ticks from {futures[fut].name} to {out_ticks}")
                        except Exception as e:
                            logger.warning(f"Failed to append ticks for {futures[fut].name}: {e}")
                        processed_ids.append(fid)

                        # optionally compute per-file OHLCV and merge to bars file immediately
                        if args.to_ohlcv and out_bars is not None and pd is not None:
                            try:
                                ohlcv = ticks_to_ohlcv(t, args.to_ohlcv)
                                if out_bars.exists():
                                    existing = pd.read_csv(out_bars, parse_dates=["ts"], index_col="ts")
                                    existing.index = pd.to_datetime(existing.index, utc=True)
                                    new = ohlcv.set_index("ts")
                                    combined = pd.concat([existing, new])
                                    combined = combined[~combined.index.duplicated(keep="first")]
                                    combined.sort_index(inplace=True)
                                    combined.to_csv(out_bars)
                                else:
                                    ohlcv.to_csv(out_bars, index=False)
                            except Exception as e:
                                logger.warning(f"Failed to process OHLCV for {futures[fut].name}: {e}")
        finally:
            if bar is not None:
                try:
                    bar.close()
                except Exception:
                    pass

    if not processed_ids:
        print("No ticks parsed; exiting")
        sys.exit(0)

    # update state file (record processed ids)
    processed.update(processed_ids)
    try:
        write_state_atomic(state_path, processed)
    except Exception as e:
        logger.warning("Could not write state file atomically: %s", e)
