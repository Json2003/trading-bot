"""Fetch economic series and ALFRED vintages from FRED and save locally.

Usage: set environment variable FRED_API_KEY or pass --api-key. The script will fetch a small set
of high-value series (CPI, Unemployment rate, GDP) as a starter and write CSVs to
`tradingbot_ibkr/datafiles/econ/`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import requests
import pandas as pd
from typing import Optional
import json
from datetime import datetime, timezone
import logging
import time
import random
from logging.handlers import RotatingFileHandler

# module logger (configured at runtime via configure_logger)
logger = logging.getLogger('economic_fetch')


def configure_logger(level: str = 'INFO', log_file: Optional[str] = None):
    """Configure the module logger. Call from main to honor CLI flags."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)
    # clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        try:
            logpath = Path(log_file)
            logpath.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(str(logpath), maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            logger.exception('Failed to create log file handler; continuing with console logger')
# load .env if present (development convenience)
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    _env_path = _Path(__file__).resolve().parents[0] / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except Exception:
    # dotenv is optional; requirements include python-dotenv so this should work in dev
    pass

FRED_BASE = "https://api.stlouisfed.org/fred"
ALFRED_BASE = "https://api.stlouisfed.org/alfred"

DEFAULT_SERIES = [
    "CPIAUCSL",  # CPI: All Urban Consumers, U.S. city average
    "UNRATE",   # Unemployment rate
    "GDP",      # Gross Domestic Product
]


def _get(url: str, params: dict, retries: int = 3, backoff_base: float = 2.0, timeout: int = 30, jitter_max: float = 0.0) -> dict:
    """HTTP GET with simple retry/backoff and limited timeout.

    Raises requests.HTTPError on final failure.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(url, params=params, timeout=timeout)
            # handle rate limiting explicitly
            if r.status_code == 429:
                # try to respect Retry-After header
                ra = r.headers.get('Retry-After')
                base_wait = int(ra) if ra and ra.isdigit() else backoff_base ** attempt
                jitter = random.uniform(0, jitter_max) if jitter_max and jitter_max > 0 else 0.0
                wait = base_wait + jitter
                logger.warning('Rate limited on %s; sleeping %.1f seconds (includes jitter %.2f)', url, wait, jitter)
                time.sleep(wait)
                raise requests.HTTPError('429')
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt >= retries:
                logger.exception('Failed HTTP GET %s after %d attempts', url, attempt)
                raise
            base_wait = backoff_base ** attempt
            jitter = random.uniform(0, jitter_max) if jitter_max and jitter_max > 0 else 0.0
            wait = base_wait + jitter
            logger.warning('HTTP GET failed (attempt %d/%d): %s; retrying in %.1fs (jitter %.2f)', attempt, retries, e, wait, jitter)
            time.sleep(wait)


def fetch_series(api_key: str, series_id: str, out_dir: Path):
    # Fetch series observations
    url = f"{FRED_BASE}/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    j = _get(url, params)
    obs = j.get("observations", [])
    df = pd.DataFrame(obs)
    if not df.empty:
        # normalize
        df = df.rename(columns={"date": "date", "value": "value"})
        df = df[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.to_csv(out_dir / f"{series_id}_fred.csv", index=False)
        print(f"Wrote {series_id}_fred.csv")


def fetch_alfred_vintages(api_key: str, series_id: str, out_dir: Path):
    # ALFRED vintages endpoint: series/observations?realtime_start/realtime_end
    url = f"{ALFRED_BASE}/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    j = _get(url, params)
    obs = j.get("observations", [])
    df = pd.DataFrame(obs)
    if not df.empty:
        # ALFRED returns realtime_start and realtime_end columns
        df.to_csv(out_dir / f"{series_id}_alfred_vintages.csv", index=False)
        print(f"Wrote {series_id}_alfred_vintages.csv")


def main(api_key: Optional[str] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", help="FRED API key (overrides FRED_API_KEY env)")
    p.add_argument("--out-dir", default="tradingbot_ibkr/datafiles/econ", help="output dir")
    p.add_argument("--series", nargs="*", help="series ids to fetch (default set)")
    p.add_argument('--retries', type=int, default=3, help='HTTP retries per request')
    p.add_argument('--backoff-base', type=float, default=2.0, help='exponential backoff base')
    p.add_argument('--timeout', type=int, default=30, help='HTTP timeout seconds')
    p.add_argument('--jitter-max', type=float, default=0.0, help='max jitter seconds to add to backoff')
    p.add_argument('--cooldown-minutes', type=float, default=0.0, help='if >0, sleep this many minutes when a series fails repeatedly')
    p.add_argument('--log-level', type=str, default='INFO', help='logging level (DEBUG/INFO/WARNING/ERROR)')
    p.add_argument('--log-file', type=str, default=None, help='optional path for rotating log file')
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not found. Please export it or pass --api-key. Exiting.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_list = args.series or DEFAULT_SERIES
    # configure logging
    configure_logger(level=args.log_level, log_file=args.log_file)

    for s in series_list:
        try:
            fetch_series(api_key, s, out_dir)
            fetch_alfred_vintages(api_key, s, out_dir)
        except Exception as e:
            logger.exception('Failed to fetch %s: %s', s, e)
            if args.cooldown_minutes and args.cooldown_minutes > 0:
                # persist cooldown metadata and exit non-blocking with a small non-zero code
                try:
                    meta = {
                        'failed_series': s,
                        'failed_at': datetime.now(timezone.utc).isoformat(),
                        'cooldown_minutes': args.cooldown_minutes,
                    }
                    cd_path = Path(args.out_dir) / 'cooldown.json'
                    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
                    cd_path.write_text(json.dumps(meta))
                    logger.info('Wrote cooldown metadata to %s and exiting', cd_path)
                except Exception:
                    logger.exception('Failed to write cooldown metadata')
                # exit non-blocking: raise SystemExit with non-zero code
                raise SystemExit(2)
            # continue to next series after cooldown


if __name__ == "__main__":
    main()
