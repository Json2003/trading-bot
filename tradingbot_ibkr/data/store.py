"""Simple data persister for bars and trades using CSV files.

This is intentionally small and easy to inspect. For production use, replace
with a proper time-series DB or parquet files.
"""
from pathlib import Path
import tempfile
import pandas as pd
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datafiles'
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_STORE = ROOT / 'model_store'
MODEL_STORE.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT = MODEL_STORE / 'online_model.pkl'
MODEL_VERSION_FILE = MODEL_STORE / 'model_version.json'

def append_bars(symbol: str, df: pd.DataFrame):
    path = DATA_DIR / f'{symbol.replace("/", "_")}_bars.csv'
    if path.exists():
        df.to_csv(path, mode='a', header=False)
    else:
        df.to_csv(path)

def load_bars(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f'{symbol.replace("/", "_")}_bars.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['ts'], index_col='ts')

def append_trades(symbol: str, trades_df: pd.DataFrame):
    path = DATA_DIR / f'{symbol.replace("/", "_")}_trades.csv'
    if path.exists():
        trades_df.to_csv(path, mode='a', header=False)
    else:
        trades_df.to_csv(path)

def load_trades(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f'{symbol.replace("/", "_")}_trades.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['time'], index_col='time')


def append_trade_record(trade: dict):
    """Append a single trade record (dict) to DATA_DIR/trades.csv. Creates file if missing."""
    path = DATA_DIR / 'trades.csv'
    # enrich trade with model_version/hash if missing
    if 'model_version' not in trade or not trade.get('model_version'):
        mv = get_or_update_model_version()
        trade['model_version'] = mv
    if 'model_hash' not in trade or not trade.get('model_hash'):
        mh = get_model_hash() or ''
        trade['model_hash'] = mh

    # store raw order/fills JSON separately to keep trades.csv compact
    raw_dir = DATA_DIR / 'order_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    if 'trade_id' in trade:
        tid = trade['trade_id']
    else:
        tid = f"auto-{pd.Timestamp.now().strftime('%Y%m%dT%H%M%S%f')}"
        trade['trade_id'] = tid

    if 'order_raw' in trade:
        try:
            raw_path = raw_dir / f"{tid}_order.json"
            raw_path.write_text(json.dumps(trade['order_raw']))
            trade['order_raw_file'] = str(raw_path.name)
            del trade['order_raw']
        except Exception:
            trade['order_raw_file'] = ''
    if 'fills' in trade:
        try:
            fills_path = raw_dir / f"{tid}_fills.json"
            fills_path.write_text(json.dumps(trade['fills']))
            trade['fills_file'] = str(fills_path.name)
            del trade['fills']
        except Exception:
            trade['fills_file'] = ''

    df = pd.DataFrame.from_records([trade])
    # ensure timestamps are timezone-aware ISO strings in UTC
    for c in ['entry_ts', 'exit_ts']:
        if c in df.columns:
            ts = pd.to_datetime(df[c])
            # convert to UTC and ISO
            ts = ts.dt.tz_convert('UTC') if ts.dt.tz is not None else ts.dt.tz_localize('UTC')
            df[c] = ts.dt.strftime('%Y-%m-%dT%H:%M:%S%z')

    # atomic write: write to temp file then append/rename
    with tempfile.NamedTemporaryFile('w', delete=False, newline='', suffix='.csv') as tmp:
        df.to_csv(tmp.name, index=False, header=not path.exists())
        tmp_path = Path(tmp.name)

    # append or move
    if path.exists():
        # append without header
        tmp_df = pd.read_csv(tmp_path)
        tmp_df.to_csv(path, mode='a', header=False, index=False)
        tmp_path.unlink()
    else:
        tmp_path.replace(path)


def get_model_hash() -> str:
    """Return SHA256 hex of model artifact if exists, else empty string."""
    if not MODEL_ARTIFACT.exists():
        return ""
    h = hashlib.sha256()
    with open(MODEL_ARTIFACT, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_or_update_model_version() -> str:
    """Return a model version string. If artifact hash changed, increment version counter.

    Version file format: {"last_version": N, "last_hash": "hex"}
    Returns: 'v{N}-{shorthash}' or 'none-0' if no artifact.
    """
    mh = get_model_hash()
    if not mh:
        return 'none-0'

    data = {"last_version": 0, "last_hash": ""}
    if MODEL_VERSION_FILE.exists():
        try:
            data = json.loads(MODEL_VERSION_FILE.read_text())
        except Exception:
            data = {"last_version": 0, "last_hash": ""}

    if data.get('last_hash') == mh:
        ver = data.get('last_version', 0)
    else:
        ver = int(data.get('last_version', 0)) + 1
        data['last_version'] = ver
        data['last_hash'] = mh
        MODEL_VERSION_FILE.write_text(json.dumps(data))

    short = mh[:8]
    return f'v{ver}-{short}'
