"""Simple data persister for bars and trades using CSV files.

This is intentionally small and easy to inspect. For production use, replace
with a proper time-series DB or parquet files.
"""
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import hashlib
import json
import io
import csv

try:  # pragma: no cover - exercised via stub in tests
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback if pandas missing
    pd = None

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


def save_bars_to_gcs(symbol: str, df: pd.DataFrame, bucket_name: str, path_prefix: str = 'data'):
    """Upload bars DataFrame to a GCS bucket."""
    if df.empty:
        return
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:  # pragma: no cover - network dependency
        raise RuntimeError('google-cloud-storage is required for GCS operations') from e
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{path_prefix}/{symbol.replace('/', '_')}_bars.csv"
    tmp_df = df.reset_index()
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp_df.to_csv(tmp.name, index=False)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(tmp.name)
        Path(tmp.name).unlink()


def load_bars_from_gcs(symbol: str, bucket_name: str, path_prefix: str = 'data') -> pd.DataFrame:
    """Load bars DataFrame from a GCS bucket if present, else return empty DataFrame."""
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:  # pragma: no cover - network dependency
        raise RuntimeError('google-cloud-storage is required for GCS operations') from e
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{path_prefix}/{symbol.replace('/', '_')}_bars.csv"
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return pd.DataFrame()
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data), parse_dates=['ts'], index_col='ts')

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
    """Append a single trade record to ``trades.csv`` using only the stdlib."""
    path = DATA_DIR / 'trades.csv'
    # enrich trade with model metadata if missing
    if 'model_version' not in trade or not trade.get('model_version'):
        trade['model_version'] = get_or_update_model_version()
    if 'model_hash' not in trade or not trade.get('model_hash'):
        trade['model_hash'] = get_model_hash() or ''

    raw_dir = DATA_DIR / 'order_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    tid = trade.get('trade_id') or f"auto-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    trade['trade_id'] = tid

    # persist optional raw order/fill data
    if 'order_raw' in trade:
        try:
            raw_path = raw_dir / f"{tid}_order.json"
            raw_path.write_text(json.dumps(trade['order_raw']))
            trade['order_raw_file'] = raw_path.name
            del trade['order_raw']
        except Exception:
            trade['order_raw_file'] = ''
    if 'fills' in trade:
        try:
            fills_path = raw_dir / f"{tid}_fills.json"
            fills_path.write_text(json.dumps(trade['fills']))
            trade['fills_file'] = fills_path.name
            del trade['fills']
        except Exception:
            trade['fills_file'] = ''

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trade.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(trade)


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
