"""Quick fetch helper: fetch one FRED series and write CSV to datafiles/econ.
Usage: python run_fetch_one.py <API_KEY> [SERIES]
"""
import sys
import requests
from pathlib import Path
import pandas as pd

def fetch_one(api_key, series='CPIAUCSL'):
    out_dir = Path(__file__).resolve().parents[0] / 'datafiles' / 'econ'
    out_dir.mkdir(parents=True, exist_ok=True)
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {'series_id': series, 'api_key': api_key, 'file_type': 'json'}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    obs = j.get('observations', [])
    df = pd.DataFrame(obs)
    if not df.empty:
        df = df.rename(columns={'date': 'date', 'value': 'value'})
        df = df[['date','value']]
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        out = out_dir / f"{series}_fred_quick.csv"
        df.to_csv(out, index=False)
        print(out.name)
    else:
        print('no_data')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python run_fetch_one.py <API_KEY> [SERIES]')
        sys.exit(1)
    key = sys.argv[1]
    series = sys.argv[2] if len(sys.argv) > 2 else 'CPIAUCSL'
    try:
        fetch_one(key, series)
    except Exception as e:
        print('error', str(e))
        raise
