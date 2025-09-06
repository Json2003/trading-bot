import zipfile
from pathlib import Path

import pandas as pd

from tradingbot_ibkr.data.load_zip_csv import load_zipped_csv


def test_load_zipped_csv(tmp_path):
    data = {
        "Open time (ms)": [1, 2],
        "Open": [10.0, 11.0],
        "High": [12.0, 13.0],
        "Low": [8.0, 9.0],
        "Close": [11.0, 12.0],
        "Volume": [100, 200],
        "Close time (ms)": [3, 4],
        "Quote asset volume": [1000, 2000],
        "Number of trades": [1, 2],
        "Taker buy base asset volume": [50, 60],
        "Taker buy quote asset volume": [500, 600],
        "Ignore": [0, 0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="data.csv")

    loaded = load_zipped_csv(zip_path)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)
