import tempfile
import pathlib
import csv
import json
from tradingbot_ibkr import binance_trade_dump_ingest as ingest


def test_read_trade_file_csv_happy_path():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        f = p / 'sample.csv'
        with open(f, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['tradeTime','price','qty','side'])
            writer.writerow([1610000000000, '29000.5', '0.001', 'buy'])
            writer.writerow([1610000001000, '29001.0', '0.002', 'sell'])
        df = ingest.read_trade_file(f)
        # should return a DataFrame-like object with two rows
        assert len(df) == 2


def test_append_ticks_and_dedup():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        out = p / 'out.csv'
        # build a minimal 'DataFrame' via list-of-dicts then to pd.DataFrame
        rows = [
            {'ts': 1, 'price': 10.0, 'qty': 0.1},
            {'ts': 2, 'price': 11.0, 'qty': 0.2},
        ]
        df = ingest.pd.DataFrame(rows)
        appended = ingest.append_ticks(df, out)
        assert appended == 2
        # append duplicate rows
        appended2 = ingest.append_ticks(df, out)
        assert appended2 == 2
        # file should contain two rows
        content = out.read_text()
        assert 'ts' in content
        assert '1' in content
        assert '2' in content
