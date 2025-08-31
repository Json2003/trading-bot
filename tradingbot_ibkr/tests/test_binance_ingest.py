import tempfile
from pathlib import Path
import pandas as pd
import json
import os

from tradingbot_ibkr.binance_trade_dump_ingest import read_trade_file, find_files, _file_id, append_ticks


def make_csv(tmpdir, name='trades.csv'):
    p = Path(tmpdir) / name
    df = pd.DataFrame({
        'tradeTime': [1620000000000, 1620000060000],
        'price': ['50000.1','50010.2'],
        'qty': ['0.1','0.2']
    })
    df.to_csv(p, index=False)
    return p


def make_jsonlines(tmpdir, name='trades.jsonl'):
    p = Path(tmpdir) / name
    rows = [
        {'T':1620000120000, 'p':'50020.3', 'q':'0.3'},
        {'T':1620000180000, 'p':'50030.4', 'q':'0.4'}
    ]
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return p


def test_read_csv(tmp_path):
    p = make_csv(tmp_path)
    df = read_trade_file(p)
    assert len(df) == 2
    assert 'ts' in df.columns
    assert df['price'].dtype == float


def test_read_jsonlines(tmp_path):
    p = make_jsonlines(tmp_path)
    df = read_trade_file(p)
    assert len(df) == 2
    assert df['qty'].dtype == float


def test_append_and_state(tmp_path):
    # create ticks and append
    p = make_csv(tmp_path)
    df = read_trade_file(p)
    out = tmp_path / 'out_trades.csv'
    appended = append_ticks(df, out)
    assert appended == 2
    # append same again (dedupe)
    appended2 = append_ticks(df, out)
    # len(ticks) returned is still 2 but file should contain only 2 unique rows
    final = pd.read_csv(out)
    assert len(final) == 2
