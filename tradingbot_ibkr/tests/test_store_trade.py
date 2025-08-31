import os
import tempfile
import pandas as pd
from tradingbot_ibkr.data import store


def test_append_trade_record_creates_file(tmp_path):
    # use a temporary data dir by monkeypatching DATA_DIR
    orig = store.DATA_DIR
    store.DATA_DIR = tmp_path
    trade = {
        'trade_id': 't1',
        'symbol': 'BTC/USDT',
        'entry_ts': '2020-01-01T00:00:00Z',
        'exit_ts': '2020-01-01T00:01:00Z',
        'entry_price': 100.0,
        'exit_price': 101.0,
        'size': 0.1,
        'side': 'buy',
        'fees': 0.0,
        'pnl': 0.1,
        'pnl_pct': 0.01,
        'model_version': 'test-0',
        'entry_reason': 'unit-test',
        'exit_reason': 'unit-test'
    }
    store.append_trade_record(trade)
    f = tmp_path / 'trades.csv'
    assert f.exists()
    df = pd.read_csv(f)
    assert df.loc[0, 'trade_id'] == 't1'
    # restore
    store.DATA_DIR = orig
