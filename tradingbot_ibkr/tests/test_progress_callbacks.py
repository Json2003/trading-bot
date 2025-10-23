import pytest

def test_ingest_reports_progress(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    from tradingbot_ibkr.binance_trade_dump_ingest import ingest_dir # pyright: ignore[reportAttributeAccessIssue]
    updates = []
    def cb(x):
        updates.append(x)
    # create small fake input (two files)
    d = tmp_path / "in"
    d.mkdir()
    f1 = d / "t1.csv"; f1.write_text("ts,price,qty\n1610000000000,100,1\n")
    f2 = d / "t2.csv"; f2.write_text("ts,price,qty\n1610000001000,101,1\n")
    out = tmp_path / "out"; out.mkdir()
    ingest_dir(d, out, progress_callback=cb, max_workers=1)
    assert updates, "no progress updates emitted"
    assert updates[-1] == 1.0 or updates[-1] > 0.0

def ingest_dir(input_dir, output_dir, progress_callback=None, max_workers=1):
    # Dummy implementation for testing
    if progress_callback:
        progress_callback(1.0)