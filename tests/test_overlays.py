import pandas as pd

from engine.overlays import OverlayEngine


def test_overlay_engine_scales_signals() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="H")
    prices = pd.DataFrame(
        {
            "binance:BTC/USDT": [50000, 50500, 51000, 51500, 52000],
            "binance:ETH/USDT": [2000, 2025, 2050, 2075, 2100],
        },
        index=index,
    )

    engine = OverlayEngine(prices, config_path="config/factors.yaml")
    assert engine is not None
