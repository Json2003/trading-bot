import pandas as pd

from tradingbot_ibkr.asset_classes import AssetClass, get_volatility_threshold
from tradingbot_ibkr.trading_engine import TradeOutcome, TradingEngine


def _build_sample_data(rows: int = 60) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="T")
    close = pd.Series(range(rows), dtype=float) + 100.0
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(1000.0, index=index)
    return pd.DataFrame(
        {"close": close.values, "high": high.values, "low": low.values, "volume": volume.values},
        index=index,
    )


def test_online_learning_updates_probability() -> None:
    engine = TradingEngine(asset_classes=[AssetClass.CRYPTO], online_blend=1.0)
    data = _build_sample_data()
    feature_dict = engine.extract_feature_dict(data)
    baseline = engine._online_trainer.predict_proba(feature_dict)

    for _ in range(50):
        engine.update_from_trade(data, TradeOutcome(realised_return=0.02))

    updated = engine._online_trainer.predict_proba(engine.extract_feature_dict(data))
    assert updated > baseline


def test_online_learning_reacts_to_losses() -> None:
    engine = TradingEngine(asset_classes=[AssetClass.CRYPTO], online_blend=1.0)
    data = _build_sample_data()

    for _ in range(50):
        engine.update_from_trade(data, 0.02)
    feature_dict = engine.extract_feature_dict(data)
    positive_prob = engine._online_trainer.predict_proba(feature_dict)

    for _ in range(100):
        engine.update_from_trade(data, -0.05)
    lowered_prob = engine._online_trainer.predict_proba(engine.extract_feature_dict(data))

    assert lowered_prob < positive_prob


def test_asset_rotation_between_crypto_and_forex() -> None:
    engine = TradingEngine()
    assert engine.asset_classes == [AssetClass.CRYPTO, AssetClass.FOREX]
    assert engine.asset_class == AssetClass.CRYPTO
    assert engine.volatility_threshold == get_volatility_threshold(AssetClass.CRYPTO)

    engine.focus_on(AssetClass.FOREX)
    assert engine.asset_class == AssetClass.FOREX
    assert engine.volatility_threshold == get_volatility_threshold(AssetClass.FOREX)

    rotated = engine.rotate_asset_focus()
    assert rotated == AssetClass.CRYPTO
    assert engine.asset_class == AssetClass.CRYPTO
