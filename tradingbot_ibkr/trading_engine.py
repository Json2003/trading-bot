"""High-level trading engine connecting the pipeline components."""
from __future__ import annotations
import pandas as pd
from .feature_extraction import technical_indicators
from .signal_prediction import SignalPredictor
from .decision_layer import PPOAgent
from .risk_management import volatility_filter
from .asset_classes import AssetClass, get_volatility_threshold


class TradingEngine:
    """Simple orchestration of feature extraction, prediction, and decision making.

    Supports multiple asset classes via :class:`~tradingbot_ibkr.asset_classes.AssetClass`.
    """

    def __init__(self, asset_class: AssetClass = AssetClass.CRYPTO) -> None:
        self.predictor = SignalPredictor()
        self.agent = PPOAgent()
        self.asset_class = asset_class
        self.volatility_threshold = get_volatility_threshold(asset_class)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return technical_indicators(data)

    def train(self, data: pd.DataFrame, target) -> None:
        features = self.prepare_features(data).dropna()
        self.predictor.fit(features.values, target)

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate a trading signal for the latest data point.

        The volatility filter threshold is determined by the configured asset class.
        """
        features = self.prepare_features(data).iloc[-1:]
        prob = float(self.predictor.predict(features.values)[0])
        if not volatility_filter(data["close"], threshold=self.volatility_threshold):
            return 0
        return self.agent.choose_action(prob)
