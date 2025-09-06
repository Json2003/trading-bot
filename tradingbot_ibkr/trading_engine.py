"""High-level trading engine connecting the pipeline components."""
from __future__ import annotations
import pandas as pd
from .feature_extraction import technical_indicators
from .signal_prediction import SignalPredictor
from .decision_layer import PPOAgent
from .risk_management import volatility_filter


class TradingEngine:
    """Simple orchestration of feature extraction, prediction, and decision making."""

    def __init__(self) -> None:
        self.predictor = SignalPredictor()
        self.agent = PPOAgent()

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return technical_indicators(data)

    def train(self, data: pd.DataFrame, target) -> None:
        features = self.prepare_features(data).dropna()
        self.predictor.fit(features.values, target)

    def generate_signal(self, data: pd.DataFrame) -> int:
        features = self.prepare_features(data).iloc[-1:]
        prob = float(self.predictor.predict(features.values)[0])
        if not volatility_filter(data["close"], threshold=0.05):
            return 0
        return self.agent.choose_action(prob)
