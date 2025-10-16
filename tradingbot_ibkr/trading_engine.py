"""High-level trading engine connecting the pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from .asset_classes import AssetClass, get_volatility_threshold
from .decision_layer import PPOAgent
from .feature_extraction import technical_indicators
from .models.online_trainer import OnlineTrainer
from .risk_management import volatility_filter
from .signal_prediction import SignalPredictor


def _normalise_asset_classes(asset_classes: Sequence[AssetClass]) -> List[AssetClass]:
    """Return a unique list of asset classes preserving order."""

    seen = set()
    normalised = []
    for asset_class in asset_classes:
        if asset_class not in seen:
            normalised.append(asset_class)
            seen.add(asset_class)
    return normalised


@dataclass
class TradeOutcome:
    """Container describing the realised outcome of a trade used for online learning."""

    realised_return: float


class TradingEngine:
    """Simple orchestration of feature extraction, prediction, and decision making.

    The engine now maintains an incremental model that can continually learn from
    realised trade outcomes via :class:`~tradingbot_ibkr.models.online_trainer.OnlineTrainer`.
    It defaults to operating across crypto and forex markets and can rotate focus
    between the configured asset classes.
    """

    def __init__(
        self,
        asset_class: AssetClass | None = None,
        *,
        asset_classes: Sequence[AssetClass] | None = None,
        online_blend: float = 0.5,
        min_positive_return: float = 0.0,
    ) -> None:
        if asset_classes is None:
            if asset_class is None:
                asset_classes = (AssetClass.CRYPTO, AssetClass.FOREX)
            else:
                asset_classes = (asset_class,)

        if not asset_classes:
            raise ValueError("TradingEngine requires at least one asset class")

        self.asset_universe = _normalise_asset_classes(tuple(asset_classes))
        self._asset_index = 0

        if asset_class is not None and asset_class in self.asset_universe:
            self._asset_index = self.asset_universe.index(asset_class)

        self.predictor = SignalPredictor()
        self.agent = PPOAgent()
        self._online_trainer = OnlineTrainer()
        self._online_weight = float(max(0.0, min(1.0, online_blend)))
        self._min_positive_return = min_positive_return

        try:
            self._online_trainer.load()
        except Exception:
            # Loading is best-effort; failures should not prevent usage.
            pass

        self._update_asset_focus(self.asset_universe[self._asset_index])

    @property
    def asset_class(self) -> AssetClass:
        """Current asset class focus."""

        return self.asset_universe[self._asset_index]

    @property
    def asset_classes(self) -> List[AssetClass]:
        """Ordered list of asset classes managed by the engine."""

        return list(self.asset_universe)

    def _update_asset_focus(self, asset_class: AssetClass) -> None:
        self._volatility_threshold = get_volatility_threshold(asset_class)

    @property
    def volatility_threshold(self) -> float:
        return self._volatility_threshold

    def focus_on(self, asset_class: AssetClass) -> None:
        """Switch the engine focus to ``asset_class`` within the configured universe."""

        if asset_class not in self.asset_universe:
            raise ValueError(f"Asset class {asset_class.value} not in configured universe")
        self._asset_index = self.asset_universe.index(asset_class)
        self._update_asset_focus(asset_class)

    def rotate_asset_focus(self) -> AssetClass:
        """Rotate to the next asset class and return it."""

        self._asset_index = (self._asset_index + 1) % len(self.asset_universe)
        asset_class = self.asset_universe[self._asset_index]
        self._update_asset_focus(asset_class)
        return asset_class

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return technical_indicators(data)

    def _latest_feature_row(self, data: pd.DataFrame) -> pd.Series:
        features = self.prepare_features(data).fillna(0.0)
        return features.iloc[-1]

    def _features_to_dict(self, row: pd.Series) -> dict[str, float]:
        return {col: float(row[col]) for col in row.index}

    def train(self, data: pd.DataFrame, target) -> None:
        features = self.prepare_features(data).dropna()
        self.predictor.fit(features.values, target)

    def extract_feature_dict(self, data: pd.DataFrame) -> dict[str, float]:
        """Public helper used by tests and downstream components to reuse features."""

        return self._features_to_dict(self._latest_feature_row(data))

    def _blend_probabilities(self, model_prob: float, online_prob: float) -> float:
        return (1 - self._online_weight) * model_prob + self._online_weight * online_prob

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate a trading signal for the latest data point.

        The volatility filter threshold is determined by the configured asset class.
        """

        latest_row = self._latest_feature_row(data)
        latest_frame = latest_row.to_frame().T
        prob = float(self.predictor.predict(latest_frame.values)[0])
        online_prob = self._online_trainer.predict_proba(self._features_to_dict(latest_row))
        prob = self._blend_probabilities(prob, online_prob)

        if not volatility_filter(data["close"], threshold=self.volatility_threshold):
            return 0
        return self.agent.choose_action(prob)

    def update_from_trade(self, data: pd.DataFrame, outcome: TradeOutcome | float) -> None:
        """Update the online trainer using the realised outcome of a trade."""

        if isinstance(outcome, TradeOutcome):
            realised_return = outcome.realised_return
        else:
            realised_return = float(outcome)

        label = 1 if realised_return > self._min_positive_return else 0
        feature_dict = self._features_to_dict(self._latest_feature_row(data))
        self._online_trainer.learn_one(feature_dict, label)

    def save_online_model(self) -> None:
        """Persist the incremental model to disk."""

        self._online_trainer.save()
