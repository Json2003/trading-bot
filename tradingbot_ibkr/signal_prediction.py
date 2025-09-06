"""Signal prediction models for trading.

Provides a common interface for different model types.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class SignalPredictor:
    """Wrapper around a prediction model.

    Parameters
    ----------
    model_type: str
        Type of model to use. Currently supports ``"gbm"``.
    """
    model_type: str = "gbm"
    model: Optional[GradientBoostingClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying model.

        Unrecognised model types fall back to gradient boosting.
        """
        if self.model_type != "gbm":
            # Placeholder for LSTM/Transformer implementations
            self.model_type = "gbm"
        self.model = GradientBoostingClassifier()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict signal probabilities for ``X``.

        Returns array of probabilities that can be fed into decision agents.
        """
        if self.model is None:
            return np.zeros(len(X))
        proba = self.model.predict_proba(X)
        return proba[:, 1]
