"""Signal prediction models for trading.

Provides a common interface for different model types and basic persistence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore

MODEL_DIR = Path(__file__).resolve().parents[1] / "model_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SignalPredictor:
    """Wrapper around a prediction model with persistence helpers."""

    model_type: str = "gbm"
    model: Optional[GradientBoostingClassifier] = None
    model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_path = MODEL_DIR / f"{self.model_type}_predictor.joblib"
        if self.model is None:
            self.load(silent=True)

    def fit(self, X: np.ndarray, y: np.ndarray, save: bool = True) -> None:
        """Fit the underlying model and optionally persist it."""
        if self.model_type != "gbm":
            # Placeholder for LSTM/Transformer implementations
            self.model_type = "gbm"
            self.model_path = MODEL_DIR / f"{self.model_type}_predictor.joblib"
        self.model = GradientBoostingClassifier()
        self.model.fit(X, y)
        if save:
            self.save()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict signal probabilities for ``X``."""
        if self.model is None:
            return np.zeros(len(X))
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def save(self, path: Optional[Path] = None) -> None:
        """Persist the trained model to disk."""
        if self.model is None or joblib is None:
            return
        target = Path(path) if path else self.model_path
        try:
            joblib.dump(self.model, target)
        except Exception:  # pragma: no cover - filesystem issues
            logging.exception("failed to save predictor model")

    def load(self, path: Optional[Path] = None, *, silent: bool = False) -> bool:
        """Load a trained model from disk, if present.

        Returns True when a model is loaded successfully.
        """
        if joblib is None:
            if not silent:
                logging.warning("joblib not available; unable to load predictor model")
            return False
        target = Path(path) if path else self.model_path
        if not target.exists():
            if not silent:
                logging.info("predictor model file not found at %s", target)
            return False
        try:
            self.model = joblib.load(target)
            return True
        except Exception:  # pragma: no cover - filesystem issues
            if not silent:
                logging.exception("failed to load predictor model")
            return False
