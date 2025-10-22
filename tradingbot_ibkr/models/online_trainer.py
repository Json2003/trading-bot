"""Online incremental trainer using River with a pure-Python fallback."""

from pathlib import Path
import logging
import pickle
import math
from typing import Dict

try:  # pragma: no cover - exercised in tests via fallback
    from river import linear_model, preprocessing  # type: ignore

    _HAS_RIVER = True
except Exception:  # pragma: no cover - offline environments
    _HAS_RIVER = False


class _FallbackLogReg:
    """Very small logistic regression trained with gradient descent."""

    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.bias = 0.0
        self.weights: dict[str, float] = {}

    def _sigmoid(self, z: float) -> float:
        z = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba_one(self, x: dict) -> dict[int, float]:
        z = self.bias
        for key, value in x.items():
            z += self.weights.get(key, 0.0) * float(value)
        prob = self._sigmoid(z)
        return {1: prob, 0: 1.0 - prob}

    def learn_one(self, x: dict, y: int):
        prob = self.predict_proba_one(x)[1]
        error = float(y) - prob
        for key, value in x.items():
            self.weights[key] = self.weights.get(key, 0.0) + self.lr * error * float(value)
        self.bias += self.lr * error
        return self


class _FallbackPipeline:
    def __init__(self):
        self.model = _FallbackLogReg()

    def predict_proba_one(self, x: dict) -> dict[int, float]:
        return self.model.predict_proba_one(x)

    def learn_one(self, x: dict, y: int):
        self.model.learn_one(x, y)
        return self
MODEL_DIR = Path(__file__).resolve().parents[1] / "model_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class OnlineTrainer:
    """Lightweight wrapper around a River model with persistence helpers."""

    def __init__(self, min_samples_ready: int = 200) -> None:
        # simple logistic regression pipeline for a binary up/down label
        if _HAS_RIVER:
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        else:
            self.model = _FallbackPipeline()
        self.path = MODEL_DIR / "online_model.pkl"

    def predict_proba(self, x: Dict) -> float:
        """Return probability of the positive class."""
        try:
            p = self.model.predict_proba_one(x)
            if isinstance(p, dict):
                # River returns either {True: prob, False: prob} or {1: prob, 0: prob}
                for key in (1, True):
                    if key in p:
                        value = float(p[key])
                        break
                else:
                    value = float(next(iter(p.values()), 0.0))
            else:
                value = float(p)
            if value != value:  # NaN check
                return 0.5
            return value
        except Exception:
            logging.exception("predict failed")
            return 0.0

    def learn_one(self, x: dict, y: int):
        self.model.learn_one(x, y)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self) -> None:
        """Load a previously saved model if present."""
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.model = pickle.load(f)
