"""Online incremental trainer using River with a pure-Python fallback."""

<<<<<<< HEAD
This trainer accepts feature dicts per bar and incrementally updates a classifier/regressor.
It exposes predict() and learn() methods and logs predictions to disk for evaluation.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict

from river import linear_model, preprocessing
=======
from pathlib import Path
import logging
import pickle
import math
>>>>>>> origin/main

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
<<<<<<< HEAD
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.path = MODEL_DIR / 'online_model.pkl'
        self._samples_seen = 0
        self._trained = False
        self._min_samples_ready = max(0, int(min_samples_ready))
=======
        if _HAS_RIVER:
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        else:
            self.model = _FallbackPipeline()
        self.path = MODEL_DIR / "online_model.pkl"
>>>>>>> origin/main

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

    def learn_one(self, x: Dict, y: int) -> None:
        """Update the model incrementally and track readiness."""
        try:
            self.model.learn_one(x, y)
            self._samples_seen += 1
            if self._samples_seen >= self._min_samples_ready:
                self._trained = True
        except Exception:
            logging.exception('learn_one failed')

<<<<<<< HEAD
    def is_ready(self) -> bool:
        """Return True once the trainer has sufficient data or a persisted model."""
        return self._trained
=======
    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.model, f)
>>>>>>> origin/main

    def save(self) -> None:
        """Persist the current model to disk."""
        try:
            payload = {
                'model': self.model,
                'samples_seen': self._samples_seen,
                'trained': self._trained,
                'min_samples_ready': self._min_samples_ready,
            }
            with open(self.path, 'wb') as f:
                pickle.dump(payload, f)
        except Exception:
            logging.exception('saving online model failed')

    def load(self) -> None:
        """Load a previously saved model if present."""
        if self.path.exists():
<<<<<<< HEAD
            try:
                with open(self.path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'model' in data:
                    self.model = data['model']
                    self._samples_seen = data.get('samples_seen', self._samples_seen)
                    self._trained = data.get('trained', self._samples_seen >= self._min_samples_ready)
                    self._min_samples_ready = data.get('min_samples_ready', self._min_samples_ready)
                else:
                    # Backwards compatibility with older payloads storing the raw model
                    self.model = data
                    self._trained = True
            except Exception:
                logging.exception('loading online model failed; keeping fresh model')
=======
            with open(self.path, "rb") as f:
                self.model = pickle.load(f)
>>>>>>> origin/main
