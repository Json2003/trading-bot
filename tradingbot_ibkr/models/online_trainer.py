"""Online incremental trainer using River with a pure-Python fallback."""

This trainer accepts feature dicts per bar and incrementally updates a classifier/regressor.
It exposes predict() and learn() methods and logs predictions to disk for evaluation.
"""
from river import linear_model, preprocessing
import pickle
from pathlib import Path
import logging

MODEL_DIR = Path(__file__).resolve().parents[1] / "model_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class OnlineTrainer:
    """Lightweight wrapper around a River model with persistence helpers."""

    def __init__(self, min_samples_ready: int = 200) -> None:
        # simple logistic regression pipeline for a binary up/down label
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.path = MODEL_DIR / 'online_model.pkl'

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
        with open(self.path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self) -> None:
        """Load a previously saved model if present."""
        if self.path.exists():
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
