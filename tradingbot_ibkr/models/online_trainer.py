"""Online incremental trainer using River.

This trainer accepts feature dicts per bar and incrementally updates a classifier/regressor.
It exposes predict() and learn() methods and logs predictions to disk for evaluation.
"""
from river import linear_model, preprocessing
import pickle
from pathlib import Path
import logging

MODEL_DIR = Path(__file__).resolve().parents[1] / 'model_store'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class OnlineTrainer:
    def __init__(self):
        # simple logistic regression pipeline for a binary up/down label
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.path = MODEL_DIR / 'online_model.pkl'

    def predict_proba(self, x: dict) -> float:
        try:
            p = self.model.predict_proba_one(x)
            # return probability of positive class if present
            return p.get(1, 0.0) if isinstance(p, dict) else 0.0
        except Exception:
            logging.exception('predict failed')
            return 0.0

    def learn_one(self, x: dict, y: int):
        self.model.learn_one(x, y)

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        if self.path.exists():
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
