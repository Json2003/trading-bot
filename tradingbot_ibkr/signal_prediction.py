"""Signal prediction models for trading.

Provides a common interface for different model types and orchestrates an
ensemble that blends gradient boosting, LSTM-style sequence modelling, and a
transformer-inspired order-flow module.  The ensemble exposes agreement-aware
decisions which are combined with fundamental filters before orders are issued.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

try:  # scikit-learn is optional in some deployment targets
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    GradientBoostingClassifier = None  # type: ignore


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass
class SignalPredictor:
    """Wrapper around a prediction model supporting several architectures."""

    model_type: str = "gbm"
    sequence_length: int = 16
    input_keys: Optional[Iterable[str]] = None
    learning_rate: float = 0.05
    model: Optional[GradientBoostingClassifier] = None
    _history: deque = field(default_factory=lambda: deque(maxlen=128), init=False)
    _gbm_X: list = field(default_factory=list, init=False)
    _gbm_y: list = field(default_factory=list, init=False)
    _weights: Optional[np.ndarray] = field(default=None, init=False)
    _bias: float = field(default=0.0, init=False)
    _lstm_params: Optional[Dict[str, np.ndarray]] = field(default=None, init=False)
    _transformer_params: Optional[Dict[str, np.ndarray]] = field(default=None, init=False)
    _last_hidden: Optional[np.ndarray] = field(default=None, init=False)
    _last_context: Optional[np.ndarray] = field(default=None, init=False)
    _last_vector: Optional[np.ndarray] = field(default=None, init=False)

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        if self.input_keys is not None:
            ordered = [features.get(key, 0.0) for key in self.input_keys]
        else:
            ordered = [features[key] for key in sorted(features.keys())]
        vec = np.asarray(ordered, dtype=float)
        if vec.ndim == 1:
            return vec
        return vec.reshape(-1)

    # ------------------------------------------------------------------
    # Observation / prediction interface
    # ------------------------------------------------------------------
    def observe(self, features: Dict[str, float]) -> None:
        """Record the latest feature vector for sequential models."""

        vector = self._vectorize(features)
        self._last_vector = vector
        if self.model_type in {"lstm", "transformer_orderflow"}:
            self._history.append(vector)

    def predict(self, features: Dict[str, float]) -> float:
        """Return probability of an upward move for the supplied features."""

        self.observe(features)
        if self.model_type == "gbm":
            return self._predict_gbm(self._last_vector)
        if self.model_type == "lstm":
            return self._predict_lstm()
        if self.model_type == "transformer_orderflow":
            return self._predict_transformer()
        # Default neutral probability when model type is unknown
        return 0.5

    # ------------------------------------------------------------------
    # Gradient boosting implementation
    # ------------------------------------------------------------------
    def _predict_gbm(self, vector: Optional[np.ndarray]) -> float:
        if vector is None or vector.size == 0:
            return 0.5
        if GradientBoostingClassifier is None:
            score = float(vector.mean())
            return float(_sigmoid(score))
        if self.model is None:
            score = float(vector.mean())
            return float(_sigmoid(score))
        proba = self.model.predict_proba(vector.reshape(1, -1))[0, 1]
        return float(proba)

    # ------------------------------------------------------------------
    # LSTM-style implementation
    # ------------------------------------------------------------------
    def _init_lstm_params(self, dim: int) -> None:
        rng = np.random.default_rng(42)
        self._lstm_params = {
            "wi": rng.normal(scale=0.1, size=dim),
            "wf": rng.normal(scale=0.1, size=dim),
            "wo": rng.normal(scale=0.1, size=dim),
            "wc": rng.normal(scale=0.1, size=dim),
        }
        self._weights = rng.normal(scale=0.1, size=dim)
        self._bias = 0.0

    def _predict_lstm(self) -> float:
        if not self._history:
            return 0.5
        dim = self._history[0].shape[0]
        if self._lstm_params is None or self._weights is None:
            self._init_lstm_params(dim)

        hidden = np.zeros(dim)
        cell = np.zeros(dim)
        for vec in list(self._history)[-self.sequence_length :]:
            wi = self._lstm_params["wi"]
            wf = self._lstm_params["wf"]
            wo = self._lstm_params["wo"]
            wc = self._lstm_params["wc"]
            input_gate = _sigmoid(vec * wi)
            forget_gate = _sigmoid(vec * wf)
            output_gate = _sigmoid(vec * wo)
            candidate = np.tanh(vec * wc)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)

        if self._weights is None:
            self._weights = np.ones(dim) / max(dim, 1)
        score = float(hidden @ self._weights + self._bias)
        self._last_hidden = hidden
        return float(_sigmoid(score))

    # ------------------------------------------------------------------
    # Transformer-style order flow implementation
    # ------------------------------------------------------------------
    def _init_transformer_params(self, dim: int) -> None:
        rng = np.random.default_rng(7)
        self._transformer_params = {
            "query": rng.normal(scale=0.2, size=(dim, dim)),
            "key": rng.normal(scale=0.2, size=(dim, dim)),
            "value": rng.normal(scale=0.2, size=(dim, dim)),
            "proj": rng.normal(scale=0.1, size=dim),
        }
        self._bias = 0.0

    def _predict_transformer(self) -> float:
        if not self._history:
            return 0.5
        dim = self._history[0].shape[0]
        if self._transformer_params is None:
            self._init_transformer_params(dim)

        query_w = self._transformer_params["query"]
        key_w = self._transformer_params["key"]
        value_w = self._transformer_params["value"]
        proj_w = self._transformer_params["proj"]

        seq = np.vstack(list(self._history)[-self.sequence_length :])
        queries = seq @ query_w
        keys = seq @ key_w
        values = seq @ value_w

        last_query = queries[-1]
        attn_scores = (last_query @ keys.T) / np.sqrt(dim)
        weights = np.exp(attn_scores - np.max(attn_scores))
        weights /= weights.sum() if weights.sum() != 0 else 1.0
        context = weights @ values
        score = float(context @ proj_w + self._bias)
        self._last_context = context
        return float(_sigmoid(score))

    # ------------------------------------------------------------------
    # Learning updates
    # ------------------------------------------------------------------
    def learn(self, features: Dict[str, float], outcome: float) -> None:
        vector = self._vectorize(features)
        target = float(outcome)

        if self.model_type == "gbm":
            if GradientBoostingClassifier is None:
                # Update simple linear weights as fallback
                if self._weights is None:
                    self._weights = np.zeros_like(vector)
                pred = float(_sigmoid(vector @ self._weights + self._bias))
                error = target - pred
                self._weights += self.learning_rate * error * vector
                self._bias += self.learning_rate * error
                return

            self._gbm_X.append(vector)
            self._gbm_y.append(target)
            if len(self._gbm_y) >= 25:
                self.model = GradientBoostingClassifier(random_state=42)
                self.model.fit(np.vstack(self._gbm_X), np.array(self._gbm_y))
            return

        if self.model_type == "lstm" and self._last_hidden is not None and self._weights is not None:
            pred = float(_sigmoid(self._last_hidden @ self._weights + self._bias))
            error = target - pred
            self._weights += self.learning_rate * error * self._last_hidden
            self._bias += self.learning_rate * error
            return

        if self.model_type == "transformer_orderflow" and self._last_context is not None:
            proj_w = self._transformer_params["proj"] if self._transformer_params else None
            if proj_w is None:
                return
            pred = float(_sigmoid(self._last_context @ proj_w + self._bias))
            error = target - pred
            self._transformer_params["proj"] = proj_w + self.learning_rate * error * self._last_context
            self._bias += self.learning_rate * error


@dataclass
class EnsembleDecision:
    """Container describing the ensemble vote."""

    consensus: float
    probabilities: Dict[str, float]
    ml_agreement: bool
    fundamentals_pass: bool
    threshold: float


@dataclass
class FundamentalFilter:
    """Rule-based filter using valuation and earnings features."""

    max_pe: float = 60.0
    min_earnings_growth: float = -0.05
    min_fundamental_score: float = -0.1
    history: deque = field(default_factory=lambda: deque(maxlen=256))

    def observe(self, fundamentals: Dict[str, float]) -> None:
        if fundamentals:
            self.history.append(fundamentals)

    def _dynamic_pe(self) -> float:
        if not self.history:
            return self.max_pe
        pe_values = [f.get("pe_ratio") for f in self.history if f.get("pe_ratio") is not None]
        if not pe_values:
            return self.max_pe
        return float(min(np.nanpercentile(pe_values, 85), self.max_pe))

    def evaluate(self, fundamentals: Dict[str, float]) -> bool:
        if not fundamentals:
            return False
        pe_ratio = fundamentals.get("pe_ratio")
        earnings_growth = fundamentals.get("earnings_growth")
        score = fundamentals.get("fundamental_score")

        if pe_ratio is None or np.isnan(pe_ratio) or pe_ratio <= 0:
            return False
        if pe_ratio > self._dynamic_pe():
            return False
        if earnings_growth is None or earnings_growth < self.min_earnings_growth:
            return False
        if score is None or score < self.min_fundamental_score:
            return False
        return True

    def learn(self, fundamentals: Dict[str, float], outcome: float) -> None:
        self.observe(fundamentals)
        if outcome <= 0:
            # Tighten filters when losses occur on permissive fundamentals
            growth = fundamentals.get("earnings_growth", 0.0)
            score = fundamentals.get("fundamental_score", 0.0)
            self.max_pe = max(10.0, self.max_pe * 0.98)
            self.min_earnings_growth = min(growth, self.min_earnings_growth + 0.01)
            self.min_fundamental_score = min(score, self.min_fundamental_score + 0.01)


@dataclass
class SignalEnsemble:
    """Coordinate multiple predictors and the fundamental filter."""

    predictors: Dict[str, SignalPredictor]
    fundamental_filter: FundamentalFilter
    min_confidence: float = 0.6

    def observe(self, features: Dict[str, float], fundamentals: Optional[Dict[str, float]] = None) -> None:
        for predictor in self.predictors.values():
            predictor.observe(features)
        if fundamentals is not None:
            self.fundamental_filter.observe(fundamentals)

    def evaluate(self, features: Dict[str, float], fundamentals: Dict[str, float]) -> EnsembleDecision:
        probabilities = {
            name: predictor.predict(features)
            for name, predictor in self.predictors.items()
        }
        consensus = float(np.mean(list(probabilities.values()))) if probabilities else 0.5
        ml_agreement = all(prob >= self.min_confidence for prob in probabilities.values())
        fundamentals_pass = self.fundamental_filter.evaluate(fundamentals)
        return EnsembleDecision(
            consensus=consensus,
            probabilities=probabilities,
            ml_agreement=ml_agreement,
            fundamentals_pass=fundamentals_pass,
            threshold=self.min_confidence,
        )

    def learn(self, features: Dict[str, float], fundamentals: Dict[str, float], outcome: float) -> None:
        for predictor in self.predictors.values():
            predictor.learn(features, outcome)
        self.fundamental_filter.learn(fundamentals, outcome)
