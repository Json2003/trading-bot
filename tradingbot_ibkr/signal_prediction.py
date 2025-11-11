"""Signal prediction models for trading.

Provides a common interface for different model types and orchestrates an
ensemble that blends gradient boosting, LSTM-style sequence modelling, and a
transformer-inspired order-flow module.  The ensemble exposes agreement-aware
decisions which are combined with fundamental filters before orders are issued.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, cast
import math

import numpy as np
NP = cast(Any, np)

try:  # scikit-learn is optional in some deployment targets
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    GradientBoostingClassifier = None


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    clipped = NP.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + NP.exp(-clipped))

def _sigmoid_scalar(x: float) -> float:
    s = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-s))


@dataclass
class SignalPredictor:
    """Wrapper around a prediction model supporting several architectures."""

    model_type: str = "gbm"
    sequence_length: int = 16
    input_keys: Optional[Iterable[str]] = None
    learning_rate: float = 0.05
    model: Optional[object] = None
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
        vec = NP.asarray(ordered, dtype=float)
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
            score = float(NP.mean(vector))
            return _sigmoid_scalar(score)
        if self.model is None:
            score = float(NP.mean(vector))
            return _sigmoid_scalar(score)
        proba = cast(Any, self.model).predict_proba(vector.reshape(1, -1))[0, 1]
        return float(proba)

    # ------------------------------------------------------------------
    # LSTM-style implementation
    # ------------------------------------------------------------------
    def _init_lstm_params(self, dim: int) -> None:
        rng = NP.random.default_rng(42)
        self._lstm_params = {
            "wi": cast(np.ndarray, rng.normal(scale=0.1, size=dim)),
            "wf": cast(np.ndarray, rng.normal(scale=0.1, size=dim)),
            "wo": cast(np.ndarray, rng.normal(scale=0.1, size=dim)),
            "wc": cast(np.ndarray, rng.normal(scale=0.1, size=dim)),
        }
        self._weights = cast(np.ndarray, rng.normal(scale=0.1, size=dim))
        self._bias = 0.0

    def _predict_lstm(self) -> float:
        if not self._history:
            return 0.5
        dim = self._history[0].shape[0]
        if self._lstm_params is None or self._weights is None:
            self._init_lstm_params(dim)
        params = self._lstm_params
        assert params is not None

        hidden = cast(np.ndarray, NP.zeros(dim))
        cell = cast(np.ndarray, NP.zeros(dim))
        for vec in list(self._history)[-self.sequence_length :]:
            wi = params["wi"]
            wf = params["wf"]
            wo = params["wo"]
            wc = params["wc"]
            input_gate = _sigmoid(vec * wi)
            forget_gate = _sigmoid(vec * wf)
            output_gate = _sigmoid(vec * wo)
            candidate = NP.tanh(vec * wc)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * NP.tanh(cell)

        if self._weights is None:
            self._weights = cast(np.ndarray, NP.ones(dim) / max(dim, 1))
        score = float(NP.dot(hidden, cast(np.ndarray, self._weights)) + float(self._bias))
        self._last_hidden = cast(np.ndarray, hidden)
        s = max(min(score, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-s))

    # ------------------------------------------------------------------
    # Transformer-style order flow implementation
    # ------------------------------------------------------------------
    def _init_transformer_params(self, dim: int) -> None:
        rng = NP.random.default_rng(7)
        self._transformer_params = {
            "query": cast(np.ndarray, rng.normal(scale=0.2, size=(dim, dim))),
            "key": cast(np.ndarray, rng.normal(scale=0.2, size=(dim, dim))),
            "value": cast(np.ndarray, rng.normal(scale=0.2, size=(dim, dim))),
            "proj": cast(np.ndarray, rng.normal(scale=0.1, size=dim)),
        }
        self._bias = 0.0

    def _predict_transformer(self) -> float:
        if not self._history:
            return 0.5
        dim = self._history[0].shape[0]
        if self._transformer_params is None:
            self._init_transformer_params(dim)
        params = self._transformer_params
        assert params is not None

        query_w = params["query"]
        key_w = params["key"]
        value_w = params["value"]
        proj_w = params["proj"]

        seq = NP.vstack(list(self._history)[-self.sequence_length :])
        queries = NP.matmul(seq, query_w)
        keys = NP.matmul(seq, key_w)
        values = NP.matmul(seq, value_w)

        last_query = queries[-1]
        attn_scores = NP.matmul(last_query, keys.T) / float(NP.sqrt(dim))
        weights = NP.exp(attn_scores - float(NP.max(attn_scores)))
        denom = float(NP.sum(weights)) or 1.0
        weights = weights / denom
        context = NP.matmul(weights, values)
        score = float(NP.dot(context, proj_w) + float(self._bias))
        self._last_context = cast(np.ndarray, context)
        return _sigmoid_scalar(score)

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
                    self._weights = cast(np.ndarray, NP.zeros_like(vector))
                sc = float(NP.dot(vector, cast(np.ndarray, self._weights)) + float(self._bias))
                pred = _sigmoid_scalar(sc)
                error = target - pred
                self._weights += self.learning_rate * error * vector
                self._bias += self.learning_rate * error
                return

            self._gbm_X.append(vector)
            self._gbm_y.append(target)
            if len(self._gbm_y) >= 25:
                self.model = GradientBoostingClassifier(random_state=42)
                cast(Any, self.model).fit(NP.vstack(self._gbm_X), NP.array(self._gbm_y))
            return

        if (
            self.model_type == "lstm"
            and self._last_hidden is not None
            and self._weights is not None
        ):
            sc = float(NP.dot(self._last_hidden, cast(np.ndarray, self._weights)) + float(self._bias))
            pred = _sigmoid_scalar(sc)
            error = target - pred
            self._weights += self.learning_rate * error * self._last_hidden
            self._bias += self.learning_rate * error
            return

        if self.model_type == "transformer_orderflow" and self._last_context is not None:
            params = self._transformer_params
            if params is None:
                return
            proj_w = params["proj"]
            sc = float(NP.dot(self._last_context, proj_w) + float(self._bias))
            pred = _sigmoid_scalar(sc)
            error = target - pred
            params["proj"] = proj_w + self.learning_rate * error * self._last_context
            self._bias += self.learning_rate * error

    # ------------------------------------------------------------------
    # Batch fit API (optional; used by TradingEngine.train)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Iterable[float]) -> None:
        if X.size == 0:
            return
        if GradientBoostingClassifier is not None and self.model_type == "gbm":
            self.model = GradientBoostingClassifier(random_state=42)
            try:
                cast(Any, self.model).fit(X, np.asarray(list(y), dtype=float))
                return
            except Exception:
                # Fall through to simple baseline if sklearn fit fails
                self.model = None
        # Lightweight baseline: set weights to normalised mean direction
        vec = NP.mean(X, axis=0)
        if vec is None or NP.all(~NP.isfinite(vec)):
            return
        if vec.ndim == 0:
            vec = NP.array([float(vec)])
        norm = NP.linalg.norm(vec)
        self._weights = cast(np.ndarray, (vec / norm) if norm > 0 else vec)
        self._bias = 0.0


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
        return float(min(NP.nanpercentile(pe_values, 85), self.max_pe))

    def evaluate(self, fundamentals: Dict[str, float]) -> bool:
        if not fundamentals:
            return False
        pe_ratio = fundamentals.get("pe_ratio")
        earnings_growth = fundamentals.get("earnings_growth")
        score = fundamentals.get("fundamental_score")

        if pe_ratio is None or NP.isnan(pe_ratio) or pe_ratio <= 0:
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

    def observe(
        self, features: Dict[str, float], fundamentals: Optional[Dict[str, float]] = None
    ) -> None:
        for predictor in self.predictors.values():
            predictor.observe(features)
        if fundamentals is not None:
            self.fundamental_filter.observe(fundamentals)

    def evaluate(
        self, features: Dict[str, float], fundamentals: Dict[str, float]
    ) -> EnsembleDecision:
        probabilities = {
            name: predictor.predict(features) for name, predictor in self.predictors.items()
        }
        consensus = float(NP.mean(list(probabilities.values()))) if probabilities else 0.5
        ml_agreement = all(prob >= self.min_confidence for prob in probabilities.values())
        fundamentals_pass = self.fundamental_filter.evaluate(fundamentals)
        return EnsembleDecision(
            consensus=consensus,
            probabilities=probabilities,
            ml_agreement=ml_agreement,
            fundamentals_pass=fundamentals_pass,
            threshold=self.min_confidence,
        )

    def learn(
        self, features: Dict[str, float], fundamentals: Dict[str, float], outcome: float
    ) -> None:
        for predictor in self.predictors.values():
            predictor.learn(features, outcome)
        self.fundamental_filter.learn(fundamentals, outcome)
