"""Online learning orchestrator that promotes checkpoints after guardrails."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional, Protocol, Tuple
from collections import deque

import torch

try:  # pragma: no cover - optional metrics backend
    from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
except Exception:  # pragma: no cover
    Gauge = None  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]
    push_to_gateway = None  # type: ignore[assignment]


class OnlineModel(Protocol):
    """Protocol describing the minimal methods required by the online learner."""

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor) -> float: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> None: ...


def default_guardrail(loss_history: Iterable[float], max_loss: float = 5.0) -> bool:
    """Simple guardrail: reject update if rolling loss exceeds threshold."""
    values = list(loss_history)
    if not values:
        return True
    return max(values) < max_loss


class OnlineLearnerService:
    """High-level orchestrator for streaming feature updates and model promotion."""

    def __init__(
        self,
        model: OnlineModel,
        checkpoint_dir: Path,
        guardrail_fn: Callable[[Iterable[float]], bool] = default_guardrail,
        window_size: int = 50,
        prometheus_gateway: Optional[str] = None,
    ) -> None:
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.guardrail_fn = guardrail_fn
        self.loss_window: Deque[float] = deque(maxlen=window_size)
        self.prometheus_gateway = prometheus_gateway
        self.registry = CollectorRegistry() if (CollectorRegistry and prometheus_gateway) else None
        if self.registry:
            self.loss_gauge = Gauge("online_train_loss", "Rolling training loss", registry=self.registry)
            self.version_gauge = Gauge("online_model_version", "Current promoted model timestamp", registry=self.registry)
        else:  # pragma: no cover - metrics optional
            self.loss_gauge = None
            self.version_gauge = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.active_version: Optional[str] = None

    def ingest_stream(self, stream: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Consume a stream of (features, labels) batches."""
        for x, y in stream:
            loss = self.model.fit_batch(x, y)
            self.loss_window.append(loss)
            self._update_metrics(loss)
            if self.guardrail_fn(self.loss_window):
                self._promote_model()

    def _promote_model(self) -> None:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ckpt_path = self.checkpoint_dir / f"online_model_{timestamp}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        meta_path = self.checkpoint_dir / "latest.json"
        meta = {"version": timestamp, "path": ckpt_path.as_posix(), "promoted_at": timestamp}
        meta_path.write_text(json.dumps(meta, indent=2))
        self.active_version = timestamp
        if self.version_gauge:
            self.version_gauge.set(float(timestamp.replace("T", "").replace("Z", "")))  # crude monotonic metric
            self._push_metrics()

    def _update_metrics(self, loss: float) -> None:
        if self.loss_gauge:
            self.loss_gauge.set(loss)
            self._push_metrics()

    def _push_metrics(self) -> None:
        if self.registry and push_to_gateway and self.prometheus_gateway:
            push_to_gateway(self.prometheus_gateway, job="online_learner", registry=self.registry)
