# Next-Generation Platform Components

This document summarises the scaffolding added for next-generation modelling,
online learning, risk, and execution capabilities. Each component is designed
as a thin slice that can be iteratively expanded.

## 1. Models / NextGen

- `tradingbot_ibkr/models/nextgen/transformer_moe.py` implements a transformer
  encoder with a mixture-of-experts head and MC-dropout uncertainty estimates.
  A PyTorch Lightning wrapper is provided when `pytorch-lightning` is installed.
- `tradingbot_ibkr/models/nextgen/ensemble.py` blends predictions from any
  uncertainty-aware model using entropy-weighted averaging.
- `tradingbot_ibkr/models/nextgen/rl_env.py` supplies a multi-agent market
  environment with a Gym-style API to prototype maker/taker policies.
- `tradingbot_ibkr/models/nextgen/gnn_correlations.py` offers a graph neural net
  for asset-correlation learning, utilising PyG when available and falling back
  to a lightweight aggregator otherwise.

### Demo snippet

```python
from tradingbot_ibkr.models.nextgen import build_default_model, UncertaintyEnsembler
import torch

model = build_default_model(input_dim=16, num_classes=2)
x = torch.randn(8, 32, 16)  # batch, seq_len, features
probs, entropy = model.predict_with_uncertainty(x)
```

## 2. Adaptive Infrastructure

- `tradingbot_ibkr/services/online_learner.py` orchestrates streaming training,
  checkpoint promotion, and Prometheus metrics pushes (when the client is
  available). Guard-rail hooks ensure only healthy checkpoints are promoted.
- `tradingbot_ibkr/services/regime_sandbox.py` replays stress windows or injects
  shocks, making it easy to run nightly CI simulations for regime testing.

### Example

```python
from tradingbot_ibkr.services import OnlineLearnerService
from tradingbot_ibkr.models.online_trainer import OnlineTrainer
from pathlib import Path
import torch

trainer = OnlineTrainer()
learner = OnlineLearnerService(trainer, Path("model_store/online"))
stream = [(torch.randn(32, 16), torch.randint(0, 2, (32,))) for _ in range(10)]
learner.ingest_stream(stream)
```

## 3. Risk IQ Layer

- `tradingbot_ibkr/risk/cvar_overlay.py` wraps model signals in a CVaR-aware
  sizing module that adjusts allocations based on downside tail estimates.

### Example

```python
from tradingbot_ibkr.risk import CVaRRiskOverlay, SignalPacket

overlay = CVaRRiskOverlay(confidence=0.95, max_allocation=0.5, capital=10_000)
signals = [
    SignalPacket("BTCUSDT", raw_signal=0.8, volatility=0.05, predicted_pnl_distribution=[-20, -10, 5, 15, 30]),
]
sizes = overlay.apply(signals)
```

## 4. Execution Edge

- `tradingbot_ibkr/execution/smart_router.py` introduces a smart router that
  weighs latency, spreads, fill rates, and reaction scores (hook for future
  game-theoretic models).

```python
from tradingbot_ibkr.execution import SmartOrderRouter, VenueStats

router = SmartOrderRouter({
    "venue_a": VenueStats(latency_ms=12, spread=0.05, fill_rate=0.92),
    "venue_b": VenueStats(latency_ms=20, spread=0.03, fill_rate=0.85),
})
venue = router.route("buy", size=1.0)
```

## Next Steps

1. Wire the feature registry outputs into the transformer training pipeline.
2. Extend `RegimeSandbox` with historical stress windows sourced from the
   feature store; schedule nightly CI jobs that run `sandbox.run(...)`.
3. Integrate the CVaR overlay and smart router into the live trading pipeline
   by wrapping existing strategy executors.
4. Replace the stubbed opponents in `MultiAgentMarketEnv` with actual RL
   policies and plug in CVaR-aware rewards.
