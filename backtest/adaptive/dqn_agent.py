"""Deep Q-Network agent for adaptive trading policy optimisation.

The module provides a light-weight DQN implementation tailored for financial
time-series experimentation.  The goal is not to replace sophisticated
research frameworks but to offer a batteries-included agent that integrates
with the existing backtesting utilities.  Key design points:

* Works with PyTorch when available.  If PyTorch is not installed the module
  raises an informative :class:`ImportError` encouraging the user to install
  ``torch`` with GPU acceleration when possible.
* Focuses on discrete action spaces (``BUY``, ``SELL``, ``HOLD``).  Additional
  actions can be appended by providing a custom ``action_space`` sequence.
* Includes an experience replay buffer, target network updates, epsilon-greedy
  exploration and basic checkpointing helpers.  The defaults mirror common DQN
  hyper-parameters used in academic momentum / trend-following research.

The agent is intentionally opinionated about the structure of the observation
vector: callers should pre-compute indicators (RSI, momentum, ATR, etc.) and
feed them as numeric numpy arrays.  This keeps the implementation framework
agnostic and avoids coupling to any particular feature-engineering pipeline.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
except Exception as exc:  # pragma: no cover - gracefully degrade
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - short-circuit when torch available
    _TORCH_IMPORT_ERROR = None


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for the DQN agent. Install with `pip install torch`.",
        ) from _TORCH_IMPORT_ERROR


class _MLP(nn.Module):  # pragma: no cover - thin wrapper around torch
    def __init__(self, in_dim: int, hidden_layers: Sequence[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class DQNAgent:
    """Deep Q-Network agent with experience replay.

    Parameters
    ----------
    state_dim:
        Number of features describing the environment state.
    action_space:
        Iterable of discrete action labels.  Defaults to ("HOLD", "BUY", "SELL").
    hidden_layers:
        Widths of the feed-forward neural network layers.
    gamma:
        Discount factor for future rewards.
    epsilon:
        Starting epsilon for epsilon-greedy exploration.
    epsilon_min:
        Minimum epsilon after exponential decay.
    epsilon_decay:
        Decay factor applied after each training step.
    lr:
        Learning rate of the Adam optimiser.
    replay_size:
        Capacity of the replay buffer.
    batch_size:
        Number of experiences sampled per gradient step.
    target_sync:
        Steps between target network updates.
    device:
        Torch device ("cpu" or "cuda").  Defaults to CUDA if available.
    seed:
        Optional random seed for reproducibility.
    """

    state_dim: int
    action_space: Sequence[str] = ("HOLD", "BUY", "SELL")
    hidden_layers: Sequence[int] = (128, 128)
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    lr: float = 1e-3
    replay_size: int = 20_000
    batch_size: int = 256
    target_sync: int = 250
    device: Optional[str] = None
    seed: Optional[int] = None
    warmup: int = 1_000
    reward_clip: Optional[float] = 10.0
    gradient_clip: Optional[float] = 1.0
    _online_net: Optional[_MLP] = field(default=None, init=False, repr=False)
    _target_net: Optional[_MLP] = field(default=None, init=False, repr=False)
    _optim: Optional[Any] = field(default=None, init=False, repr=False)
    _replay: Deque[Transition] = field(default_factory=deque, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        self._build_models()

    # ------------------------------------------------------------------ utils
    def _build_models(self) -> None:
        action_dim = len(self.action_space)
        self._online_net = _MLP(self.state_dim, self.hidden_layers, action_dim).to(self.device)
        self._target_net = _MLP(self.state_dim, self.hidden_layers, action_dim).to(self.device)
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._optim = torch.optim.Adam(self._online_net.parameters(), lr=self.lr)
        self._replay = deque(maxlen=self.replay_size)
        self._steps = 0

    # ------------------------------------------------------------------- API
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Return an action index using epsilon-greedy policy."""

        _require_torch()
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(len(self.action_space)))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self._online_net(state_tensor)  # type: ignore[arg-type]
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, transition: Transition) -> None:
        if self.reward_clip is not None:
            reward = float(max(min(transition.reward, self.reward_clip), -self.reward_clip))
        else:
            reward = float(transition.reward)
        self._replay.append(
            Transition(
                transition.state, transition.action, reward, transition.next_state, transition.done
            )
        )

    # ---------------------------------------------------------------- training
    def step(self, batch_size: Optional[int] = None) -> Optional[float]:
        """Perform a single optimisation step.

        Returns the loss value or ``None`` if the replay buffer is not large
        enough yet (based on ``warmup``).
        """

        _require_torch()
        if len(self._replay) < max(self.warmup, self.batch_size):
            return None

        batch_size = batch_size or self.batch_size
        indices = np.random.choice(len(self._replay), batch_size, replace=False)
        batch = [self._replay[idx] for idx in indices]

        states = torch.as_tensor(
            np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [b.action for b in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [b.reward for b in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [b.done for b in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        q_values = self._online_net(states).gather(1, actions)  # type: ignore[arg-type]
        with torch.no_grad():
            next_q = self._target_net(next_states).max(1, keepdim=True)[0]  # type: ignore[arg-type]
            target = rewards + (1 - dones) * (self.gamma * next_q)

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, target)

        self._optim.zero_grad()
        loss.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._online_net.parameters(), self.gradient_clip)
        self._optim.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self._target_net.load_state_dict(self._online_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return float(loss.item())

    # ------------------------------------------------------------- high-level
    def train_episode(
        self,
        env: Iterable[Tuple[np.ndarray, int, float, np.ndarray, bool]],
        reward_shaping: Optional[Callable[[float], float]] = None,
    ) -> List[float]:
        """Train on a pre-generated sequence of transitions.

        ``env`` is any iterable yielding (state, action, reward, next_state,
        done).  The helper is intentionally generic to integrate with both
        vectorised simulations and discrete backtesting loops.
        """

        losses: List[float] = []
        for state, action, reward, next_state, done in env:
            shaped_reward = reward_shaping(reward) if reward_shaping else reward
            self.remember(Transition(state, action, shaped_reward, next_state, done))
            loss = self.step()
            if loss is not None:
                losses.append(loss)
        return losses

    def policy(self, state: np.ndarray) -> str:
        """Return the action label with the highest Q-value without exploration."""

        idx = self.act(state, explore=False)
        return str(self.action_space[idx])

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        """Persist the online network and optimiser state."""

        _require_torch()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self._online_net.state_dict(),  # type: ignore[union-attr]
            "optimizer": self._optim.state_dict(),  # type: ignore[union-attr]
            "epsilon": self.epsilon,
            "steps": self._steps,
            "config": {
                "state_dim": self.state_dim,
                "action_space": list(self.action_space),
                "hidden_layers": list(self.hidden_layers),
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "lr": self.lr,
                "replay_size": self.replay_size,
                "batch_size": self.batch_size,
                "target_sync": self.target_sync,
                "device": self.device,
                "seed": self.seed,
                "warmup": self.warmup,
                "reward_clip": self.reward_clip,
                "gradient_clip": self.gradient_clip,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        """Restore an agent checkpoint created with :meth:`save`."""

        _require_torch()
        payload = torch.load(path, map_location="cpu")
        cfg = payload["config"]
        agent = cls(**cfg)
        agent._online_net.load_state_dict(payload["model"])  # type: ignore[union-attr]
        agent._target_net.load_state_dict(payload["model"])  # type: ignore[union-attr]
        agent._optim.load_state_dict(payload["optimizer"])  # type: ignore[union-attr]
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        agent._steps = int(payload.get("steps", agent._steps))
        return agent


def build_transition_stream(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    done: np.ndarray,
) -> Iterable[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
    """Convenience generator for historical OHLCV datasets.

    All arrays should be shaped ``(n_samples, feature_dim)`` (except ``actions``
    which is ``(n_samples,)``).  This helper simply zips the numpy arrays into
    a replay-friendly stream and performs minimal validation to catch mismatched
    lengths early.
    """

    if not (len(states) == len(actions) == len(rewards) == len(next_states) == len(done)):
        raise ValueError("All transition arrays must share the same length")
    for idx in range(len(states)):
        yield (
            states[idx].astype(np.float32, copy=False),
            int(actions[idx]),
            float(rewards[idx]),
            next_states[idx].astype(np.float32, copy=False),
            bool(done[idx]),
        )


__all__ = ["DQNAgent", "Transition", "build_transition_stream"]
