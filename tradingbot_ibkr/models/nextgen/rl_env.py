"""Lightweight multi-agent market environment with Gym-style API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - create minimal stand-ins
    class _Space:  # minimal replacement
        def sample(self):
            raise NotImplementedError

    class spaces:  # type: ignore[assignment]
        class Box(_Space):
            def __init__(self, low, high, shape):
                self.low = low
                self.high = high
                self.shape = shape

            def sample(self):
                return np.random.uniform(self.low, self.high, self.shape).astype(np.float32)

        class Discrete(_Space):
            def __init__(self, n):
                self.n = n

            def sample(self):
                return np.random.randint(self.n)

    class gym:  # type: ignore[assignment]
        Env = object


Action = Dict[str, float]
Observation = Dict[str, np.ndarray]


@dataclass
class AgentState:
    position: float = 0.0
    pnl: float = 0.0


class MultiAgentMarketEnv(gym.Env):  # type: ignore[misc]
    """Two-agent market maker/taker environment with synthetic microprice dynamics."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 100, seed: Optional[int] = None) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.seed(seed)
        self.step_count = 0
        self.price = 100.0
        self.agents: Dict[str, AgentState] = {"maker": AgentState(), "taker": AgentState()}
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(4,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):  # type: ignore[override]
        self.seed(seed)
        self.step_count = 0
        self.price = 100.0
        for agent in self.agents.values():
            agent.position = 0.0
            agent.pnl = 0.0
        observation = self._build_observation()
        return observation, {}

    def step(self, action: Dict[str, Action]):  # type: ignore[override]
        self.step_count += 1
        maker_action = action.get("maker", {})
        taker_action = action.get("taker", {})

        spread_adj = float(maker_action.get("spread", 0.0))
        taker_volume = float(taker_action.get("volume", 0.0))

        price_move = self._rng.normal(0, 0.2) + 0.05 * taker_volume - 0.03 * spread_adj
        self.price = max(0.1, self.price + price_move)

        maker_fill = taker_volume * self._rng.uniform(0.5, 1.0)
        maker_price = self.price - max(0.01, spread_adj)
        taker_price = self.price + 0.01

        self._update_agent("maker", maker_fill, maker_price)
        self._update_agent("taker", taker_volume, taker_price)

        observation = self._build_observation()
        rewards = self._compute_rewards()
        terminated = self.step_count >= self.max_steps
        info = {"price": self.price, "agents": {k: dataclasses.asdict(v) for k, v in self.agents.items()}}
        return observation, rewards, terminated, False, info

    def _build_observation(self) -> Observation:
        maker = self.agents["maker"]
        taker = self.agents["taker"]
        obs_vec = np.array(
            [
                self.price,
                maker.position,
                taker.position,
                maker.pnl - taker.pnl,
            ],
            dtype=np.float32,
        )
        return {"global": obs_vec}

    def _update_agent(self, name: str, volume: float, trade_price: float) -> None:
        agent = self.agents[name]
        agent.pnl -= agent.position * (trade_price - self.price)
        agent.position += volume if name == "maker" else -volume
        agent.pnl += volume * (self.price - trade_price)

    def _compute_rewards(self) -> Dict[str, float]:
        maker = self.agents["maker"]
        taker = self.agents["taker"]
        maker_reward = maker.pnl - 0.01 * maker.position**2
        taker_reward = taker.pnl - 0.02 * abs(taker.position)
        return {"maker": maker_reward, "taker": taker_reward}

    def render(self):  # pragma: no cover - optional visualisation
        print(f"step={self.step_count} price={self.price:.2f} maker={self.agents['maker']} taker={self.agents['taker']}")
