"""Decision layer with reinforcement learning agents."""
from __future__ import annotations
import numpy as np


class PPOAgent:
    """Very small placeholder for PPO-based decision logic."""

    def choose_action(self, signal: float) -> int:
        """Return 1 for long, -1 for short, 0 for hold."""
        if signal > 0.55:
            return 1
        if signal < 0.45:
            return -1
        return 0


class DDQNAgent:
    """Placeholder Double DQN-style agent using random tie-breaking."""

    rng = np.random.default_rng()

    def choose_action(self, signal: float) -> int:
        if signal > 0.6:
            return 1
        if signal < 0.4:
            return -1
        return int(self.rng.choice([-1, 0, 1]))
