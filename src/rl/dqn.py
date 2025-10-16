"""Reinforcement-learning helpers built around Stable Baselines3 PPO."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


def _lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all columns lower-cased."""
    renamed = {col: col.lower() for col in df.columns}
    return df.rename(columns=renamed, copy=True)


def _ensure_column(df: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    """Return the first matching column from ``names`` (case insensitive)."""
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return df[lowered[key]].astype(float)
    return pd.Series(np.full(len(df), default, dtype=float), index=df.index)


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


@dataclass(slots=True)
class TradingMetrics:
    """Container for per-step portfolio diagnostics."""

    portfolio_value: float
    balance: float
    position: float


class TradingEnv(gym.Env):
    """Simple long-only trading environment with buy/sell/hold actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        initial_balance: float = 10_000.0,
        commission: float = 0.0005,
    ) -> None:
        super().__init__()
        if data.empty:
            raise ValueError("TradingEnv requires non-empty price data")

        self.data = data.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.last_price = float(self.data.loc[0, "close"])

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.last_price = float(self.data.loc[0, "close"])
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        row = self.data.loc[self.current_step, ["close", "sma_fast", "sma_slow", "rsi", "volume"]]
        return row.to_numpy(dtype=np.float32)

    def _get_metrics(self, price: float) -> TradingMetrics:
        portfolio_value = self.balance + self.position * price
        return TradingMetrics(
            portfolio_value=portfolio_value, balance=self.balance, position=self.position
        )

    def step(self, action: int):  # type: ignore[override]
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}; expected 0 (hold), 1 (buy), or 2 (sell)")

        price = float(self.data.loc[self.current_step, "close"])
        metrics_before = self._get_metrics(price)

        if action == 1:  # Buy
            max_shares = int(self.balance // (price * (1.0 + self.commission)))
            if max_shares > 0:
                cost = max_shares * price * (1.0 + self.commission)
                self.balance -= cost
                self.position += max_shares
        elif action == 2 and self.position > 0:  # Sell
            proceeds = self.position * price * (1.0 - self.commission)
            self.balance += proceeds
            self.position = 0.0

        terminated = self.current_step >= len(self.data) - 1
        self.current_step += 1
        self.last_price = price

        next_obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        metrics_after = self._get_metrics(price)
        portfolio_change = metrics_after.portfolio_value - metrics_before.portfolio_value
        reward = portfolio_change / self.initial_balance

        drawdown = max(0.0, 1.0 - metrics_after.portfolio_value / self.initial_balance)
        reward -= 0.5 * drawdown

        info = {
            "balance": metrics_after.balance,
            "position": metrics_after.position,
            "portfolio_value": metrics_after.portfolio_value,
        }
        return next_obs, float(reward), bool(terminated), False, info


def prepare_rl_features(
    data: pd.DataFrame,
    *,
    fast_window: int = 10,
    slow_window: int = 30,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Create technical indicators expected by :class:`TradingEnv`."""

    if data.empty:
        raise ValueError("Cannot prepare features for an empty DataFrame")

    df = _lowercase_columns(data)
    close = _ensure_column(df, ["close"]).fillna(method="ffill").fillna(method="bfill")
    volume = _ensure_column(df, ["volume"], default=0.0)

    features = pd.DataFrame(index=df.index)
    features["close"] = close
    features["volume"] = volume.fillna(0.0)
    features["sma_fast"] = close.rolling(window=fast_window, min_periods=1).mean()
    features["sma_slow"] = close.rolling(window=slow_window, min_periods=1).mean()
    features["rsi"] = _compute_rsi(close, rsi_period)

    return features.reset_index(drop=True)


def train_rl_agent(
    data: pd.DataFrame,
    *,
    model_path: str | Path = "artifacts/rl_ppo_model.zip",
    total_timesteps: int = 100_000,
    env_kwargs: dict[str, Any] | None = None,
    ppo_kwargs: dict[str, Any] | None = None,
    feature_kwargs: dict[str, Any] | None = None,
):
    """Train a PPO agent on OHLCV data and persist the checkpoint."""

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    features = prepare_rl_features(data, **(feature_kwargs or {}))
    env_args = env_kwargs or {}

    def _make_env():
        return TradingEnv(features, **env_args)

    vec_env = DummyVecEnv([_make_env])
    model = PPO("MlpPolicy", vec_env, verbose=1, **(ppo_kwargs or {}))
    model.learn(total_timesteps=total_timesteps)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    return model


def generate_signals_rl(
    data: pd.DataFrame,
    *,
    model_path: str | Path,
    env_kwargs: dict[str, Any] | None = None,
    feature_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run inference with a trained agent and emit buy/sell signals."""

    from stable_baselines3 import PPO

    features = prepare_rl_features(data, **(feature_kwargs or {}))
    env = TradingEnv(features, **(env_kwargs or {}))

    model = PPO.load(str(model_path))
    obs, _ = env.reset()

    signals = pd.Series(0, index=features.index, dtype=int)

    for step in range(len(features)):
        action, _ = model.predict(obs, deterministic=True)
        if action == 1:
            signals.iloc[step] = 1
        elif action == 2:
            signals.iloc[step] = -1

        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    result = data.copy()
    result["signal"] = signals.to_numpy()
    return result
