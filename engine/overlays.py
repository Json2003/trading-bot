"""Portfolio overlay utilities (beta hedging, Comp-M tilts, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Any
import math

import pandas as pd
import yaml  # type: ignore[import-untyped]

from factors.beta import compute_rolling_beta
from factors.comp_m import compute_comp_m_factor
from engine.portfolio import PortfolioSnapshot
from strategies.base import StrategySignal


def _normalise_symbol(symbol: str) -> str:
    return symbol.replace(":", "/").split("/")[0] if ":" in symbol else symbol


@dataclass
class OverlayEngine:
    """Apply portfolio overlays based on factor data."""

    price_history: pd.DataFrame
    config_path: str
    momentum_strategy: str = "momentum"

    def __post_init__(self) -> None:
        cfg = yaml.safe_load(Path(self.config_path).read_text()) if self.config_path else {}
        factors_cfg = cfg.get("factors", {})
        overlays_cfg = cfg.get("overlays", {})

        self._step = 0

        self._symbol_prices = self._build_symbol_prices(self.price_history)
        self._betas = self._compute_betas(factors_cfg)
        self._beta_target = self._extract_beta_target(factors_cfg, overlays_cfg)
        comp_cfg = factors_cfg.get("comp_m") or {}
        self._comp_m = self._compute_comp_m(comp_cfg)
        self._tilt_strength = float(comp_cfg.get("tilt_strength", 0.0))
        self._beta_cache: dict[int, float] = {}

    def adjust_signal(
        self,
        cycle: int,
        strategy: str,
        signal: StrategySignal,
        snapshot: PortfolioSnapshot,
    ) -> StrategySignal:
        result = signal

        symbol_key = signal.symbol.replace(":", "/")
        if self._comp_m is not None and strategy == self.momentum_strategy:
            idx = min(cycle, len(self._comp_m) - 1)
            if idx >= 0:
                factor_value = float(self._comp_m.iloc[idx].get(symbol_key, 0.0))
                if not math.isnan(factor_value):
                    tilt = max(0.0, 1.0 + self._tilt_strength * factor_value)
                    result = replace(result, quantity=result.quantity * tilt)

        beta_scale = self._beta_scale(cycle, snapshot)
        if beta_scale is not None and strategy == self.momentum_strategy:
            result = replace(result, quantity=result.quantity * beta_scale)

        return result

    def after_cycle(self) -> None:
        self._step += 1

    # -- Internal helpers -------------------------------------------------
    def _build_symbol_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        buckets: dict[str, list[pd.Series]] = {}
        for column in prices.columns:
            symbol = column.split(":")[-1]
            buckets.setdefault(symbol, []).append(prices[column])
        combined: dict[str, pd.Series] = {}
        for symbol, series_list in buckets.items():
            base = series_list[0].astype(float)
            acc_values = list(base.to_list())
            for extra in series_list[1:]:
                extra_values = list(extra.astype(float).to_list())
                acc_values = [a + b for a, b in zip(acc_values, extra_values)]
            combined_series = pd.Series(acc_values, index=base.index)
            combined[symbol] = combined_series / max(len(series_list), 1)
        return pd.DataFrame(combined)

    def _compute_betas(self, factors_cfg: Mapping[str, object]) -> pd.DataFrame:
        raw_beta = factors_cfg.get("beta") or {}
        beta_cfg: dict[str, Any] = raw_beta if isinstance(raw_beta, dict) else {}
        window = int(beta_cfg.get("window", 240))
        min_periods = int(beta_cfg.get("min_periods", window))
        market_symbol = str(beta_cfg.get("market_symbol", "BTCUSDT"))
        market_symbol = market_symbol.replace("/", "") if "/" in market_symbol else market_symbol
        market_column = None
        for column in self._symbol_prices.columns:
            key = column.replace("/", "")
            if key.upper() == market_symbol.upper():
                market_column = column
                break
        if market_column is None:
            return pd.DataFrame(index=self._symbol_prices.index)

        market_returns = self._symbol_prices[market_column].pct_change()
        betas = pd.DataFrame(index=self._symbol_prices.index)
        for column in self._symbol_prices.columns:
            if column == market_column:
                continue
            asset_returns = self._symbol_prices[column].pct_change()
            beta_series = compute_rolling_beta(
                asset_returns, market_returns, window=window, min_periods=min_periods
            )
            betas[column] = beta_series
        return betas

    def _extract_beta_target(
        self, factors_cfg: Mapping[str, object], overlays_cfg: Mapping[str, object]
    ) -> tuple[float, float] | None:
        target = None
        raw_beta = factors_cfg.get("beta") or {}
        beta_cfg: dict[str, Any] = raw_beta if isinstance(raw_beta, dict) else {}
        raw_overlay = overlays_cfg.get("beta_hedge") or {}
        overlay_cfg: dict[str, Any] = raw_overlay if isinstance(raw_overlay, dict) else {}
        for key in ("target_beta", "target"):
            value = overlay_cfg.get(key) if key in overlay_cfg else beta_cfg.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                lower, upper = sorted(float(v) for v in value)
                target = (lower, upper)
                break
        return target

    def _compute_comp_m(self, comp_cfg: Mapping[str, object]) -> pd.DataFrame | None:
        if not comp_cfg:
            return None
        raw_cfg: dict[str, Any] = comp_cfg if isinstance(comp_cfg, dict) else {}
        lookback = int(raw_cfg.get("lookback", 60))
        skip = int(raw_cfg.get("skip", 0))
        lag = int(raw_cfg.get("lag", 1))
        frame = self._symbol_prices.copy()
        if not hasattr(frame, "columns") or getattr(frame.columns, "empty", False):
            return None
        return compute_comp_m_factor(frame, lookback=lookback, skip=skip, lag=lag)

    def _beta_scale(self, cycle: int, snapshot: PortfolioSnapshot) -> float | None:
        if (
            self._beta_target is None
            or not hasattr(self._betas, "columns")
            or getattr(self._betas.columns, "empty", False)
        ):
            return None
        idx = min(cycle, len(self._betas) - 1)
        if idx < 0:
            return None
        if idx in self._beta_cache:
            return self._beta_cache[idx]

        row = self._betas.iloc[idx]
        total_equity = snapshot.total_equity
        if total_equity <= 0:
            self._beta_cache[idx] = 1.0
            return 1.0

        beta_contrib = 0.0
        for state in snapshot.states.values():
            for position in state.positions:
                symbol = position.symbol
                beta = row.get(symbol)
                if beta is None or math.isnan(beta):
                    continue
                beta_contrib += beta * position.market_value

        current_beta = beta_contrib / total_equity if total_equity else 0.0
        lower, upper = self._beta_target
        target = (lower + upper) / 2
        if abs(current_beta) < 1e-9:
            current_beta = 1e-9 if target >= 0 else -1e-9
        if current_beta > upper:
            scale = max(target / current_beta, 0.5)
        elif current_beta < lower:
            scale = min(target / current_beta, 1.5)
        else:
            scale = 1.0

        self._beta_cache[idx] = scale
        return scale


__all__ = ["OverlayEngine"]
