"""Automated nightly research loop leveraging Optuna to tune multi-strategy configs.

The module orchestrates time-sliced optimisation runs against non-overlapping
windows of historical data.  Each candidate configuration adjusts the key
hyperparameters for the live strategies (grid trading, EMA momentum, DCA, and
cross-exchange arbitrage).  The objective maximises out-of-sample Sharpe while
penalising turnover and excessive drawdowns.  Results are persisted to a JSON
"model registry" so the best performers can be promoted to paper trading after
manual review.

Example
-------
>>> import pandas as pd
>>> from pathlib import Path
>>> from backtest.optimization.research_loop import (
...     NightlyResearchLoop,
...     create_non_overlapping_windows,
... )
>>> df = pd.read_csv("backtest/sample_data/sample_ohlcv.csv")
>>> windows = create_non_overlapping_windows(df, window_size=400, test_fraction=0.3)
>>> loop = NightlyResearchLoop(windows, Path("configs/research_registry.json"))
>>> loop.run(max_windows=1)  # doctest: +SKIP

The JSON registry captures the top-N configurations along with diagnostics so
operators can audit each candidate before enabling the strategy in live
trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import random
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_site(mod_name: str):
    to_delete = [name for name in sys.modules if name == mod_name or name.startswith(f"{mod_name}.")]
    for name in to_delete:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        try:
            if str(REPO_ROOT) in os.path.abspath(mod_file):
                del sys.modules[name]
        except Exception:
            del sys.modules[name]
    original = sys.path.copy()
    removed_finders: list[object] = []
    for finder in list(sys.meta_path):
        if finder.__class__.__name__ == "_StubFinder":
            sys.meta_path.remove(finder)
            removed_finders.append(finder)
    repo_paths = {p for p in original if str(REPO_ROOT) in os.path.abspath(p)}
    non_repo = [p for p in original if p not in repo_paths]
    try:
        sys.path = non_repo + [p for p in original if p in repo_paths]
        module = importlib.import_module(mod_name)
        sys.modules[mod_name] = module
        return module
    finally:
        sys.path = original


np = _import_site("numpy")
pd = _import_site("pandas")

import optuna

from backtest.metrics import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class ResearchWindow:
    """Represents an in-sample / out-of-sample split for optimisation."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class StrategyEvaluation:
    """Holds train/test metrics for a single strategy component."""

    name: str
    train_summary: Mapping[str, float]
    test_summary: Mapping[str, float]
    train_returns: Sequence[float]
    test_returns: Sequence[float]
    train_trades: int
    test_trades: int


@dataclass
class RegistryEntry:
    """Serializable representation of a winning configuration."""

    timestamp: str
    window: str
    params: Mapping[str, float]
    score: float
    oos_sharpe: float
    train_max_drawdown: float
    oos_max_drawdown: float
    turnover: float
    diagnostics: Mapping[str, object]
    flag_for_paper_trial: bool = False


class ModelRegistry:
    """Maintains the top-N research results in JSON form."""

    def __init__(self, path: Path, top_n: int = 5) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._top_n = int(max(1, top_n))
        self._entries: list[RegistryEntry] = []
        self._load()

    # ------------------------------------------------------------------ utils
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return
        entries: list[RegistryEntry] = []
        for item in raw:
            try:
                entries.append(
                    RegistryEntry(
                        timestamp=str(item["timestamp"]),
                        window=str(item["window"]),
                        params=dict(item.get("params", {})),
                        score=float(item["score"]),
                        train_max_drawdown=float(item.get("train_max_drawdown", float("nan"))),
                        oos_sharpe=float(item["oos_sharpe"]),
                        oos_max_drawdown=float(item["oos_max_drawdown"]),
                        turnover=float(item.get("turnover", 0.0)),
                        diagnostics=dict(item.get("diagnostics", {})),
                        flag_for_paper_trial=bool(item.get("flag_for_paper_trial", False)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._entries = entries[: self._top_n]

    def save(self) -> None:
        payload = [
            {
                "timestamp": entry.timestamp,
                "window": entry.window,
                "params": dict(entry.params),
                "score": entry.score,
                "train_max_drawdown": entry.train_max_drawdown,
                "oos_sharpe": entry.oos_sharpe,
                "oos_max_drawdown": entry.oos_max_drawdown,
                "turnover": entry.turnover,
                "diagnostics": entry.diagnostics,
                "flag_for_paper_trial": entry.flag_for_paper_trial,
            }
            for entry in self._entries
        ]
        self._path.write_text(json.dumps(payload, indent=2))

    # ----------------------------------------------------------------- public
    @property
    def best_score(self) -> float:
        if not self._entries:
            return float("nan")
        return max(entry.score for entry in self._entries)

    def add(self, entry: RegistryEntry) -> None:
        self._entries.append(entry)
        self._entries.sort(key=lambda item: item.score, reverse=True)
        self._entries = self._entries[: self._top_n]
        self.save()


def create_non_overlapping_windows(
    df: pd.DataFrame,
    *,
    window_size: int,
    test_fraction: float,
    max_windows: int | None = None,
) -> list[ResearchWindow]:
    """Split *df* into contiguous non-overlapping optimisation windows.

    Parameters
    ----------
    df:
        Historical OHLCV data sorted in chronological order.
    window_size:
        Number of rows allocated to each window (train + test combined).
    test_fraction:
        Fraction of the window reserved for out-of-sample evaluation.  Values
        are clipped to ``[0.1, 0.9]``.
    max_windows:
        Optional hard cap on the number of windows.  When omitted the function
        returns as many windows as the dataset permits.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    frac = float(max(0.1, min(0.9, test_fraction)))
    total = len(df)
    if total < window_size:
        raise ValueError("DataFrame shorter than requested window_size")

    windows: list[ResearchWindow] = []
    offset = 0
    counter = 1
    max_allowed = max_windows if max_windows is None else max(int(max_windows), 0)

    while offset + window_size <= total:
        chunk = df.iloc[offset : offset + window_size].reset_index(drop=True)
        split = int(max(1, round(len(chunk) * (1.0 - frac))))
        train = chunk.iloc[:split].reset_index(drop=True)
        test = chunk.iloc[split:].reset_index(drop=True)
        if len(test) < 10 or len(train) < 10:
            offset += window_size
            counter += 1
            continue
        windows.append(ResearchWindow(name=f"window_{counter}", train=train, test=test))
        offset += window_size
        counter += 1
        if max_allowed and len(windows) >= max_allowed:
            break
    return windows


def _grid_signals_factory(levels: int, range_pct: float):
    level_count = max(int(levels), 2)
    span = max(float(range_pct), 0.0)

    def _signals(df: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(df).reset_index(drop=True)
        if "close" not in frame.columns:
            raise KeyError("Dataset requires a 'close' column for grid signals")
        prices = frame["close"].astype(float)
        if len(prices) == 0:
            return pd.DataFrame({"signals": np.zeros(0, dtype=int)})
        mid = float(prices.mean())
        lower = mid * (1.0 - span)
        upper = mid * (1.0 + span)
        grid_levels = np.linspace(lower, upper, level_count)
        signals: list[int] = []
        for price in prices:
            if not math.isfinite(price):
                signals.append(0)
                continue
            below = int(np.sum(price > grid_levels))
            above = int(np.sum(price < grid_levels))
            diff = above - below
            if diff > 0:
                signals.append(1)
            elif diff < 0:
                signals.append(-1)
            else:
                signals.append(0)
        return pd.DataFrame({"signals": np.asarray(signals, dtype=int)})

    _signals.__name__ = "grid"
    return _signals


def _momentum_signals_factory(fast: int, slow: int):
    fast = int(max(fast, 1))
    slow = int(max(slow, fast + 1))

    def _signals(df: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(df).reset_index(drop=True)
        if "close" not in frame.columns:
            raise KeyError("Dataset requires a 'close' column for momentum signals")
        close = frame["close"].astype(float)
        if len(close) == 0:
            return pd.DataFrame({"signals": np.zeros(0, dtype=int)})
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        diff = fast_ema - slow_ema
        direction = np.sign(np.nan_to_num(diff.to_numpy(), nan=0.0))
        return pd.DataFrame({"signals": direction.astype(int)})

    _signals.__name__ = "momentum"
    return _signals


def _simulate_dca(df: pd.DataFrame, step_pct: float, max_layers: int) -> tuple[Mapping[str, float], list[float], int]:
    prices = pd.Series(df["close"], dtype=float)
    capital = 1.0
    cash = capital
    position = 0.0
    avg_entry = 0.0
    layers = 0
    trade_pnls: list[float] = []
    equity_curve: list[float] = []
    returns: list[float] = []
    prev_equity = capital
    trades = 0

    for price in prices:
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            equity = cash + position * prev_equity
            equity_curve.append(equity)
            returns.append(0.0)
            prev_equity = equity
            continue

        # Exit condition first so recoveries are captured promptly.
        if position > 0 and price >= avg_entry * (1.0 + step_pct):
            proceeds = position * price
            pnl = proceeds - position * avg_entry
            cash += proceeds
            trade_pnls.append(pnl)
            position = 0.0
            avg_entry = 0.0
            layers = 0
            trades += 1

        # Initial entry or scale-in on drawdowns.
        unit_notional = capital * 0.1
        if position == 0.0:
            qty = unit_notional / price
            cost = qty * price
            if cash >= cost:
                cash -= cost
                position += qty
                avg_entry = price
                layers = 1
                trades += 1
        else:
            target = avg_entry * (1.0 - step_pct)
            if price <= target and layers < max_layers:
                qty = unit_notional / price
                cost = qty * price
                if cash >= cost:
                    new_pos = position + qty
                    avg_entry = (avg_entry * position + price * qty) / new_pos
                    position = new_pos
                    cash -= cost
                    layers += 1
                    trades += 1

        equity = cash + position * price
        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        returns.append(float(ret))
        equity_curve.append(float(equity))
        prev_equity = equity

    if position > 0:
        proceeds = position * prices.iloc[-1]
        pnl = proceeds - position * avg_entry
        cash += proceeds
        trade_pnls.append(pnl)
        trades += 1
        position = 0.0
        if equity_curve:
            equity_curve[-1] = cash

    equity_series = pd.Series(equity_curve if equity_curve else [capital])
    ret_series = pd.Series(returns if returns else [0.0])
    losses = -sum(p for p in trade_pnls if p < 0)
    profit_factor = float(sum(p for p in trade_pnls if p > 0) / losses) if losses > 0 else (
        float("inf") if trade_pnls else 0.0
    )
    summary = {
        "total_return": float(equity_series.iloc[-1] / capital - 1.0),
        "max_drawdown": float(max_drawdown(equity_series)),
        "sharpe": float(sharpe_ratio(ret_series)),
        "sortino": 0.0,
        "profit_factor": profit_factor,
        "num_trades": int(trades),
    }

    return summary, ret_series.to_list(), trades


def _simulate_arbitrage(
    df: pd.DataFrame, edge_bps: float
) -> tuple[Mapping[str, float], list[float], int]:
    spot = pd.Series(df.get("spot_close", df.get("close")), dtype=float)
    futures = pd.Series(df.get("futures_close", spot * 1.001), dtype=float)
    basis = futures / spot - 1.0
    threshold = float(edge_bps) / 10_000.0
    fee = 0.0005  # 5 bps per leg
    equity = 1.0
    prev_equity = equity
    equity_curve: list[float] = []
    returns: list[float] = []
    trade_pnls: list[float] = []
    trades = 0

    for val in basis:
        val = float(val)
        ret = 0.0
        if math.isfinite(val):
            if val > threshold + 2 * fee:
                profit = max(val - 2 * fee, 0.0)
                equity += profit
                trade_pnls.append(profit)
                trades += 1
                ret = profit / prev_equity if prev_equity > 0 else 0.0
            elif val < -(threshold + 2 * fee):
                profit = max(-val - 2 * fee, 0.0)
                equity += profit
                trade_pnls.append(profit)
                trades += 1
                ret = profit / prev_equity if prev_equity > 0 else 0.0
        equity_curve.append(equity)
        returns.append(ret)
        prev_equity = equity

    equity_series = pd.Series(equity_curve) if equity_curve else pd.Series([1.0])
    ret_series = pd.Series(returns) if returns else pd.Series([0.0])
    losses = -sum(p for p in trade_pnls if p < 0)
    profit_factor = float(sum(p for p in trade_pnls if p > 0) / losses) if losses > 0 else (
        float("inf") if trade_pnls else 0.0
    )
    summary = {
        "total_return": float(equity_series.iloc[-1] - 1.0),
        "max_drawdown": float(max_drawdown(equity_series)),
        "sharpe": float(sharpe_ratio(ret_series)),
        "sortino": 0.0,
        "profit_factor": profit_factor,
        "num_trades": int(trades),
    }

    return summary, ret_series.to_list(), trades


def _aggregate_returns(evaluations: Iterable[StrategyEvaluation], attr: str) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for evaluation in evaluations:
        data = getattr(evaluation, attr, [])
        arr = np.asarray(list(data), dtype=float)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.asarray([], dtype=float)
    min_len = min(arr.size for arr in arrays)
    trimmed = [arr[-min_len:] for arr in arrays]
    return np.vstack(trimmed).mean(axis=0)


class NightlyResearchLoop:
    """Run nightly Optuna sweeps and persist the strongest performers."""

    def __init__(
        self,
        windows: Sequence[ResearchWindow],
        registry_path: Path,
        *,
        trials_range: tuple[int, int] = (20, 50),
        turnover_penalty: float = 0.05,
        drawdown_penalty: float = 2.0,
        overfit_ratio: float = 1.6,
        max_oos_drawdown: float = 0.35,
        drawdown_spike_ratio: float = 2.0,
        drawdown_spike_floor: float = 0.05,
        paper_improve_threshold: float = 0.05,
        top_n: int = 5,
        seed: int | None = None,
    ) -> None:
        if not windows:
            raise ValueError("At least one research window is required")
        self._windows = list(windows)
        self._min_trials, self._max_trials = sorted(int(x) for x in trials_range)
        self._turnover_penalty = float(turnover_penalty)
        self._drawdown_penalty = float(drawdown_penalty)
        self._overfit_ratio = float(overfit_ratio)
        self._max_oos_drawdown = float(max_oos_drawdown)
        self._dd_spike_ratio = float(max(drawdown_spike_ratio, 1.0))
        self._dd_spike_floor = float(max(drawdown_spike_floor, 0.0))
        self._paper_threshold = float(paper_improve_threshold)
        self._rng = random.Random(seed)
        self._registry = ModelRegistry(Path(registry_path), top_n=top_n)

    # ---------------------------------------------------------------- helpers
    def _evaluate_signal_strategy(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, builder
    ) -> StrategyEvaluation:
        def _simulate(df: pd.DataFrame) -> tuple[Mapping[str, float], list[float], int]:
            frame = pd.DataFrame(df).reset_index(drop=True)
            signals_df = builder(frame)
            signals = pd.Series(signals_df["signals"], dtype=float)
            close = frame["close"].astype(float)
            returns = signals.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)
            equity = (1.0 + returns).cumprod()
            summary = {
                "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
                "max_drawdown": float(max_drawdown(equity)) if not equity.empty else 0.0,
                "sharpe": float(sharpe_ratio(returns)) if returns.size else 0.0,
                "sortino": 0.0,
                "profit_factor": 0.0,
                "num_trades": int((signals.fillna(0.0).diff().abs() > 0).sum()),
            }
            pos = returns[returns > 0].sum()
            neg = -returns[returns < 0].sum()
            if neg > 0:
                summary["profit_factor"] = float(pos / neg)
            elif pos > 0:
                summary["profit_factor"] = float("inf")
            return summary, returns.to_list(), summary["num_trades"]

        train_summary, train_returns, train_trades = _simulate(df_train)
        test_summary, test_returns, test_trades = _simulate(df_test)
        return StrategyEvaluation(
            name=getattr(builder, "__name__", "strategy"),
            train_summary=train_summary,
            test_summary=test_summary,
            train_returns=train_returns,
            test_returns=test_returns,
            train_trades=train_trades,
            test_trades=test_trades,
        )

    def _evaluate_params(
        self, window: ResearchWindow, params: Mapping[str, float]
    ) -> tuple[float, Dict[str, object]]:
        evaluations: list[StrategyEvaluation] = []

        grid_eval = self._evaluate_signal_strategy(
            window.train,
            window.test,
            _grid_signals_factory(int(params["grid_levels"]), float(params["grid_range_pct"])),
        )
        evaluations.append(grid_eval)

        momentum_eval = self._evaluate_signal_strategy(
            window.train,
            window.test,
            _momentum_signals_factory(int(params["momentum_fast"]), int(params["momentum_slow"])),
        )
        evaluations.append(momentum_eval)

        dca_train_summary, dca_train_returns, dca_train_trades = _simulate_dca(
            window.train, float(params["dca_step_pct"]), int(params["dca_max_layers"])
        )
        dca_test_summary, dca_test_returns, dca_test_trades = _simulate_dca(
            window.test, float(params["dca_step_pct"]), int(params["dca_max_layers"])
        )
        evaluations.append(
            StrategyEvaluation(
                name="dca",
                train_summary=dca_train_summary,
                test_summary=dca_test_summary,
                train_returns=dca_train_returns,
                test_returns=dca_test_returns,
                train_trades=dca_train_trades,
                test_trades=dca_test_trades,
            )
        )

        arb_train_summary, arb_train_returns, arb_train_trades = _simulate_arbitrage(
            window.train, float(params["arb_edge_bps"])
        )
        arb_test_summary, arb_test_returns, arb_test_trades = _simulate_arbitrage(
            window.test, float(params["arb_edge_bps"])
        )
        evaluations.append(
            StrategyEvaluation(
                name="arbitrage",
                train_summary=arb_train_summary,
                test_summary=arb_test_summary,
                train_returns=arb_train_returns,
                test_returns=arb_test_returns,
                train_trades=arb_train_trades,
                test_trades=arb_test_trades,
            )
        )

        agg_train = _aggregate_returns(evaluations, "train_returns")
        agg_test = _aggregate_returns(evaluations, "test_returns")

        if agg_test.size == 0:
            raise optuna.TrialPruned("No evaluable out-of-sample returns")

        agg_train_sharpe = float(sharpe_ratio(pd.Series(agg_train))) if agg_train.size else 0.0
        agg_test_sharpe = float(sharpe_ratio(pd.Series(agg_test)))
        equity_train = pd.Series(np.cumprod(1.0 + agg_train)) if agg_train.size else pd.Series([1.0])
        equity_test = pd.Series(np.cumprod(1.0 + agg_test))
        agg_train_dd = float(max_drawdown(equity_train)) if agg_train.size else 0.0
        agg_dd = float(max_drawdown(equity_test))
        turnover = sum(eval.test_trades for eval in evaluations) / max(len(window.test), 1)

        if agg_test_sharpe <= 0.0:
            raise optuna.TrialPruned("Non-positive out-of-sample Sharpe")
        if abs(agg_dd) > abs(self._max_oos_drawdown):
            raise optuna.TrialPruned("Excessive out-of-sample drawdown")
        baseline_train_dd = max(abs(agg_train_dd), self._dd_spike_floor)
        if baseline_train_dd > 0 and abs(agg_dd) > baseline_train_dd * self._dd_spike_ratio:
            raise optuna.TrialPruned("Out-of-sample drawdown spike vs training")
        if agg_train_sharpe > 0 and agg_test_sharpe > 0:
            if agg_train_sharpe / max(agg_test_sharpe, 1e-6) > self._overfit_ratio:
                raise optuna.TrialPruned("Overfit signature detected")

        score = agg_test_sharpe - self._turnover_penalty * turnover - self._drawdown_penalty * abs(agg_dd)

        diagnostics: Dict[str, object] = {
            "train_sharpe": agg_train_sharpe,
            "oos_sharpe": agg_test_sharpe,
            "oos_max_drawdown": agg_dd,
            "train_max_drawdown": agg_train_dd,
            "turnover": turnover,
            "components": [
                {
                    "name": ev.name,
                    "train": dict(ev.train_summary),
                    "test": dict(ev.test_summary),
                }
                for ev in evaluations
            ],
        }
        return float(score), diagnostics

    # --------------------------------------------------------------- objective
    def _objective(self, window: ResearchWindow, trial: optuna.trial.Trial) -> float:
        params = {
            "grid_levels": trial.suggest_int("grid_levels", 5, 30),
            "grid_range_pct": trial.suggest_float("grid_range_pct", 0.01, 0.15),
            "momentum_fast": trial.suggest_int("momentum_fast", 5, 20),
            "momentum_slow": trial.suggest_int("momentum_slow", 10, 80),
            "dca_step_pct": trial.suggest_float("dca_step_pct", 0.01, 0.1),
            "dca_max_layers": trial.suggest_int("dca_max_layers", 2, 6),
            "arb_edge_bps": trial.suggest_float("arb_edge_bps", 5.0, 40.0),
        }
        # Ensure slow > fast
        if params["momentum_slow"] <= params["momentum_fast"]:
            params["momentum_slow"] = params["momentum_fast"] + 1
        score, diagnostics = self._evaluate_params(window, params)
        trial.set_user_attr("diagnostics", diagnostics)
        return score

    # ------------------------------------------------------------------ public
    def run(self, *, max_windows: int | None = None) -> list[RegistryEntry]:
        results: list[RegistryEntry] = []
        current_best = self._registry.best_score
        if math.isnan(current_best):
            current_best = float("-inf")

        for idx, window in enumerate(self._windows):
            if max_windows is not None and idx >= int(max_windows):
                break

            trial_count = self._rng.randint(self._min_trials, self._max_trials)
            study = optuna.create_study(direction="maximize")
            try:
                study.optimize(
                    lambda trial: self._objective(window, trial),
                    n_trials=trial_count,
                    gc_after_trial=True,
                    catch=(ValueError,),
                )
            except optuna.exceptions.TrialPruned:
                continue
            except ValueError:
                continue

            if not study.best_trials:
                continue
            best = study.best_trial
            score = float(best.value)
            diagnostics: MutableMapping[str, object] = dict(best.user_attrs.get("diagnostics", {}))
            entry = RegistryEntry(
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                window=window.name,
                params={k: float(v) for k, v in best.params.items()},
                score=score,
                oos_sharpe=float(diagnostics.get("oos_sharpe", float("nan"))),
                train_max_drawdown=float(diagnostics.get("train_max_drawdown", float("nan"))),
                oos_max_drawdown=float(diagnostics.get("oos_max_drawdown", float("nan"))),
                turnover=float(diagnostics.get("turnover", float("nan"))),
                diagnostics=diagnostics,
            )
            if current_best not in (float("-inf"), float("nan")) and current_best > float("-inf"):
                if score > current_best * (1.0 + self._paper_threshold):
                    entry.flag_for_paper_trial = True
            elif score > current_best:
                entry.flag_for_paper_trial = True

            self._registry.add(entry)
            current_best = max(current_best, score)
            results.append(entry)

        return results


__all__ = [
    "NightlyResearchLoop",
    "ResearchWindow",
    "ModelRegistry",
    "RegistryEntry",
    "create_non_overlapping_windows",
]

