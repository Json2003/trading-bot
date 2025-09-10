"""Signal execution engine with ATR-based exits and risk-based sizing.

Includes:
- ExecConfig dataclass for costs, exits (bps or ATR multiples), and sizing.
- run_backtest() supporting TP/SL by bps or ATR, risk-per-trade sizing, max_bars timeout.

Pandas import is guarded to avoid local pandas.py shadows in the repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _pd():
    import importlib
    # If a repo-local pandas stub is already imported, remove it so we can load real pandas
    mod = sys.modules.get('pandas')
    if mod is not None:
        mod_file = getattr(mod, '__file__', '') or ''
        try:
            if REPO_ROOT in os.path.abspath(mod_file):
                del sys.modules['pandas']
        except Exception:
            pass
    original = sys.path.copy()
    try:
        repo_paths = {p for p in original if REPO_ROOT in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        return importlib.import_module('pandas')
    finally:
        sys.path = original


@dataclass
class ExecConfig:
    # costs
    fees_bps: float = 10.0        # per fill
    slip_bps: float = 5.0         # per fill

    # exits (choose either fixed bps OR ATR multiples)
    tp_bps: float = 0.0           # 0 disables fixed-bps TP
    sl_bps: float = 0.0           # 0 disables fixed-bps SL
    tp_atr_mult: float = 0.0      # 0 disables ATR TP
    sl_atr_mult: float = 0.0      # 0 disables ATR SL
    atr_period: int = 14          # for ATR-based exits/sizing

    # risk & sizing
    notional: float = 1.0         # starting equity units
    risk_per_trade: float = 0.005 # risk % of equity per trade (e.g., 0.005 = 0.5%)
    max_notional_frac: float = 1.0# cap: position notional / equity (acts like max leverage)
    allow_short: bool = False

    # trade management
    max_bars: int = 0             # 0 = disabled; otherwise force exit after N bars
    # dynamic risk management
    break_even_atr_mult: float = 0.0  # when unrealized move >= this * ATR(entry), move stop to entry (0=disabled)
    trail_atr_mult: float = 0.0       # trail stop by this * ATR(current) from best price (0=disabled)
    trail_method: str = "atr"        # "atr" | "donchian"
    trail_ref: str = "best"          # reference for ATR trailing: "best" | "close"

    # R-multiple partial take-profit logic
    # R is defined by initial stop distance fraction (from _sl_frac_from_cfg at entry)
    tp_r_multiple: float = 1.0        # payday at k*R (e.g., 1.0–1.5R). 0 disables.
    partial_tp_frac: float = 0.5      # fraction of position to close at payday (0 disables)
    lock_in_r_after_tp: float = 0.25  # after payday, move stop to BE + X*R (0 keeps at BE)
    # Pullback/structure-based exit
    pullback_ema_len: int = 0         # 0 disables; fast structure EMA length
    pullback_atr_mult: float = 0.0    # band depth in ATRs (0 disables)
    pullback_confirm: int = 0         # bars to confirm below/above band
    donch_mid_n: int = 0              # if using Donchian trailing, midline window (0 disables)
    # Momentum guard: must achieve R multiple within N bars from entry
    min_rr_by_bars_r: float = 0.0     # e.g., 0.5 means must reach 0.5R; 0 disables
    min_rr_by_bars_n: int = 0         # bars deadline; 0 disables


def recommended_exec_config() -> ExecConfig:
    """Return an ExecConfig with sensible starting defaults for both longs/shorts.

    Notes:
    - Keeps fees/slippage/atr_period and sizing at ExecConfig defaults; override as needed.
    - Symmetric for longs/shorts via engine logic.
    """
    return ExecConfig(
        tp_r_multiple=1.2,
        partial_tp_frac=0.5,
        lock_in_r_after_tp=0.25,
        trail_method="atr",
        trail_atr_mult=1.0,
        pullback_ema_len=21,
        pullback_atr_mult=1.0,
        pullback_confirm=1,
        min_rr_by_bars_r=0.5,
        min_rr_by_bars_n=6,
        max_bars=16,
    )


def _apply_cost(price: float, fees_bps: float, slip_bps: float, side: int) -> float:
    """
    Apply fee+slip to a fill.
    side: +1 for buy (enter long), -1 for sell (enter short or exit long),
          +1 on short-exit (buy to cover), -1 on long-exit (sell)
    """
    fee  = float(price) * (fees_bps / 10_000.0)
    slip = float(price) * (slip_bps / 10_000.0) * side
    return float(price + fee + slip)


def _atr(df, period: int):
    pd = _pd()
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _sl_frac_from_cfg(entry_px: float, atr_val: float, cfg: ExecConfig) -> float:
    """Return fractional stop distance (e.g., 0.004 = 40 bps).
    Priority: sl_bps if >0, else sl_atr_mult*ATR.
    """
    if cfg.sl_bps > 0:
        return float(cfg.sl_bps) / 10_000.0
    if cfg.sl_atr_mult > 0 and atr_val > 0:
        return float((cfg.sl_atr_mult * atr_val) / entry_px)
    # fallback tiny value to avoid div-by-zero; sizing will cap via max_notional_frac
    return 0.001


def _tp_hit(entry_px: float, px: float, position: int, atr_val: float, cfg: ExecConfig) -> bool:
    if cfg.tp_bps > 0:
        tgt = float(cfg.tp_bps) / 10_000.0
        move = (px - entry_px) / entry_px if position == 1 else (entry_px - px) / entry_px
        return bool(move >= tgt)
    if cfg.tp_atr_mult > 0 and atr_val > 0:
        tgt = float((cfg.tp_atr_mult * atr_val) / entry_px)
        move = (px - entry_px) / entry_px if position == 1 else (entry_px - px) / entry_px
        return bool(move >= tgt)
    return False


def _sl_hit(entry_px: float, px: float, position: int, atr_val: float, cfg: ExecConfig) -> bool:
    if cfg.sl_bps > 0:
        thr = -float(cfg.sl_bps) / 10_000.0
        move = (px - entry_px) / entry_px if position == 1 else (entry_px - px) / entry_px
        return bool(move <= thr)
    if cfg.sl_atr_mult > 0 and atr_val > 0:
        thr = -float((cfg.sl_atr_mult * atr_val) / entry_px)
        move = (px - entry_px) / entry_px if position == 1 else (entry_px - px) / entry_px
        return bool(move <= thr)
    return False


def run_backtest(df, signals_fn: Callable[["pd.DataFrame"], "pd.DataFrame"], cfg: ExecConfig) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.Series"]:
    pd = _pd()
    data = pd.DataFrame(df).reset_index(drop=True)
    out = signals_fn(data).copy()
    if "signals" not in out.columns:
        raise ValueError("signals_fn must return a DataFrame with a 'signals' column.")

    sig = out["signals"].astype(int).values  # -1,0,1
    ts = data["timestamp"].values
    close = data["close"].astype(float).values
    high_arr = data["high"].astype(float).values
    low_arr = data["low"].astype(float).values

    # ATR for sizing/optional exits
    atr_ser = _atr(data, cfg.atr_period).fillna(0.0)
    atr = atr_ser.values
    # Optional EMA for pullback detection
    if int(getattr(cfg, 'pullback_ema_len', 0) or 0) > 0:
        ema_len = int(cfg.pullback_ema_len)
        ema_ser = data["close"].ewm(span=ema_len, adjust=False, min_periods=ema_len).mean()
        ema = ema_ser.values
    else:
        ema = None
    # Optional Donchian midline for trailing
    if getattr(cfg, 'donch_mid_n', 0) and int(cfg.donch_mid_n) > 0:
        import numpy as _np
        n = int(cfg.donch_mid_n)
        roll_high = data["high"].rolling(n, min_periods=n).max()
        roll_low = data["low"].rolling(n, min_periods=n).min()
        donch_mid = ((roll_high + roll_low) / 2.0).values
    else:
        donch_mid = None

    position = 0       # -1,0,1
    entry_price = 0.0
    f_notional = 0.0   # position notional fraction of equity (like leverage fraction)
    bars_in_trade = 0
    hit_half_R = False
    atr_entry = 0.0
    r_frac_entry = 0.0   # initial stop distance fraction (R) at entry
    stop_price = None  # dynamic stop (break-even / trailing)
    best_price = None  # peak (long) or trough (short)
    be_armed = False
    partial_taken = False
    pullback_count = 0

    equity = float(cfg.notional)
    trades = []
    equity_curve = []

    def maybe_exit(i: int, px: float, reason: str) -> bool:
        nonlocal position, entry_price, equity, f_notional, bars_in_trade
        nonlocal atr_entry, stop_price, best_price, be_armed, r_frac_entry, partial_taken, pullback_count, hit_half_R
        if position == 0:
            return False
        # closing side is opposite of position
        fill_exit = _apply_cost(px, cfg.fees_bps, cfg.slip_bps, side=-position)
        ret_frac = (fill_exit / entry_price - 1.0) * position  # price move
        # equity change scaled by notional fraction
        equity *= (1.0 + f_notional * ret_frac)
        trades.append({
            "exit_ts": ts[i],
            "side": "long" if position == 1 else "short",
            "entry_price": float(entry_price),
            "exit_price": float(fill_exit),
            "notional_frac": float(f_notional),
            "ret_price": float(ret_frac),
            "pnl": float(f_notional * ret_frac),
            "reason": reason,
            "bars": bars_in_trade,
        })
        position, entry_price, f_notional, bars_in_trade = 0, 0.0, 0.0, 0
        atr_entry, stop_price, best_price, be_armed = 0.0, None, None, False
        r_frac_entry, partial_taken = 0.0, False
        pullback_count = 0
        hit_half_R = False
        return True

    for i, px in enumerate(close):
        desired = int(sig[i])
        at = float(atr[i])

        # Evaluate dynamic stops, TP/SL/timeout if in position
        if position != 0:
            # Update best favorable price
            if best_price is None:
                best_price = px
            else:
                if position == 1:
                    best_price = max(best_price, px)
                else:
                    best_price = min(best_price, px)
            # Check min-R-by-N momentum requirement
            if (not hit_half_R) and r_frac_entry > 0.0 and int(getattr(cfg, 'min_rr_by_bars_n', 0) or 0) > 0 and float(getattr(cfg, 'min_rr_by_bars_r', 0.0) or 0.0) > 0.0:
                target_r = float(cfg.min_rr_by_bars_r)
                # price move in R units relative to entry
                move_frac = (px / entry_price - 1.0) * position
                rr = move_frac / r_frac_entry
                if rr >= target_r:
                    hit_half_R = True

            # Arm break-even when threshold reached (based on ATR at entry)
            if (not be_armed) and cfg.break_even_atr_mult > 0.0 and atr_entry > 0.0:
                move = (px - entry_price) if position == 1 else (entry_price - px)
                if move >= cfg.break_even_atr_mult * atr_entry:
                    stop_price = entry_price
                    be_armed = True

            # Apply trailing stop update by method
            if (cfg.trail_method or "atr").lower() == "atr":
                if cfg.trail_atr_mult > 0.0 and at > 0.0:
                    trail_dist = cfg.trail_atr_mult * at
                    # Choose reference price for trailing: best favorable price (default) or current close
                    ref = best_price if (getattr(cfg, 'trail_ref', 'best') or 'best').lower() == 'best' else px
                    if position == 1:
                        candidate = ref - trail_dist
                        stop_price = max(stop_price if stop_price is not None else -float('inf'), candidate)
                    else:
                        candidate = ref + trail_dist  # for shorts, stop trails above reference
                        stop_price = min(stop_price if stop_price is not None else float('inf'), candidate)
            elif (cfg.trail_method or "atr").lower() == "donchian":
                if donch_mid is not None:
                    dm_val = donch_mid[i]
                    if dm_val == dm_val:  # not NaN
                        dm = float(dm_val)
                        if position == 1:
                            stop_price = max(stop_price if stop_price is not None else -float('inf'), dm)
                        else:
                            stop_price = min(stop_price if stop_price is not None else float('inf'), dm)

            # Check dynamic stop first
            # Check partial take-profit at R-multiple (payday) before stops/TP/SL
            if (not partial_taken) and cfg.partial_tp_frac > 0.0 and cfg.tp_r_multiple > 0.0 and r_frac_entry > 0.0:
                # Compute payday price target and check intrabar extremes
                if position == 1:
                    payday_px = entry_price * (1.0 + cfg.tp_r_multiple * r_frac_entry)
                    hit = high_arr[i] >= payday_px
                else:
                    payday_px = entry_price * (1.0 - cfg.tp_r_multiple * r_frac_entry)
                    hit = low_arr[i] <= payday_px
                if hit:
                    # Execute partial exit
                    pf = float(max(0.0, min(cfg.partial_tp_frac, 1.0)))
                    if pf > 0.0 and f_notional > 0.0:
                        fill_exit = _apply_cost(payday_px, cfg.fees_bps, cfg.slip_bps, side=-position)
                        ret_frac = (fill_exit / entry_price - 1.0) * position
                        delta_f = f_notional * pf
                        equity *= (1.0 + delta_f * ret_frac)
                        trades.append({
                            "exit_ts": ts[i],
                            "side": "long" if position == 1 else "short",
                            "entry_price": float(entry_price),
                            "exit_price": float(fill_exit),
                            "notional_frac": float(delta_f),
                            "ret_price": float(ret_frac),
                            "pnl": float(delta_f * ret_frac),
                            "reason": "partial_tp",
                            "bars": bars_in_trade,
                        })
                        f_notional = float(max(0.0, f_notional - delta_f))
                        partial_taken = True
                        # Lock stop to BE + X*R (or BE-X*R for short)
                        if cfg.lock_in_r_after_tp >= 0.0:
                            lock = cfg.lock_in_r_after_tp * r_frac_entry
                            if position == 1:
                                candidate = entry_price * (1.0 + lock)
                                stop_price = max(stop_price if stop_price is not None else -float('inf'), candidate)
                            else:
                                candidate = entry_price * (1.0 - lock)
                                stop_price = min(stop_price if stop_price is not None else float('inf'), candidate)
                            be_armed = True

            # Optional pullback-confirm exit using EMA +/- k*ATR bands with stateful count
            if (ema is not None) and cfg.pullback_atr_mult > 0.0 and int(cfg.pullback_confirm) > 0 and position != 0:
                k = float(cfg.pullback_atr_mult)
                need = int(cfg.pullback_confirm)
                e = ema[i]
                if e == e:  # not NaN
                    band = (e - k * at) if position == 1 else (e + k * at)
                    cond = (px <= band) if position == 1 else (px >= band)
                    pullback_count = pullback_count + 1 if cond else 0
                    if pullback_count >= need:
                        maybe_exit(i, px, reason="real_pullback")
                else:
                    pullback_count = 0

            # Check dynamic stop first
            if stop_price is not None:
                if (position == 1 and px <= stop_price) or (position == -1 and px >= stop_price):
                    if maybe_exit(i, px, reason="trail_stop"):
                        pass
            # Then TP / SL / timeout
            if position != 0:
                if _tp_hit(entry_price, px, position, at, cfg):
                    maybe_exit(i, px, reason="tp")
                elif _sl_hit(entry_price, px, position, at, cfg):
                    maybe_exit(i, px, reason="sl")
                elif cfg.max_bars > 0 and bars_in_trade >= int(cfg.max_bars):
                    maybe_exit(i, px, reason="timeout")
                # Momentum timeout: didn’t hit target R within N bars from entry
                elif (int(getattr(cfg, 'min_rr_by_bars_n', 0) or 0) > 0) and (bars_in_trade >= int(cfg.min_rr_by_bars_n)) and (not hit_half_R):
                    maybe_exit(i, px, reason="timebox_missed_halfR")

        # Flip/flat exit
        if position != 0 and ((desired == 0) or (desired != position)):
            maybe_exit(i, px, reason="flip")

        # Entry (after exits)
        if position == 0 and desired != 0 and (desired != -1 or cfg.allow_short):
            fill_entry = _apply_cost(px, cfg.fees_bps, cfg.slip_bps, side=desired)
            # compute stop distance fraction for sizing
            sl_frac = _sl_frac_from_cfg(fill_entry, at, cfg)
            sl_frac = max(sl_frac, 1e-5)
            # risk-based fraction of equity deployed
            f = float(cfg.risk_per_trade) / sl_frac
            f = float(min(max(f, 0.0), float(cfg.max_notional_frac)))
            entry_price = float(fill_entry)
            entry_price = float(fill_entry)
            position = desired
            f_notional = f
            bars_in_trade = 0
            atr_entry = at
            r_frac_entry = sl_frac
            stop_price = None
            best_price = px
            be_armed = False
            partial_taken = False
            pullback_count = 0
            hit_half_R = False

        # progress time in trade
        if position != 0:
            bars_in_trade += 1

        equity_curve.append({"timestamp": ts[i], "equity": float(equity)})

    # Force-close at end if still in position
    if position != 0:
        maybe_exit(len(close) - 1, float(close[-1]), reason="eod")

    eq = pd.DataFrame(equity_curve)
    eq["equity_prev"] = eq["equity"].shift(1).fillna(eq["equity"].iloc[0])
    bar_returns = eq["equity"] / eq["equity_prev"] - 1.0

    return pd.DataFrame(trades), eq[["timestamp", "equity"]], bar_returns
