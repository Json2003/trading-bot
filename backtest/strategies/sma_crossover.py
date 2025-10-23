"""
Enhanced SMA crossover strategies compatible with the backtest framework.
These are the signal generators from the original PR #28.
"""

from typing import Any, Optional


def generate_signals(df: Any) -> Any:
    """
    Generate trading signals based on moving average crossover.
    
    This is the original strategy from PR #28 that uses fast (20-period) 
    and slow (60-period) simple moving averages.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with 'signals' column (1 for long, 0 for no position)
    """
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(20, min_periods=20).mean()
    out["sma_slow"] = out["close"].rolling(60, min_periods=60).mean()
    out["signals"] = 0
    out.loc[out["sma_fast"] > out["sma_slow"], "signals"] = 1
    return out[["signals"]]


def generate_enhanced_signals(df: Any, 
                            fast_period: int = 20, 
                            slow_period: int = 60,
                            rsi_period: int = 14,
                            rsi_oversold: float = 30,
                            rsi_overbought: float = 70,
                            volume_threshold: float = 1.2) -> Any:
    """
    Enhanced signal generation with additional filters.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast moving average period
        slow_period: Slow moving average period  
        rsi_period: RSI calculation period
        rsi_oversold: RSI oversold threshold
        rsi_overbought: RSI overbought threshold
        volume_threshold: Volume multiplier vs average
        
    Returns:
        DataFrame with signals and indicator columns
    """
    import pandas as pd
    import numpy as np
    
    out = df.copy()
    
    # Moving averages
    out["sma_fast"] = out["close"].rolling(fast_period, min_periods=fast_period).mean()
    out["sma_slow"] = out["close"].rolling(slow_period, min_periods=slow_period).mean()
    
    # RSI calculation
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    out["rsi"] = 100 - (100 / (1 + rs))
    
    # Volume filter (if volume column exists)
    if "volume" in out.columns:
        out["volume_ma"] = out["volume"].rolling(20).mean()
        out["volume_ratio"] = out["volume"] / out["volume_ma"]
        volume_filter = out["volume_ratio"] > volume_threshold
    else:
        volume_filter = True
    
    # Basic MA crossover signal
    ma_signal = (out["sma_fast"] > out["sma_slow"]).astype(int)
    
    # Apply filters
    rsi_filter = (out["rsi"] > rsi_oversold) & (out["rsi"] < rsi_overbought)
    
    # Combined signal (all conditions must be true for long signal)
    out["signals"] = ma_signal & rsi_filter & volume_filter
    out["signals"] = out["signals"].astype(int)
    
    return out[["signals"]]


def generate_breakout_signals(df: Any,
                            lookback_period: int = 20,
                            breakout_threshold: float = 1.02) -> Any:
    """
    Generate breakout signals based on rolling high/low.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_period: Period for rolling high/low calculation
        breakout_threshold: Multiplier for breakout confirmation
        
    Returns:
        DataFrame with breakout signals
    """
    out = df.copy()
    
    # Calculate rolling highs and lows
    out["rolling_high"] = out["high"].rolling(lookback_period).max()
    out["rolling_low"] = out["low"].rolling(lookback_period).min()
    
    # Breakout signals
    breakout_long = out["close"] > (out["rolling_high"] * breakout_threshold)
    
    # For now, only long signals
    out["signals"] = breakout_long.astype(int)
    
    return out[["signals"]]