#!/usr/bin/env python3
"""
Test script to verify the signal generation and backtesting functionality.
"""

import sys
import os
import tempfile
import types
import importlib
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Add repo to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

scripts_dir = repo_root / "scripts"
import_third_party = importlib.import_module  # fallback default
run_backtest_module = None

# Load run_backtest module
scripts_path = str(scripts_dir)
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

spec = importlib.util.spec_from_file_location("run_backtest", scripts_dir / "run_backtest.py")
if spec and spec.loader:
    run_backtest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_backtest_module)
    if hasattr(run_backtest_module, "import_third_party"):
        import_third_party = run_backtest_module.import_third_party

    scripts_pkg = sys.modules.get("scripts")
    if scripts_pkg is None:
        scripts_pkg = types.ModuleType("scripts")
        scripts_pkg.__path__ = [scripts_path]
        sys.modules["scripts"] = scripts_pkg
    sys.modules["scripts.run_backtest"] = run_backtest_module

    try:
        sys.modules['requests'] = import_third_party('requests')
    except ImportError:
        pass

def create_synthetic_data(bars=100, trend_up=True):
    """Create synthetic OHLCV data for testing."""
    # Start with a base price
    base_price = 50000.0
    dates = pd.date_range(start='2024-01-01', periods=bars, freq='h')
    
    # Generate trending price data
    if trend_up:
        # Upward trend with some noise
        trend = np.linspace(0, 0.2, bars)  # 20% total increase
        noise = np.random.normal(0, 0.01, bars)  # 1% noise
        price_multiplier = 1 + trend + noise
    else:
        # Sideways with some oscillation
        oscillation = 0.05 * np.sin(np.linspace(0, 4*np.pi, bars))
        noise = np.random.normal(0, 0.005, bars)
        price_multiplier = 1 + oscillation + noise
    
    # Calculate OHLCV
    close_prices = base_price * price_multiplier
    
    # Generate OHLC with some realistic patterns
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    high_multiplier = 1 + np.abs(np.random.normal(0, 0.005, bars))
    low_multiplier = 1 - np.abs(np.random.normal(0, 0.005, bars))
    
    high_prices = np.maximum(open_prices, close_prices) * high_multiplier
    low_prices = np.minimum(open_prices, close_prices) * low_multiplier
    
    volumes = np.random.uniform(10, 100, bars)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return df

def test_signal_generation():
    """Test the signal generation functions."""
    print("Testing signal generation...")
    
    from tradingbot_ibkr.signal_generators import generate_signals, generate_enhanced_signals, generate_breakout_signals
    
    # Create upward trending data (should generate SMA crossover signals)
    df = create_synthetic_data(bars=200, trend_up=True)
    
    # Test basic SMA crossover
    signals = generate_signals(df)
    signal_count = signals['signals'].sum()
    signal_pct = signals['signals'].mean() * 100
    
    print(f"SMA Crossover Signals:")
    print(f"  Total signals: {signal_count}")
    print(f"  Signal percentage: {signal_pct:.2f}%")
    
    # Test enhanced signals
    enhanced = generate_enhanced_signals(df)
    enhanced_count = enhanced['signals'].sum()
    enhanced_pct = enhanced['signals'].mean() * 100
    
    print(f"Enhanced Signals:")
    print(f"  Total signals: {enhanced_count}")
    print(f"  Signal percentage: {enhanced_pct:.2f}%")
    
    # Test breakout signals
    breakout = generate_breakout_signals(df, lookback_period=10)
    breakout_count = breakout['signals'].sum()
    breakout_pct = breakout['signals'].mean() * 100
    
    print(f"Breakout Signals:")
    print(f"  Total signals: {breakout_count}")
    print(f"  Signal percentage: {breakout_pct:.2f}%")
    
    return signal_count > 0 or enhanced_count > 0 or breakout_count > 0

def test_backtest_with_signals():
    """Test the backtesting with signal generation."""
    print("\nTesting backtest with signal generation...")
    
    # Create CSV file with synthetic data
    df = create_synthetic_data(bars=200, trend_up=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df_reset = df.reset_index().rename(columns={'index': 'ts'})
        df_reset.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        if run_backtest_module is None:
            raise RuntimeError("run_backtest module unavailable")

        sys.modules['requests'] = import_third_party('requests')
        parser = run_backtest_module.build_parser()
        
        # Create arguments for CSV backtest with SMA cross strategy
        args = parser.parse_args([
            '--source', 'csv',
            '--path', csv_path,
            '--strategy', 'sma_cross',
            '--tp', '0.02',
            '--sl', '0.01',
            '--hold', '10'
        ])
        
        # Run backtest
        results = run_backtest_module.run_backtest(args)
        
        print(f"Backtest results:")
        print(f"  Total trades: {results.get('total_trades', 0)}")
        print(f"  Win rate: {results.get('win_rate_pct', 0):.2f}%")
        print(f"  PnL: {results.get('pnl', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists(csv_path):
            os.unlink(csv_path)

def test_electron_file_paths():
    """Test that Electron app files exist and are properly structured."""
    print("\nTesting Electron app file structure...")
    
    electron_dir = repo_root / 'dashboard' / 'electron-app'
    files_to_check = [
        'main.js',
        'package.json',
        'preload.js',
        'renderer/index.html'
    ]
    
    all_exist = True
    for file in files_to_check:
        path = electron_dir / Path(file)
        exists = path.exists()
        print(f"  {file}: {'✅' if exists else '❌'}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("  All Electron files present - 'Cannot GET /' issue likely resolved")
    
    return all_exist

def main():
    """Run all tests."""
    print("="*60)
    print("TRADING BOT FUNCTIONALITY TEST")
    print("="*60)
    
    tests = [
        ("Signal Generation", test_signal_generation),
        ("Backtest with Signals", test_backtest_with_signals),
        ("Electron File Structure", test_electron_file_paths)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print(f"\n{name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
    
    overall_pass = all(result for _, result in results)
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    
    return overall_pass

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
