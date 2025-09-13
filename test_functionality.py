#!/usr/bin/env python3
"""
Test script to verify the signal generation and backtesting functionality.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Add repo to path
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(repo_root)

def create_synthetic_data(bars=100, trend_up=True):
    """Create synthetic OHLCV data for testing."""
    # Start with a base price
    base_price = 50000.0
    dates = pd.date_range(start='2024-01-01', periods=bars, freq='H')
    
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
    
    try:
        # Test the backtest strategies
        from backtest.strategies.sma_crossover import generate_signals, generate_enhanced_signals, generate_breakout_signals
        
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
        
    except Exception as e:
        print(f"Signal generation test failed: {e}")
        return False

def test_backtest_integration():
    """Test the backtesting with signal generation."""
    print("\nTesting backtest integration with strategies...")
    
    try:
        # Create CSV file with synthetic data
        df = create_synthetic_data(bars=200, trend_up=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write with timestamp column for the backtest script
            df.reset_index().rename(columns={'index': 'timestamp'}).to_csv(f.name, index=False)
            csv_path = f.name
        
        # Test with the existing run_backtest.py script using our new strategy
        import subprocess
        cmd = [
            sys.executable, os.path.join(repo_root, 'scripts', 'run_backtest.py'),
            '--source', 'csv',
            '--path', csv_path,
            '--strategy', 'backtest.strategies.sma_crossover:generate_signals'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Backtest integration successful!")
            print("Output:", result.stdout[-200:])  # Last 200 chars
            return True
        else:
            print(f"Backtest failed with return code {result.returncode}")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Backtest integration test failed: {e}")
        return False
        
    finally:
        # Clean up temp file
        if 'csv_path' in locals() and os.path.exists(csv_path):
            os.unlink(csv_path)

def test_electron_file_paths():
    """Test that Electron app files exist and are properly structured."""
    print("\nTesting Electron app file structure...")
    
    electron_dir = os.path.join(repo_root, 'dashboard', 'electron-app')
    files_to_check = [
        'main.js',
        'package.json',
        'preload.js',
        'renderer/index.html'
    ]
    
    all_exist = True
    for file in files_to_check:
        path = os.path.join(electron_dir, file)
        exists = os.path.exists(path)
        print(f"  {file}: {'✅' if exists else '❌'}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("  All Electron files present - 'Cannot GET /' issue likely resolved")
        
        # Check if main.js has the fix
        main_js_path = os.path.join(electron_dir, 'main.js')
        try:
            with open(main_js_path, 'r') as f:
                content = f.read()
                has_path_join = 'path.join(__dirname' in content
                has_error_handling = '.catch(err' in content
                print(f"  main.js has path.join fix: {'✅' if has_path_join else '❌'}")
                print(f"  main.js has error handling: {'✅' if has_error_handling else '❌'}")
                return all_exist and has_path_join and has_error_handling
        except Exception as e:
            print(f"  Error reading main.js: {e}")
            return False
    
    return all_exist

def main():
    """Run all tests."""
    print("="*60)
    print("TRADING BOT FUNCTIONALITY TEST")
    print("="*60)
    
    tests = [
        ("Signal Generation", test_signal_generation),
        ("Backtest Integration", test_backtest_integration), 
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