#!/usr/bin/env python3
"""
Test script for new trading strategies

Tests all implemented strategies without external dependencies.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def create_test_data(periods=100):
    """Create test OHLCV data for validation"""
    import random
    
    data = []
    price = 50000.0
    
    for i in range(periods):
        # Simple price evolution
        change = random.gauss(0, 0.015)  # 1.5% volatility
        price = price * (1 + change)
        
        # Create OHLCV
        high = price * (1 + abs(random.gauss(0, 0.008)))
        low = price * (1 - abs(random.gauss(0, 0.008)))
        open_price = price * (1 + random.gauss(0, 0.003))
        volume = random.uniform(100, 500)
        
        data.append({
            'timestamp': datetime.now() + timedelta(hours=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
        
    return data

def test_base_strategy():
    """Test base strategy class"""
    try:
        from strategies.base import BaseStrategy, MarketData, Trade
        
        strategy = BaseStrategy("Test Strategy")
        print("✓ Base strategy class works")
        return True
    except Exception as e:
        print(f"✗ Base strategy failed: {e}")
        return False

def test_amm_strategy():
    """Test AMM strategy"""
    try:
        from strategies.amm_strategy import amm_strategy_backtest
        
        data = create_test_data(50)
        result = amm_strategy_backtest(data)
        
        print(f"✓ AMM Strategy: {result['trades']} trades, PnL: {result['pnl']:.2f}")
        return True
    except Exception as e:
        print(f"✗ AMM strategy failed: {e}")
        return False

def test_multipool_strategy():
    """Test multipool staking strategy"""
    try:
        from strategies.multipool_strategy import multipool_staking_backtest
        
        data = create_test_data(30)  # Shorter for staking test
        result = multipool_staking_backtest(data)
        
        print(f"✓ Multipool Staking: {result['trades']} transactions, Rewards: {result['total_rewards']:.2f}")
        return True
    except Exception as e:
        print(f"✗ Multipool staking failed: {e}")
        return False

def test_intelligent_strategies():
    """Test intelligent trading strategies"""
    try:
        from strategies.intelligent_strategies import (
            momentum_strategy_backtest,
            mean_reversion_strategy_backtest, 
            arbitrage_strategy_backtest
        )
        
        data = create_test_data(80)
        
        # Test momentum
        momentum_result = momentum_strategy_backtest(data)
        print(f"✓ Momentum Strategy: {momentum_result['trades']} trades, PnL: {momentum_result['pnl']:.2f}")
        
        # Test mean reversion
        reversion_result = mean_reversion_strategy_backtest(data)
        print(f"✓ Mean Reversion: {reversion_result['trades']} trades, PnL: {reversion_result['pnl']:.2f}")
        
        # Test arbitrage
        arbitrage_result = arbitrage_strategy_backtest(data)
        print(f"✓ Arbitrage Strategy: {arbitrage_result['trades']} trades, PnL: {arbitrage_result['pnl']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Intelligent strategies failed: {e}")
        return False

def test_advanced_integration():
    """Test the advanced strategies integration"""
    try:
        from advanced_strategies import run_all_strategies_backtest
        
        data = create_test_data(60)
        results = run_all_strategies_backtest(data)
        
        if 'error' in results:
            print(f"✗ Integration error: {results['error']}")
            return False
            
        strategies_tested = [k for k in results.keys() if k != 'comparison']
        print(f"✓ Integration test: {len(strategies_tested)} strategies tested")
        
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_configuration():
    """Test strategy configuration loading"""
    try:
        import json
        with open('strategies_config.json', 'r') as f:
            config = json.load(f)
            
        strategies = config.get('strategies', {})
        print(f"✓ Configuration: {len(strategies)} strategies configured")
        
        # Verify all expected strategies are configured
        expected = ['amm', 'multipool_staking', 'momentum', 'mean_reversion', 'arbitrage']
        missing = [s for s in expected if s not in strategies]
        if missing:
            print(f"! Missing configurations: {missing}")
        else:
            print("✓ All strategies have configurations")
            
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing New Trading Strategies Implementation")
    print("=" * 60)
    
    tests = [
        ("Base Strategy Class", test_base_strategy),
        ("AMM Strategy", test_amm_strategy),
        ("Multipool Staking", test_multipool_strategy), 
        ("Intelligent Strategies", test_intelligent_strategies),
        ("Integration Module", test_advanced_integration),
        ("Configuration Loading", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
        else:
            print("  (This test failed - check implementation)")
            
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! New strategies are ready to use.")
        print("\nTo run backtests:")
        print("  python advanced_strategies.py")
        print("  python backtest_ccxt.py")
    else:
        print("✗ Some tests failed. Please check the implementations.")
        
    print("=" * 60)

if __name__ == '__main__':
    main()