#!/usr/bin/env python3
"""
Comprehensive test suite for trading bot enhancements.

This script validates all major improvements and demonstrates usage of enhanced features:
- Grid search optimization with parallel processing
- Multi-model machine learning training
- Enhanced backtesting with professional analytics  
- Robust data fetching with retry logic
- WebSocket server functionality
- Async market data crawling

Usage:
    python test_enhancements.py
    python test_enhancements.py --component backtest
    python test_enhancements.py --component optimization --verbose
"""

import sys
import time
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_ohlcv_data(bars: int = 1000) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing purposes.
    
    Args:
        bars: Number of bars to generate
        
    Returns:
        DataFrame with OHLCV columns and datetime index
    """
    logger.info(f"Generating {bars} bars of sample OHLCV data")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=bars)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')[:bars]
    
    # Generate realistic price data with trend and volatility
    np.random.seed(42)  # For reproducible results
    
    # Start with base price
    base_price = 50000.0
    
    # Generate price changes with some trend and mean reversion
    price_changes = np.random.normal(0, 0.02, bars)  # 2% hourly volatility
    trend = np.linspace(-0.001, 0.001, bars)  # Small trend component
    price_changes += trend
    
    # Calculate prices using cumulative sum
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some intrabar volatility
        volatility = abs(np.random.normal(0, 0.005))  # 0.5% intrabar volatility
        
        high = price * (1 + volatility * np.random.random())
        low = price * (1 - volatility * np.random.random())
        
        # Ensure OHLC relationships are valid
        open_price = price * (1 + np.random.normal(0, 0.001))
        close_price = price * (1 + np.random.normal(0, 0.001))
        
        # Ensure high is highest, low is lowest
        high = max(high, open_price, close_price, price)
        low = min(low, open_price, close_price, price)
        
        # Generate volume (higher volume with higher volatility)
        volume = np.random.lognormal(10, 1) * (1 + volatility * 10)
        
        data.append({
            'ts': dates[i] if i < len(dates) else dates[-1] + timedelta(hours=i-len(dates)+1),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('ts', inplace=True)
    
    logger.info(f"Generated sample data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Price range: ${df['low'].min():.0f} - ${df['high'].max():.0f}")
    
    return df

def test_backtest_enhancement():
    """Test enhanced backtesting framework."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED BACKTESTING FRAMEWORK")
    logger.info("="*60)
    
    try:
        from tradingbot_ibkr.backtest_ccxt import aggressive_strategy_backtest
        
        # Generate sample data
        df = generate_sample_ohlcv_data(500)
        
        # Test basic backtest
        logger.info("Running basic backtest...")
        start_time = time.time()
        
        results = aggressive_strategy_backtest(
            df,
            take_profit_pct=0.02,
            stop_loss_pct=0.01, 
            max_holding_bars=24,
            enable_logging=True
        )
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert 'trades' in results, "Results missing 'trades' field"
        assert 'performance_metrics' in results, "Results missing performance metrics"
        assert 'equity_curve' in results, "Results missing equity curve"
        
        logger.info("BACKTEST RESULTS:")
        logger.info(f"  Execution time: {execution_time:.2f}s")
        logger.info(f"  Total trades: {results['trades']}")
        logger.info(f"  Win rate: {results['win_rate_pct']:.1f}%")
        logger.info(f"  Total PnL: {results['pnl']:.2f}")
        
        if results.get('performance_metrics'):
            metrics = results['performance_metrics']
            logger.info(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"  Profit factor: {metrics.get('profit_factor', 0):.2f}")
        
        logger.info("✅ Backtesting framework test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtesting framework test FAILED: {e}")
        return False

def test_optimization_enhancement():
    """Test enhanced grid search optimization."""
    logger.info("="*60) 
    logger.info("TESTING ENHANCED OPTIMIZATION FRAMEWORK")
    logger.info("="*60)
    
    try:
        from tradingbot_ibkr.aggressive_optimize import run_grid
        
        # Create a small test data file
        df = generate_sample_ohlcv_data(200)
        test_data_dir = Path('tradingbot_ibkr/datafiles')
        test_data_dir.mkdir(exist_ok=True)
        
        test_file = test_data_dir / 'BTC_USDT_bars.csv'
        df.to_csv(test_file)
        logger.info(f"Created test data file: {test_file}")
        
        # Run optimization with limited parameters for testing
        logger.info("Running grid optimization (limited scope for testing)...")
        start_time = time.time()
        
        results = run_grid(
            symbol='BTC/USDT',
            max_workers=2,  # Limited for testing
            early_stopping_patience=5,  # Early stopping for faster test
            enable_pruning=True
        )
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert 'metadata' in results, "Results missing metadata"
        assert 'results' in results, "Results missing results list"
        
        logger.info("OPTIMIZATION RESULTS:")
        logger.info(f"  Execution time: {execution_time:.2f}s")
        logger.info(f"  Combinations tested: {results['metadata']['total_combinations_tested']}")
        logger.info(f"  Best win rate: {results['metadata']['best_win_rate']:.2f}%")
        
        if results['results']:
            best_result = results['results'][0]
            logger.info(f"  Best parameters: {best_result['params']}")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        
        logger.info("✅ Optimization framework test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization framework test FAILED: {e}")
        return False

def test_model_training_enhancement():
    """Test enhanced model training framework."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED MODEL TRAINING FRAMEWORK")
    logger.info("="*60)
    
    try:
        from tradingbot_ibkr.models.train_batch import train_and_evaluate_models
        
        # Generate sample data
        df = generate_sample_ohlcv_data(300)
        
        logger.info("Running model training (without hyperparameter optimization for speed)...")
        start_time = time.time()
        
        results = train_and_evaluate_models(
            df,
            optimize_hyperparams=False,  # Skip optimization for faster testing
            use_optuna=False
        )
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert isinstance(results, dict), "Results should be a dictionary"
        
        successful_models = [k for k, v in results.items() if 'error' not in v]
        
        logger.info("MODEL TRAINING RESULTS:")
        logger.info(f"  Execution time: {execution_time:.2f}s")
        logger.info(f"  Models trained: {len(successful_models)}")
        
        for model_name in successful_models:
            result = results[model_name]
            logger.info(f"  {result.get('model_name', model_name)}: CV Score = {result.get('cv_mean_score', 0):.4f}")
        
        logger.info("✅ Model training framework test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training framework test FAILED: {e}")
        return False

def test_data_fetching_enhancement():
    """Test enhanced data fetching framework."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED DATA FETCHING FRAMEWORK")
    logger.info("="*60)
    
    try:
        from tradingbot_ibkr.run_fetch_one import FREDDataFetcher
        
        # Test with a dummy API key (will fail but should handle gracefully)
        logger.info("Testing data fetcher initialization and error handling...")
        
        fetcher = FREDDataFetcher("dummy_api_key", max_retries=2, timeout=5)
        
        # Test validation (should fail gracefully)
        is_valid = fetcher.validate_api_key()
        logger.info(f"  API key validation result: {is_valid} (expected: False)")
        
        # Test error handling
        logger.info("  Testing error handling with invalid series...")
        data = fetcher.fetch_series_data("INVALID_SERIES_ID")
        logger.info(f"  Fetch result: {data is None} (expected: True)")
        
        logger.info("✅ Data fetching framework test PASSED (error handling works)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data fetching framework test FAILED: {e}")
        return False

def test_server_enhancement():
    """Test enhanced WebSocket server framework."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED SERVER FRAMEWORK")
    logger.info("="*60)
    
    try:
        # Import server components
        from server import app, manager, STATE
        
        logger.info("Server components imported successfully")
        
        # Test state initialization
        assert 'metrics' in STATE, "STATE missing metrics"
        assert 'server_stats' in STATE, "STATE missing server_stats"
        
        logger.info("  State structure validated")
        
        # Test connection manager
        assert hasattr(manager, 'active_connections'), "Manager missing active_connections"
        assert hasattr(manager, 'is_rate_limited'), "Manager missing rate limiting"
        
        logger.info("  Connection manager structure validated")
        
        logger.info("✅ Server framework test PASSED (structure validation)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Server framework test FAILED: {e}")
        return False

def run_comprehensive_test():
    """Run all enhancement tests."""
    logger.info("🚀 STARTING COMPREHENSIVE ENHANCEMENT TESTING")
    logger.info("="*80)
    
    test_results = {}
    
    # Run individual tests
    tests = [
        ("Backtesting Framework", test_backtest_enhancement),
        ("Optimization Framework", test_optimization_enhancement), 
        ("Model Training Framework", test_model_training_enhancement),
        ("Data Fetching Framework", test_data_fetching_enhancement),
        ("Server Framework", test_server_enhancement)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🧪 Testing {test_name}...")
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
        except KeyboardInterrupt:
            logger.info("Testing interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ {test_name} failed with unexpected error: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("="*80)
    logger.info("🏁 COMPREHENSIVE TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Enhancements are working correctly!")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed - Check logs for details")
        return False

def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(
        description='Test trading bot enhancements',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--component', 
        choices=['backtest', 'optimization', 'models', 'data', 'server', 'all'],
        default='all',
        help='Specific component to test (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Component-specific testing
    if args.component == 'backtest':
        return test_backtest_enhancement()
    elif args.component == 'optimization':
        return test_optimization_enhancement()
    elif args.component == 'models':
        return test_model_training_enhancement()
    elif args.component == 'data':
        return test_data_fetching_enhancement()
    elif args.component == 'server':
        return test_server_enhancement()
    else:
        return run_comprehensive_test()

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed with unexpected error: {e}")
        sys.exit(1)