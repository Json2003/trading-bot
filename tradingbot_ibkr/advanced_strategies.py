"""
Advanced Trading Strategies Integration

This module integrates the new trading strategies with the existing backtest framework.
Provides unified interface for running multiple strategies and comparing performance.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta

# Import new strategies
try:
    from strategies.amm_strategy import AMMStrategy, amm_strategy_backtest
    from strategies.multipool_strategy import MultipoolStakingStrategy, multipool_staking_backtest
    from strategies.intelligent_strategies import (
        MomentumStrategy, MeanReversionStrategy, ArbitrageStrategy,
        momentum_strategy_backtest, mean_reversion_strategy_backtest, arbitrage_strategy_backtest
    )
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    STRATEGIES_AVAILABLE = False


def create_sample_data(periods: int = 1000):
    """Create sample OHLCV data for testing when real data is not available"""
    import random
    import math
    
    data = []
    base_price = 50000.0  # Starting BTC price
    
    for i in range(periods):
        # Create more realistic price action with trends and reversals
        if i < periods // 3:
            # Uptrend phase
            trend = 0.001
        elif i < 2 * periods // 3:
            # Sideways/volatile phase  
            trend = 0.0
        else:
            # Downtrend phase
            trend = -0.0008
            
        # Random walk with trend and volatility
        change_pct = trend + random.gauss(0, 0.025)  # 2.5% daily volatility
        base_price *= (1 + change_pct)
        
        # Create OHLCV bar with realistic intrabar movement
        volatility = abs(random.gauss(0, 0.015))
        high = base_price * (1 + volatility)
        low = base_price * (1 - volatility)
        
        # Open price from previous close (with gap)
        gap = random.gauss(0, 0.005)
        open_price = base_price * (1 + gap)
        
        # Close is the base price
        close = base_price
        
        # Volume with realistic patterns (higher on moves)
        base_volume = 200
        volume_multiplier = 1 + 2 * abs(change_pct)  # Higher volume on big moves
        volume = base_volume * volume_multiplier * random.uniform(0.5, 2.0)
        
        timestamp = datetime.now() + timedelta(hours=i)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low, 
            'close': close,
            'volume': volume
        })
        
    return data


def run_all_strategies_backtest(df_data=None, config: Dict[str, Any] = None):
    """
    Run backtests for all available strategies and compare performance
    
    Args:
        df_data: OHLCV data (if None, will generate sample data)
        config: Configuration dict for all strategies
        
    Returns:
        Dictionary with results from all strategies
    """
    if not STRATEGIES_AVAILABLE:
        return {'error': 'Strategy modules not available - import errors'}
        
    # Create sample data if none provided
    if df_data is None:
        sample_data = create_sample_data(500)  # 500 periods
        df_data = sample_data
        
    # Default configurations for each strategy
    if config is None:
        config = {
            'amm': {
                'starting_balance': 10000.0,
                'spread_bps': 20,
                'order_size': 100.0,
                'max_inventory': 500.0,
                'fee_pct': 0.001
            },
            'multipool_staking': {
                'starting_balance': 10000.0,
                'rebalance_threshold': 0.1,
                'max_pool_allocation': 0.4,
                'risk_tolerance': 5,
                'pools': [
                    {'name': 'BTC_Staking', 'apy': 4.5, 'risk_score': 3},
                    {'name': 'ETH_Staking', 'apy': 6.0, 'risk_score': 4},
                    {'name': 'USDC_Lending', 'apy': 8.0, 'risk_score': 2},
                    {'name': 'DeFi_Pool', 'apy': 15.0, 'risk_score': 7}
                ]
            },
            'momentum': {
                'starting_balance': 10000.0,
                'lookback_period': 10,  # Shorter lookback
                'momentum_threshold': 0.015,  # Lower threshold
                'position_size_pct': 0.1
            },
            'mean_reversion': {
                'starting_balance': 10000.0,
                'lookback_period': 15,  # Shorter lookback
                'std_dev_threshold': 1.5,  # Lower threshold
                'position_size_pct': 0.15
            },
            'arbitrage': {
                'starting_balance': 10000.0,
                'min_profit_threshold': 0.003,
                'max_position_size': 1000.0
            }
        }
    
    results = {}
    
    # Run each strategy backtest
    strategies_to_test = [
        ('amm', amm_strategy_backtest),
        ('multipool_staking', multipool_staking_backtest), 
        ('momentum', momentum_strategy_backtest),
        ('mean_reversion', mean_reversion_strategy_backtest),
        ('arbitrage', arbitrage_strategy_backtest)
    ]
    
    for strategy_name, backtest_func in strategies_to_test:
        try:
            print(f"Running {strategy_name} strategy backtest...")
            strategy_config = config.get(strategy_name, {})
            result = backtest_func(df_data, strategy_config)
            results[strategy_name] = result
            print(f"✓ {strategy_name}: {result.get('trades', 0)} trades, PnL: {result.get('pnl', 0):.2f}")
        except Exception as e:
            print(f"✗ {strategy_name} failed: {str(e)}")
            results[strategy_name] = {'error': str(e)}
            
    # Add comparison metrics
    results['comparison'] = create_strategy_comparison(results)
    
    return results


def create_strategy_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create comparison metrics across all strategies"""
    comparison = {
        'summary': {},
        'rankings': {},
        'risk_metrics': {}
    }
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        return comparison
        
    # Extract key metrics for comparison
    metrics = {}
    for strategy, result in valid_results.items():
        pnl = result.get('pnl', 0)
        trades = result.get('trades', 0)
        win_rate = result.get('win_rate_pct', 0)
        
        metrics[strategy] = {
            'pnl': pnl,
            'trades': trades, 
            'win_rate': win_rate,
            'pnl_per_trade': pnl / trades if trades > 0 else 0,
            'expected_return': result.get('apy_achieved', win_rate * 0.1) if 'apy_achieved' in result else win_rate * 0.1
        }
        
    # Create rankings
    for metric in ['pnl', 'win_rate', 'pnl_per_trade', 'expected_return']:
        sorted_strategies = sorted(metrics.items(), 
                                 key=lambda x: x[1][metric], 
                                 reverse=True)
        comparison['rankings'][f'best_{metric}'] = [s[0] for s in sorted_strategies]
        
    # Summary statistics
    if metrics:
        comparison['summary'] = {
            'total_strategies': len(metrics),
            'avg_pnl': sum(m['pnl'] for m in metrics.values()) / len(metrics),
            'avg_win_rate': sum(m['win_rate'] for m in metrics.values()) / len(metrics),
            'best_performing': comparison['rankings']['best_pnl'][0] if comparison['rankings'].get('best_pnl') else None,
            'most_consistent': comparison['rankings']['best_win_rate'][0] if comparison['rankings'].get('best_win_rate') else None
        }
        
    return comparison


def get_strategy_recommendations(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Provide strategy recommendations based on backtest results"""
    recommendations = []
    
    comparison = results.get('comparison', {})
    summary = comparison.get('summary', {})
    
    if not summary:
        return recommendations
        
    best_pnl = summary.get('best_performing')
    most_consistent = summary.get('most_consistent')
    
    if best_pnl:
        recommendations.append({
            'strategy': best_pnl,
            'reason': 'Highest absolute returns',
            'recommendation': 'Consider for aggressive growth allocation'
        })
        
    if most_consistent and most_consistent != best_pnl:
        recommendations.append({
            'strategy': most_consistent,
            'reason': 'Most consistent win rate',
            'recommendation': 'Consider for stable income allocation'
        })
        
    # Strategy-specific recommendations
    if 'multipool_staking' in results and 'error' not in results['multipool_staking']:
        staking_result = results['multipool_staking']
        apy = staking_result.get('apy_achieved', 0)
        if apy > 8:  # If achieved good APY
            recommendations.append({
                'strategy': 'multipool_staking',
                'reason': f'Achieved {apy:.1f}% APY with lower risk',
                'recommendation': 'Excellent for passive income generation'
            })
            
    if 'amm' in results and 'error' not in results['amm']:
        amm_result = results['amm']
        trades = amm_result.get('trades', 0)
        if trades > 100:  # High frequency
            recommendations.append({
                'strategy': 'amm',
                'reason': 'High frequency trading with spread capture',
                'recommendation': 'Good for market making in volatile conditions'
            })
            
    return recommendations


def main():
    """Main function to demonstrate all strategies"""
    print("=== Advanced Trading Strategies Backtest ===")
    print("Testing all implemented strategies with sample data...")
    print()
    
    # Run all strategies
    results = run_all_strategies_backtest()
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
        
    # Display results summary
    print("\n=== Strategy Performance Summary ===")
    for strategy, result in results.items():
        if strategy == 'comparison':
            continue
            
        if 'error' in result:
            print(f"{strategy:20} | ERROR: {result['error']}")
        else:
            pnl = result.get('pnl', 0)
            trades = result.get('trades', 0) 
            win_rate = result.get('win_rate_pct', 0)
            print(f"{strategy:20} | PnL: {pnl:8.2f} | Trades: {trades:4d} | Win Rate: {win_rate:5.1f}%")
            
    # Display comparison and recommendations
    if 'comparison' in results and results['comparison'].get('summary'):
        summary = results['comparison']['summary']
        print(f"\n=== Comparison Summary ===")
        print(f"Best Performing: {summary.get('best_performing', 'N/A')}")
        print(f"Most Consistent: {summary.get('most_consistent', 'N/A')}")
        print(f"Average PnL: {summary.get('avg_pnl', 0):.2f}")
        print(f"Average Win Rate: {summary.get('avg_win_rate', 0):.1f}%")
        
    recommendations = get_strategy_recommendations(results)
    if recommendations:
        print(f"\n=== Strategy Recommendations ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['strategy']}: {rec['reason']}")
            print(f"   → {rec['recommendation']}")
            
    # Save results to file
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_strategies_backtest_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {
                    key: (str(val) if isinstance(val, datetime) else val)
                    for key, val in v.items()
                    if key != 'trade_list' or len(str(val)) < 10000  # Limit large trade lists
                }
            else:
                json_results[k] = v
                
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
        print(f"\n✓ Results saved to {filename}")
        
    except Exception as e:
        print(f"✗ Could not save results: {e}")
        

if __name__ == '__main__':
    main()