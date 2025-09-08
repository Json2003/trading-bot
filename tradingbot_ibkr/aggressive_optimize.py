"""Grid search optimizer for the aggressive strategy.

Features:
- Parallel processing using multiprocessing for faster execution
- Early stopping criteria to reduce unnecessary computations
- Intermediate result saving to prevent data loss
- Enhanced logging and progress tracking

Saves ranked results to `opt_results.json`.
"""
import itertools
import json
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pickle
import time
from typing import Dict, List, Tuple, Any

import pandas as pd

from backtest_ccxt import aggressive_strategy_backtest

HERE = Path(__file__).resolve().parent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_bars(symbol='BTC/USDT'):
    """Load price bar data for backtesting.
    
    Args:
        symbol: Trading symbol to load data for
        
    Returns:
        DataFrame with price data indexed by timestamp
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    path = HERE / 'datafiles' / f"{symbol.replace('/','_')}_bars.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path, parse_dates=['ts'], index_col='ts')


def evaluate_single_combination(args: Tuple[pd.DataFrame, float, float, int, float]) -> Dict[str, Any]:
    """Evaluate a single parameter combination for backtesting.
    
    Args:
        args: Tuple containing (df, tp, sl, hold, risk) parameters
        
    Returns:
        Dictionary with backtest results and parameters
    """
    df, tp, sl, hold, risk = args
    try:
        stats = aggressive_strategy_backtest(
            df, 
            take_profit_pct=tp, 
            stop_loss_pct=sl, 
            max_holding_bars=hold, 
            fee_pct=0.001, 
            slippage_pct=0.0005, 
            starting_balance=10000.0
        )
        return {
            'params': {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk},
            'win_rate': stats.get('win_rate_pct', 0.0),
            'pnl': stats.get('pnl', 0.0),
            'trades': stats.get('trades', 0),
            'status': 'success'
        }
    except Exception as e:
        logger.warning(f"Error evaluating params tp={tp}, sl={sl}, hold={hold}, risk={risk}: {e}")
        return {
            'params': {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk},
            'win_rate': 0.0,
            'pnl': 0.0,
            'trades': 0,
            'status': 'error',
            'error': str(e)
        }


def save_intermediate_results(results: List[Dict], iteration: int, out_path: Path):
    """Save intermediate results to prevent data loss.
    
    Args:
        results: List of result dictionaries
        iteration: Current iteration number
        out_path: Base output path
    """
    intermediate_path = out_path.parent / f"{out_path.stem}_intermediate_{iteration}.pkl"
    try:
        with open(intermediate_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Saved intermediate results to {intermediate_path}")
    except Exception as e:
        logger.error(f"Failed to save intermediate results: {e}")


def should_early_stop(results: List[Dict], min_results: int = 50, patience: int = 10) -> bool:
    """Check if early stopping criteria are met.
    
    Args:
        results: List of current results
        min_results: Minimum number of results before considering early stopping
        patience: Number of iterations without improvement to trigger early stopping
        
    Returns:
        True if early stopping should be triggered
    """
    if len(results) < min_results:
        return False
    
    # Sort by combined metric (win_rate + pnl)
    sorted_results = sorted(results, key=lambda x: (x['win_rate'] + x['pnl'] / 1000), reverse=True)
    
    # Check if best result hasn't improved in last `patience` results
    if len(sorted_results) > patience:
        best_score = sorted_results[0]['win_rate'] + sorted_results[0]['pnl'] / 1000
        recent_best = max([r['win_rate'] + r['pnl'] / 1000 for r in results[-patience:]])
        
        if recent_best < best_score * 0.95:  # Allow 5% tolerance
            logger.info("Early stopping triggered - no improvement in recent iterations")
            return True
    
    return False

def run_grid(symbol='BTC/USDT', n_workers: int = None, enable_early_stopping: bool = True, 
             save_intermediate: bool = True):
    """Run parallel grid search optimization.
    
    Args:
        symbol: Trading symbol to optimize for
        n_workers: Number of parallel workers (defaults to CPU count)
        enable_early_stopping: Whether to enable early stopping
        save_intermediate: Whether to save intermediate results
    """
    logger.info(f"Starting grid search optimization for {symbol}")
    
    # Load data
    df = load_bars(symbol)
    logger.info(f"Loaded {len(df)} bars of data")
    
    # Define parameter grid
    params = {
        'take_profit_pct': [0.002, 0.004, 0.006],
        'stop_loss_pct': [0.001, 0.002, 0.003],
        'max_holding_bars': [6, 12, 24],
        'risk_pct': [0.01, 0.02]
    }
    
    # Generate all parameter combinations
    combos = list(itertools.product(
        params['take_profit_pct'], 
        params['stop_loss_pct'], 
        params['max_holding_bars'], 
        params['risk_pct']
    ))
    
    total_combinations = len(combos)
    logger.info(f"Total combinations to evaluate: {total_combinations}")
    
    # Prepare arguments for parallel processing
    args_list = [(df, tp, sl, hold, risk) for tp, sl, hold, risk in combos]
    
    # Set up multiprocessing
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Limit to 8 to avoid overwhelming system
    
    logger.info(f"Using {n_workers} parallel workers")
    
    # Output path setup
    out_path = Path('opt_results.json')
    
    results = []
    start_time = time.time()
    
    try:
        # Use multiprocessing pool for parallel execution
        with Pool(processes=n_workers) as pool:
            # Process in chunks for better memory management and intermediate saving
            chunk_size = max(1, total_combinations // (n_workers * 4))
            
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i + chunk_size]
                chunk_results = pool.map(evaluate_single_combination, chunk)
                results.extend(chunk_results)
                
                # Progress logging
                progress = len(results) / total_combinations * 100
                elapsed = time.time() - start_time
                eta = (elapsed / len(results) * total_combinations) - elapsed if results else 0
                
                logger.info(f"Progress: {progress:.1f}% ({len(results)}/{total_combinations}) "
                          f"- Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
                
                # Save intermediate results
                if save_intermediate and len(results) % (chunk_size * 2) == 0:
                    save_intermediate_results(results, len(results), out_path)
                
                # Check early stopping criteria
                if enable_early_stopping and should_early_stop(results):
                    logger.info(f"Early stopping after {len(results)} evaluations")
                    break
    
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise
    
    # Filter out failed results
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        logger.warning(f"{failed_count} evaluations failed")
    
    # Rank results by win_rate then pnl
    results_sorted = sorted(successful_results, 
                           key=lambda x: (x['win_rate'], x['pnl']), 
                           reverse=True)
    
    # Save final results
    final_results = {
        'metadata': {
            'symbol': symbol,
            'total_combinations': total_combinations,
            'successful_evaluations': len(successful_results),
            'failed_evaluations': failed_count,
            'execution_time_seconds': time.time() - start_time,
            'early_stopped': len(results) < total_combinations,
            'workers_used': n_workers,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': results_sorted
    }
    
    out_path.write_text(json.dumps(final_results, indent=2))
    
    elapsed = time.time() - start_time
    logger.info(f'Optimization completed in {elapsed:.1f}s')
    logger.info(f'Saved {len(results_sorted)} results to {out_path}')
    
    # Print top 5 results
    if results_sorted:
        logger.info("Top 5 results:")
        for i, result in enumerate(results_sorted[:5]):
            params = result['params']
            logger.info(f"  {i+1}. Win Rate: {result['win_rate']:.2f}%, "
                       f"PnL: ${result['pnl']:.2f}, Trades: {result['trades']}, "
                       f"Params: tp={params['tp']}, sl={params['sl']}, "
                       f"hold={params['hold']}, risk={params['risk']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid search optimizer with parallel processing')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--no-early-stopping', action='store_true', 
                       help='Disable early stopping')
    parser.add_argument('--no-intermediate', action='store_true',
                       help='Disable intermediate result saving')
    
    args = parser.parse_args()
    
    run_grid(
        symbol=args.symbol,
        n_workers=args.workers,
        enable_early_stopping=not args.no_early_stopping,
        save_intermediate=not args.no_intermediate
    )
