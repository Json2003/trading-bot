"""Expanded grid search optimizer for the aggressive strategy.

Features:
- Parallel processing using multiprocessing for faster execution
- Early stopping criteria and adaptive search to reduce unnecessary computations
- Intermediate result saving to prevent data loss on failure
- Enhanced logging and progress tracking
- Larger parameter space for comprehensive optimization

Writes ranked results to `opt_results_expanded.json` in the repository root.
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


def evaluate_single_combination_expanded(args: Tuple[pd.DataFrame, float, float, int, float]) -> Dict[str, Any]:
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


def adaptive_search_pruning(results: List[Dict], current_params: Dict, threshold: float = 0.1) -> bool:
    """Determine if current parameter combination should be pruned based on adaptive search.
    
    Args:
        results: List of completed results
        current_params: Current parameter combination being considered
        threshold: Threshold for pruning decision
        
    Returns:
        True if the combination should be skipped
    """
    if len(results) < 20:  # Need some baseline data
        return False
    
    # Find similar parameter combinations
    similar_results = []
    for result in results:
        if result.get('status') != 'success':
            continue
            
        p = result['params']
        # Calculate parameter similarity (simple distance metric)
        param_diff = (
            abs(p['tp'] - current_params['tp']) / current_params['tp'] +
            abs(p['sl'] - current_params['sl']) / current_params['sl'] +
            abs(p['hold'] - current_params['hold']) / current_params['hold'] +
            abs(p['risk'] - current_params['risk']) / current_params['risk']
        ) / 4
        
        if param_diff < 0.3:  # Consider similar if within 30% on average
            similar_results.append(result)
    
    # If similar combinations all perform poorly, prune this one
    if len(similar_results) >= 3:
        avg_performance = sum(r['win_rate'] + r['pnl'] / 1000 for r in similar_results) / len(similar_results)
        if avg_performance < threshold:
            return True
    
    return False


def save_intermediate_results_expanded(results: List[Dict], iteration: int, out_path: Path):
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
        logger.info(f"Saved intermediate results to {intermediate_path} ({len(results)} results)")
    except Exception as e:
        logger.error(f"Failed to save intermediate results: {e}")


def should_early_stop_expanded(results: List[Dict], min_results: int = 100, patience: int = 50) -> bool:
    """Check if early stopping criteria are met for expanded search.
    
    Args:
        results: List of current results
        min_results: Minimum number of results before considering early stopping
        patience: Number of iterations without improvement to trigger early stopping
        
    Returns:
        True if early stopping should be triggered
    """
    if len(results) < min_results:
        return False
    
    # Sort by combined metric (win_rate + normalized pnl)
    successful_results = [r for r in results if r.get('status') == 'success']
    if len(successful_results) < min_results:
        return False
    
    sorted_results = sorted(successful_results, 
                           key=lambda x: (x['win_rate'] + x['pnl'] / 10000), 
                           reverse=True)
    
    # Check if best result hasn't improved significantly in recent results
    if len(sorted_results) > patience:
        best_score = sorted_results[0]['win_rate'] + sorted_results[0]['pnl'] / 10000
        recent_results = successful_results[-patience:]
        recent_best = max([r['win_rate'] + r['pnl'] / 10000 for r in recent_results])
        
        improvement_threshold = 0.02  # 2% improvement required
        if recent_best < best_score * (1 - improvement_threshold):
            logger.info("Early stopping triggered - no significant improvement in recent iterations")
            return True
    
    return False


def run_grid(symbol='BTC/USDT', n_workers: int = None, enable_early_stopping: bool = True,
             save_intermediate: bool = True, adaptive_pruning: bool = True):
    """Run expanded parallel grid search optimization with adaptive features.
    
    Args:
        symbol: Trading symbol to optimize for
        n_workers: Number of parallel workers (defaults to CPU count)
        enable_early_stopping: Whether to enable early stopping
        save_intermediate: Whether to save intermediate results
        adaptive_pruning: Whether to enable adaptive search pruning
    """
    logger.info(f"Starting expanded grid search optimization for {symbol}")
    
    # Load data
    df = load_bars(symbol)
    logger.info(f"Loaded {len(df)} bars of data")

    # Expanded parameter grid
    params = {
        'take_profit_pct': [0.002, 0.004, 0.006, 0.01, 0.02],
        'stop_loss_pct': [0.001, 0.002, 0.003, 0.005, 0.01],
        'max_holding_bars': [3, 6, 12, 24, 48],
        'risk_pct': [0.005, 0.01, 0.02]
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

    # Set up multiprocessing
    if n_workers is None:
        n_workers = min(cpu_count(), 12)  # Allow more workers for expanded search
    
    logger.info(f"Using {n_workers} parallel workers")
    logger.info(f"Adaptive pruning: {'enabled' if adaptive_pruning else 'disabled'}")
    logger.info(f"Early stopping: {'enabled' if enable_early_stopping else 'disabled'}")

    # Output path setup
    out_path = Path(HERE.parent) / 'opt_results_expanded.json'

    results = []
    pruned_count = 0
    start_time = time.time()

    try:
        # Process combinations in batches for better control
        batch_size = max(n_workers * 2, 20)
        
        for batch_start in range(0, len(combos), batch_size):
            batch_end = min(batch_start + batch_size, len(combos))
            batch_combos = combos[batch_start:batch_end]
            
            # Apply adaptive pruning if enabled
            if adaptive_pruning:
                filtered_combos = []
                for tp, sl, hold, risk in batch_combos:
                    current_params = {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk}
                    if not adaptive_search_pruning(results, current_params):
                        filtered_combos.append((tp, sl, hold, risk))
                    else:
                        pruned_count += 1
                batch_combos = filtered_combos
            
            if not batch_combos:
                continue
            
            # Prepare arguments for parallel processing
            args_list = [(df, tp, sl, hold, risk) for tp, sl, hold, risk in batch_combos]
            
            # Process batch in parallel
            with Pool(processes=n_workers) as pool:
                batch_results = pool.map(evaluate_single_combination_expanded, args_list)
                results.extend(batch_results)
            
            # Progress logging
            evaluated_count = len(results) + pruned_count
            progress = evaluated_count / total_combinations * 100
            elapsed = time.time() - start_time
            
            if results:
                rate = len(results) / elapsed
                eta = (total_combinations - evaluated_count) / rate if rate > 0 else 0
            else:
                eta = 0
            
            successful_count = len([r for r in results if r.get('status') == 'success'])
            
            logger.info(f"Progress: {progress:.1f}% ({evaluated_count}/{total_combinations}) "
                       f"- Successful: {successful_count}, Pruned: {pruned_count} "
                       f"- Rate: {rate:.1f}/s - ETA: {eta/60:.1f}min")
            
            # Save intermediate results periodically
            if save_intermediate and len(results) % 100 == 0:
                save_intermediate_results_expanded(results, len(results), out_path)
            
            # Check early stopping criteria
            if enable_early_stopping and should_early_stop_expanded(results):
                logger.info(f"Early stopping after {len(results)} evaluations")
                break

    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise

    # Filter and analyze results
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_count = len(results) - len(successful_results)

    if failed_count > 0:
        logger.warning(f"{failed_count} evaluations failed")
    
    if pruned_count > 0:
        logger.info(f"{pruned_count} combinations pruned by adaptive search")

    # Rank results by win_rate then pnl
    results_sorted = sorted(successful_results,
                           key=lambda x: (x['win_rate'], x['pnl']),
                           reverse=True)

    # Calculate execution statistics
    elapsed = time.time() - start_time
    
    # Save final results with comprehensive metadata
    final_results = {
        'metadata': {
            'symbol': symbol,
            'total_combinations': total_combinations,
            'evaluated_combinations': len(results),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': failed_count,
            'pruned_combinations': pruned_count,
            'execution_time_seconds': elapsed,
            'evaluations_per_second': len(results) / elapsed if elapsed > 0 else 0,
            'early_stopped': len(results) + pruned_count < total_combinations,
            'workers_used': n_workers,
            'adaptive_pruning_enabled': adaptive_pruning,
            'early_stopping_enabled': enable_early_stopping,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameter_ranges': params
        },
        'results': results_sorted[:500]  # Limit to top 500 results to keep file size manageable
    }

    # Save results
    out_path.write_text(json.dumps(final_results, indent=2))

    logger.info(f'Expanded optimization completed in {elapsed:.1f}s')
    logger.info(f'Evaluation rate: {len(results) / elapsed:.1f} combinations/second')
    logger.info(f'Saved top {len(results_sorted)} results to {out_path}')

    # Print top 10 results
    if results_sorted:
        logger.info("Top 10 results:")
        for i, result in enumerate(results_sorted[:10]):
            params = result['params']
            logger.info(f"  {i+1}. Win Rate: {result['win_rate']:.2f}%, "
                       f"PnL: ${result['pnl']:.2f}, Trades: {result['trades']}, "
                       f"Params: tp={params['tp']}, sl={params['sl']}, "
                       f"hold={params['hold']}, risk={params['risk']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Expanded grid search optimizer with advanced features')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--no-early-stopping', action='store_true', 
                       help='Disable early stopping')
    parser.add_argument('--no-intermediate', action='store_true',
                       help='Disable intermediate result saving')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive search pruning')
    
    args = parser.parse_args()
    
    run_grid(
        symbol=args.symbol,
        n_workers=args.workers,
        enable_early_stopping=not args.no_early_stopping,
        save_intermediate=not args.no_intermediate,
        adaptive_pruning=not args.no_adaptive
    )
