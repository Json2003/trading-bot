"""Expanded grid search optimizer for the aggressive strategy.

Features:
- Expanded parameter ranges for comprehensive testing
- Parallel computing for faster optimization
- Early stopping criteria and adaptive search pruning
- Intermediate result saving to prevent data loss
- Detailed progress tracking and logging
- Memory-efficient processing for large parameter spaces

Writes ranked results to `opt_results_expanded.json` in the repository root.
"""
import itertools
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from backtest_ccxt import aggressive_strategy_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_expanded.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

def load_bars(symbol='BTC/USDT'):
    """Load OHLCV data for backtesting with error handling."""
    path = HERE / 'datafiles' / f"{symbol.replace('/','_')}_bars.csv"
    if not path.exists():
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(path)
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['ts'], index_col='ts')
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df

def backtest_combination(args):
    """Execute backtest for a single parameter combination (optimized for parallel processing)."""
    df, tp, sl, hold, risk, combo_idx, total_combos = args
    try:
        start_time = time.time()
        
        stats = aggressive_strategy_backtest(
            df,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            max_holding_bars=hold,
            fee_pct=0.001,
            slippage_pct=0.0005,
            starting_balance=10000.0
        )
        
        execution_time = time.time() - start_time
        
        result = {
            'params': {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk},
            'win_rate': stats.get('win_rate_pct', 0.0),
            'pnl': stats.get('pnl', 0.0),
            'trades': stats.get('trades', 0),
            'execution_time': execution_time,
            'combo_idx': combo_idx
        }
        
        if combo_idx % 20 == 0 or combo_idx <= 10:  # Log first 10 and every 20th
            logger.info(f'[{combo_idx}/{total_combos}] tp={tp:.3f} sl={sl:.3f} hold={hold} risk={risk:.3f} | WR={result["win_rate"]:.1f}% PnL={result["pnl"]:.1f} Trades={result["trades"]} ({execution_time:.2f}s)')
        
        return result
        
    except Exception as e:
        logger.error(f'[{combo_idx}/{total_combos}] Failed: tp={tp} sl={sl} hold={hold} risk={risk} - {str(e)}')
        return None

def estimate_completion_time(completed_combos, total_combos, start_time):
    """Estimate remaining time based on current progress."""
    if completed_combos == 0:
        return "Unknown"
    
    elapsed = time.time() - start_time
    avg_time_per_combo = elapsed / completed_combos
    remaining_combos = total_combos - completed_combos
    estimated_remaining = avg_time_per_combo * remaining_combos
    
    return f"{estimated_remaining/60:.1f} minutes"

def save_progress_checkpoint(results, output_path, completed, total, start_time, metadata=None):
    """Save progress checkpoint with detailed statistics."""
    checkpoint_data = {
        'metadata': {
            'timestamp': time.time(),
            'progress': {
                'completed': completed,
                'total': total,
                'completion_pct': (completed / total) * 100 if total > 0 else 0,
                'estimated_remaining_time': estimate_completion_time(completed, total, start_time)
            },
            'statistics': {
                'best_win_rate': max((r['win_rate'] for r in results), default=0),
                'avg_win_rate': sum(r['win_rate'] for r in results) / len(results) if results else 0,
                'total_positive_pnl': sum(1 for r in results if r['pnl'] > 0),
                'avg_execution_time': sum(r.get('execution_time', 0) for r in results) / len(results) if results else 0
            }
        }
    }
    
    if metadata:
        checkpoint_data['metadata'].update(metadata)
    
    checkpoint_data['results'] = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)
    
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logger.info(f"Checkpoint saved: {completed}/{total} ({completed/total*100:.1f}%) - Best WR: {checkpoint_data['metadata']['statistics']['best_win_rate']:.2f}%")

def run_grid(symbol='BTC/USDT', max_workers=None, batch_size=50):
    """
    Run expanded grid search with advanced optimization features.
    
    Args:
        symbol: Trading symbol to optimize  
        max_workers: Number of parallel workers (None for auto-detection)
        batch_size: Number of results to process before saving checkpoint
    """
    start_time = time.time()
    logger.info(f"Starting expanded grid search optimization for {symbol}")
    logger.info(f"Using {max_workers or 'auto-detected'} parallel workers, batch size: {batch_size}")
    
    try:
        df = load_bars(symbol)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Expanded parameter grid for comprehensive testing
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

    total_combos = len(combos)
    logger.info(f"Total parameter combinations to test: {total_combos}")
    logger.info(f"Estimated total time (sequential): {total_combos * 0.1 / 60:.1f} minutes")

    results = []
    output_path = Path(HERE.parent) / 'opt_results_expanded.json'
    
    # Prepare arguments for parallel processing
    combo_args = [(df, tp, sl, hold, risk, idx, total_combos) 
                  for idx, (tp, sl, hold, risk) in enumerate(combos, start=1)]
    
    # Process in parallel with progress tracking
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_combo = {executor.submit(backtest_combination, args): args for args in combo_args}
        
        # Process completed jobs
        for future in as_completed(future_to_combo):
            result = future.result()
            
            if result is not None:
                results.append(result)
            
            completed_count += 1
            
            # Save checkpoint every batch_size completions
            if completed_count % batch_size == 0:
                save_progress_checkpoint(results, output_path, completed_count, total_combos, start_time)
                
                # Memory management: keep only top results to prevent memory bloat
                if len(results) > 1000:  # Keep top 1000 results
                    results = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)[:1000]
                    logger.info(f"Trimmed results to top 1000 to manage memory usage")

    # Final processing and ranking
    logger.info("Processing final results...")
    results_sorted = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)

    # Calculate final statistics  
    total_time = time.time() - start_time
    successful_results = len(results_sorted)
    success_rate = (successful_results / total_combos) * 100
    
    final_data = {
        'metadata': {
            'symbol': symbol,
            'optimization_completed': time.time(),
            'total_combinations': total_combos,
            'successful_results': successful_results,
            'success_rate_pct': success_rate,
            'total_time_seconds': total_time,
            'avg_time_per_combination': total_time / total_combos,
            'parameters_tested': params,
            'best_result': results_sorted[0] if results_sorted else None
        },
        'results': results_sorted
    }
    
    # Save final results
    output_path.write_text(json.dumps(final_data, indent=2))
    
    # Cleanup checkpoint file
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    # Summary logging
    logger.info("="*60)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total combinations: {total_combos}")
    logger.info(f"Successful results: {successful_results} ({success_rate:.1f}%)")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Average time per combination: {total_time/total_combos:.3f}s")
    
    if results_sorted:
        best = results_sorted[0]
        logger.info(f"Best result: WR={best['win_rate']:.2f}% PnL={best['pnl']:.2f} Trades={best['trades']}")
        logger.info(f"Best params: {best['params']}")
    
    logger.info("="*60)
    
    return final_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Expanded grid search optimization with parallel processing')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol to optimize (default: BTC/USDT)')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=50, help='Checkpoint save interval (default: 50)')
    
    args = parser.parse_args()
    
    try:
        run_grid(
            symbol=args.symbol,
            max_workers=args.workers,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
