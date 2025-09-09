"""Grid search optimizer for the aggressive strategy.

Features:
- Parallel computing for faster optimization
- Early stopping criteria to prevent overfitting
- Adaptive search pruning to focus on promising parameters
- Intermediate result saving to prevent data loss
- Detailed logging for progress tracking

Saves ranked results to `opt_results.json`.
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
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

def load_bars(symbol='BTC/USDT'):
    """Load OHLCV data for backtesting."""
    path = HERE / 'datafiles' / f"{symbol.replace('/','_')}_bars.csv"
    if not path.exists():
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(path)
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path, parse_dates=['ts'], index_col='ts')

def backtest_single_combination(args):
    """Run backtest for a single parameter combination (for parallel processing)."""
    df, tp, sl, hold, risk, combo_idx, total_combos = args
    try:
        logger.debug(f"[{combo_idx}/{total_combos}] Testing: tp={tp}, sl={sl}, hold={hold}, risk={risk}")
        stats = aggressive_strategy_backtest(
            df, 
            take_profit_pct=tp, 
            stop_loss_pct=sl, 
            max_holding_bars=hold, 
            fee_pct=0.001, 
            slippage_pct=0.0005, 
            starting_balance=10000.0
        )
        result = {
            'params': {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk},
            'win_rate': stats['win_rate_pct'],
            'pnl': stats['pnl'],
            'trades': stats['trades'],
            'combo_idx': combo_idx
        }
        logger.debug(f"[{combo_idx}/{total_combos}] Completed - WR: {result['win_rate']:.2f}%, PnL: {result['pnl']:.2f}")
        return result
    except Exception as e:
        logger.error(f"[{combo_idx}/{total_combos}] Failed: {e}")
        return None

def should_prune_parameter(results, param_name, param_value, threshold_percentile=25):
    """Adaptive pruning: check if parameter consistently performs poorly."""
    if len(results) < 20:  # Need enough data for meaningful statistics
        return False
    
    param_results = [r for r in results if r['params'][param_name] == param_value]
    if len(param_results) < 5:  # Need enough samples for this parameter
        return False
    
    param_win_rates = [r['win_rate'] for r in param_results]
    all_win_rates = [r['win_rate'] for r in results]
    
    param_median = sorted(param_win_rates)[len(param_win_rates)//2]
    overall_threshold = sorted(all_win_rates)[int(len(all_win_rates) * threshold_percentile / 100)]
    
    should_prune = param_median < overall_threshold
    if should_prune:
        logger.info(f"Pruning parameter {param_name}={param_value} (median WR: {param_median:.2f}% < threshold: {overall_threshold:.2f}%)")
    
    return should_prune

def save_intermediate_results(results, output_path, combo_idx, total_combos):
    """Save intermediate results to prevent data loss."""
    intermediate_path = output_path.with_suffix('.tmp.json')
    results_with_progress = {
        'progress': {
            'completed': combo_idx,
            'total': total_combos,
            'completion_pct': (combo_idx / total_combos) * 100,
            'timestamp': time.time()
        },
        'results': results
    }
    
    with open(intermediate_path, 'w') as f:
        json.dump(results_with_progress, f, indent=2)
    
    logger.info(f"Intermediate results saved ({combo_idx}/{total_combos} - {combo_idx/total_combos*100:.1f}%)")

def run_grid(symbol='BTC/USDT', max_workers=None, early_stopping_patience=50, enable_pruning=True):
    """
    Run grid search optimization with advanced features.
    
    Args:
        symbol: Trading symbol to optimize
        max_workers: Number of parallel processes (None for auto-detection)
        early_stopping_patience: Stop if no improvement for N consecutive results
        enable_pruning: Whether to enable adaptive parameter pruning
    """
    start_time = time.time()
    logger.info(f"Starting grid search optimization for {symbol}")
    
    try:
        df = load_bars(symbol)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Parameter grid
    params = {
        'take_profit_pct': [0.002, 0.004, 0.006],
        'stop_loss_pct': [0.001, 0.002, 0.003],
        'max_holding_bars': [6, 12, 24],
        'risk_pct': [0.01, 0.02]
    }
    
    # Generate all combinations
    initial_combos = list(itertools.product(
        params['take_profit_pct'],
        params['stop_loss_pct'],
        params['max_holding_bars'],
        params['risk_pct']
    ))
    
    logger.info(f"Initial parameter combinations: {len(initial_combos)}")
    
    results = []
    best_win_rate = 0
    no_improvement_count = 0
    pruned_params = set()
    
    # Output path setup
    out = Path('opt_results.json')
    
    # Prepare arguments for parallel processing
    combo_args = []
    for idx, (tp, sl, hold, risk) in enumerate(initial_combos, 1):
        # Skip pruned parameter combinations
        param_key = ('tp', tp)
        if param_key in pruned_params:
            continue
        param_key = ('sl', sl)
        if param_key in pruned_params:
            continue
        param_key = ('hold', hold)
        if param_key in pruned_params:
            continue
        param_key = ('risk', risk)
        if param_key in pruned_params:
            continue
            
        combo_args.append((df, tp, sl, hold, risk, idx, len(initial_combos)))
    
    logger.info(f"Processing {len(combo_args)} parameter combinations with {max_workers or 'auto'} workers")
    
    # Process combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_single_combination, args): args for args in combo_args}
        
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
                
            results.append(result)
            current_combo = result['combo_idx']
            
            # Check for improvement (early stopping)
            if result['win_rate'] > best_win_rate:
                best_win_rate = result['win_rate']
                no_improvement_count = 0
                logger.info(f"New best win rate: {best_win_rate:.2f}% - {result['params']}")
            else:
                no_improvement_count += 1
            
            # Adaptive pruning (only after enough data)
            if enable_pruning and len(results) >= 20 and len(results) % 10 == 0:
                for param_name in ['tp', 'sl', 'hold', 'risk']:
                    param_values = list(set(params[f'{param_name}_pct' if param_name != 'hold' else 'max_holding_bars'][0] for r in results))
                    for param_value in param_values:
                        if should_prune_parameter(results, param_name, param_value):
                            pruned_params.add((param_name, param_value))
            
            # Save intermediate results every 10 completions
            if len(results) % 10 == 0:
                save_intermediate_results(results, out, current_combo, len(initial_combos))
            
            # Early stopping check
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {no_improvement_count} combinations without improvement")
                break
    
    # Sort results by win rate, then PnL
    results_sorted = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)
    
    # Save final results
    final_data = {
        'metadata': {
            'symbol': symbol,
            'total_combinations_tested': len(results),
            'total_combinations_possible': len(initial_combos),
            'optimization_time_seconds': time.time() - start_time,
            'best_win_rate': best_win_rate,
            'early_stopping_triggered': no_improvement_count >= early_stopping_patience,
            'pruned_parameters': list(pruned_params)
        },
        'results': results_sorted
    }
    
    out.write_text(json.dumps(final_data, indent=2))
    
    # Clean up intermediate file
    intermediate_path = out.with_suffix('.tmp.json')
    if intermediate_path.exists():
        intermediate_path.unlink()
    
    elapsed = time.time() - start_time
    logger.info(f'Optimization completed! Saved {len(results_sorted)} results to {out}')
    logger.info(f'Total time: {elapsed:.1f}s, Best win rate: {best_win_rate:.2f}%')
    
    return final_data

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid search optimization with advanced features')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol to optimize')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--no-pruning', action='store_true', help='Disable adaptive pruning')
    
    args = parser.parse_args()
    
    try:
        run_grid(
            symbol=args.symbol,
            max_workers=args.workers,
            early_stopping_patience=args.patience,
            enable_pruning=not args.no_pruning
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
