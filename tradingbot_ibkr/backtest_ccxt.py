"""Enhanced CCXT backtest runner with comprehensive features.

This script provides advanced backtesting capabilities including:
- Comprehensive trade execution logging
- Vectorized pandas operations for performance
- Professional-grade analytics (Sharpe ratio, profit factor, etc.)
- Detailed performance metrics and risk analysis
- Memory-efficient processing for large datasets
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use package-relative imports with error handling
try:
    from .money_engine import choose_position_size, fixed_fractional, round_qty, round_price
except ImportError:
    logger.warning("Money engine not available, using simplified position sizing")
    def choose_position_size(balance, risk_pct, entry_price, stop_price, leverage=1.0, min_qty=0.0):
        return balance * risk_pct / abs(entry_price - stop_price), balance * risk_pct

try:
    from .models.online_trainer import OnlineTrainer
except ImportError:
    logger.warning("Online trainer not available, adaptive learning disabled")
    class OnlineTrainer:
        def load(self): pass
        def predict_proba(self, features): return 0.5
        def learn_one(self, features, outcome): pass

load_dotenv()
EXCHANGE = os.getenv('EXCHANGE', 'binance')
PAPER = os.getenv('PAPER', 'true').lower() == 'true'

def fetch_ohlcv(symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange with error handling and logging.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        limit: Number of candles to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Fetching {limit} {timeframe} candles for {symbol} from {EXCHANGE}")
    
    try:
        ex = getattr(ccxt, EXCHANGE)()
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not data:
            raise ValueError("No data returned from exchange")
        
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        
        # Validate data quality
        if df.isnull().any().any():
            logger.warning("Missing values detected in OHLCV data")
            df = df.fillna(method='ffill')
        
        logger.info(f"Successfully fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV data: {e}")
        raise

def calculate_performance_metrics(equity_curve: List[float], trades: List[Dict], 
                                risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate comprehensive performance metrics and risk analysis.
    
    Args:
        equity_curve: List of equity values over time
        trades: List of individual trade results
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        
    Returns:
        Dictionary with detailed performance metrics
    """
    if not equity_curve or len(equity_curve) < 2:
        return {}
    
    # Convert to numpy array for vectorized operations
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    # Basic metrics
    total_return = (equity[-1] - equity[0]) / equity[0]
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    
    # Win rate and profit factor
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 1
    profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (np.mean(returns) * 252 - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Drawdown analysis using vectorized operations
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)
    max_drawdown_duration = 0
    
    # Calculate drawdown duration
    in_drawdown = False
    current_dd_duration = 0
    for dd in drawdown:
        if dd < -0.001:  # In drawdown (threshold to avoid noise)
            if not in_drawdown:
                in_drawdown = True
                current_dd_duration = 1
            else:
                current_dd_duration += 1
        else:
            if in_drawdown:
                max_drawdown_duration = max(max_drawdown_duration, current_dd_duration)
                in_drawdown = False
                current_dd_duration = 0
    
    # Calmar ratio (return/max_drawdown)
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    # Additional metrics
    sortino_ratio = 0
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        sortino_ratio = (np.mean(returns) * 252 - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'annualized_return': (1 + total_return) ** (252 / len(equity)) - 1,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'max_drawdown_duration': max_drawdown_duration,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_trade_duration': np.mean([t.get('duration_bars', 1) for t in trades]) if trades else 0,
        'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss > 0 else float('inf')
    }
    
    logger.info(f"Performance metrics calculated: Sharpe={sharpe_ratio:.3f}, Max DD={max_drawdown*100:.2f}%, Win Rate={win_rate*100:.1f}%")
    
    return metrics

def simple_backtest(df: pd.DataFrame) -> Tuple[float, List[Tuple]]:
    """
    Simple moving average crossover backtest with enhanced logging.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Tuple of (total_pnl, trade_list)
    """
    logger.info("Running simple MA crossover backtest")
    
    # Vectorized moving average calculation
    df = df.copy()
    df['ma_fast'] = df['close'].rolling(10, min_periods=1).mean()
    df['ma_slow'] = df['close'].rolling(30, min_periods=1).mean()
    df.dropna(inplace=True)
    
    if len(df) == 0:
        logger.warning("No data available after calculating moving averages")
        return 0, []
    
    position = 0
    entry_price = 0
    pnl = 0
    trades = []
    
    logger.info(f"Processing {len(df)} bars for backtest")
    
    for idx, row in df.iterrows():
        if row['ma_fast'] > row['ma_slow'] and position == 0:
            # Long entry
            position = 1
            entry_price = row['close']
            trades.append(('buy', idx, entry_price))
            logger.debug(f"Long entry at {idx}: price={entry_price:.2f}")
            
        elif row['ma_fast'] < row['ma_slow'] and position == 1:
            # Long exit
            position = 0
            exit_price = row['close']
            trade_pnl = exit_price - entry_price
            pnl += trade_pnl
            trades.append(('sell', idx, exit_price))
            logger.debug(f"Long exit at {idx}: price={exit_price:.2f}, PnL={trade_pnl:.2f}")
    
    logger.info(f"Simple backtest completed: {len(trades)//2} trades, Total PnL: {pnl:.2f}")
    return pnl, trades


def aggressive_strategy_backtest(df: pd.DataFrame, take_profit_pct: float = 0.004, 
                                stop_loss_pct: float = 0.002, max_holding_bars: int = 12,
                                fee_pct: float = 0.0, slippage_pct: float = 0.0, 
                                starting_balance: float = 10000.0,
                                trend_filter: bool = False, ema_fast: int = 50, ema_slow: int = 200,
                                vol_filter: bool = False, vol_lookback: int = 20, vol_multiplier: float = 1.0,
                                trailing_stop_pct: Optional[float] = None,
                                risk_per_trade: Optional[float] = None,
                                leverage: float = 1.0, min_qty: float = 0.0,
                                slippage_vs_volume: bool = False,
                                slippage_k: float = 0.0, slippage_cap: float = 0.05,
                                enable_logging: bool = True) -> Dict:
    """
    Enhanced aggressive intraday-style strategy with comprehensive analytics.

    Features:
    - Vectorized technical indicator calculations for performance
    - Comprehensive trade execution logging
    - Professional-grade performance metrics
    - Adaptive learning integration
    - Risk management and position sizing
    - Memory-efficient processing

    Args:
        df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        take_profit_pct: Take profit percentage (e.g., 0.004 = 0.4%)
        stop_loss_pct: Stop loss percentage (e.g., 0.002 = 0.2%)
        max_holding_bars: Maximum bars to hold position
        fee_pct: Trading fees percentage per side
        slippage_pct: Slippage percentage
        starting_balance: Initial capital
        trend_filter: Enable trend filtering with EMAs
        ema_fast: Fast EMA period for trend filter
        ema_slow: Slow EMA period for trend filter
        vol_filter: Enable volatility filtering
        vol_lookback: Lookback period for volatility calculation
        vol_multiplier: Volatility threshold multiplier
        trailing_stop_pct: Trailing stop percentage (optional)
        risk_per_trade: Risk per trade as fraction of balance
        leverage: Trading leverage
        min_qty: Minimum order quantity
        slippage_vs_volume: Enable volume-based slippage adjustment
        slippage_k: Volume slippage factor
        slippage_cap: Maximum slippage percentage
        enable_logging: Enable detailed trade logging

    Returns:
        Dictionary with comprehensive backtest results and analytics
    """
    start_time = time.time()
    
    if enable_logging:
        logger.info(f"Starting aggressive strategy backtest:")
        logger.info(f"  Parameters: TP={take_profit_pct*100:.2f}%, SL={stop_loss_pct*100:.2f}%, Hold={max_holding_bars}")
        logger.info(f"  Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        logger.info(f"  Filters: Trend={trend_filter}, Vol={vol_filter}")
    
    # Validate input data
    if df.empty:
        logger.error("Empty DataFrame provided")
        return {'error': 'Empty DataFrame'}
    
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return {'error': f'Missing columns: {missing_columns}'}
    
    # Make a copy and prepare data with vectorized operations
    df = df.copy()
    lookback = 5
    
    # Technical indicators using vectorized pandas operations
    logger.debug("Calculating technical indicators...")
    
    # Rolling high for breakout detection
    df['rolling_high'] = df['high'].shift(1).rolling(lookback, min_periods=1).max()
    
    # Trend filter EMAs
    if trend_filter:
        df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    
    # Volatility filter
    if vol_filter:
        df['ret'] = df['close'].pct_change()
        df['vol'] = df['ret'].rolling(vol_lookback).std()
    
    # Additional features for adaptive learning (vectorized)
    df['ret1'] = df['close'].pct_change().fillna(0.0)
    df['ma3'] = df['close'].rolling(3, min_periods=1).mean()
    df['mom5'] = df['close'].pct_change(5).fillna(0.0)
    df['mom10'] = df['close'].pct_change(10).fillna(0.0)
    
    # Volume-based features
    if 'volume' in df.columns:
        df['vol_mean20'] = df['volume'].rolling(20, min_periods=1).mean()
        df['vol_ratio'] = df['volume'] / df['vol_mean20'].replace(0, 1)
    else:
        df['vol_mean20'] = 1.0
        df['vol_ratio'] = 1.0
    
    # Volatility and ATR indicators
    df['vol20'] = df['ret1'].rolling(20).std().fillna(0.0)
    
    # True Range and ATR calculation (vectorized)
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14, min_periods=1).mean()
    
    # RSI calculation (vectorized)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = (-1 * delta).clip(lower=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, 1e-8)
    df['rsi14'] = 100.0 - (100.0 / (1.0 + rs))
    
    # Initialize trading variables
    trades = []
    detailed_trades = []
    position = None
    entry_idx = None
    entry_price = None
    balance = starting_balance
    equity_curve = [balance]
    
    # Risk management
    if risk_per_trade is None:
        risk_per_trade = 0.01
        
    # Adaptive learning setup
    trainer = OnlineTrainer()
    trainer.load()
    
    # Trading statistics
    stats = {
        'entries_attempted': 0,
        'entries_executed': 0,
        'entries_filtered_trend': 0,
        'entries_filtered_vol': 0,
        'entries_filtered_ml': 0,
        'exits_tp': 0,
        'exits_sl': 0,
        'exits_trailing': 0,
        'exits_time': 0
    }
    
    holding = 0
    last_features = None
    last_outcome = None
    peak_price = 0
    trailing_stop_price = None
    
    # Main trading loop with enhanced logging
    logger.debug(f"Starting trading simulation on {len(df)} bars...")
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        # Prepare features for ML
        features = {
            'close': float(row['close']),
            'high': float(row['high']),
            'low': float(row['low']),
            'volume': float(row.get('volume', 1)),
            'ret1': float(row['ret1']),
            'ma3': float(row['ma3']),
            'mom5': float(row['mom5']),
            'mom10': float(row['mom10']),
            'vol20': float(row['vol20']),
            'vol_ratio': float(row['vol_ratio']),
            'atr14': float(row['atr14']),
            'rsi14': float(row['rsi14'])
        }
        
        if position is None:
            # Check for entry signal
            if not pd.isna(row['rolling_high']) and row['close'] > row['rolling_high']:
                stats['entries_attempted'] += 1
                
                # ML-based filtering
                prob = trainer.predict_proba(features)
                if prob < 0.6:
                    stats['entries_filtered_ml'] += 1
                    if enable_logging and i % 100 == 0:
                        logger.debug(f"Entry filtered by ML at {timestamp}: prob={prob:.3f}")
                    continue
                
                # Trend filter
                if trend_filter:
                    if pd.isna(row.get('ema_fast')) or pd.isna(row.get('ema_slow')):
                        continue
                    if row['ema_fast'] <= row['ema_slow']:
                        stats['entries_filtered_trend'] += 1
                        continue
                
                # Volatility filter
                if vol_filter:
                    vol = row.get('vol', None)
                    if vol is None or pd.isna(vol):
                        continue
                    vol_median = df['vol'].median()
                    if vol_median == 0 or vol < vol_median * vol_multiplier:
                        stats['entries_filtered_vol'] += 1
                        continue
                
                # Execute entry
                position = 'long'
                entry_idx = timestamp
                entry_price = row['close']
                
                # Position sizing
                stop_price = entry_price * (1 - stop_loss_pct)
                try:
                    raw_qty, notional = choose_position_size(
                        balance, risk_per_trade, entry_price, stop_price, 
                        leverage=leverage, min_qty=min_qty
                    )
                    qty = round_qty(raw_qty, step=0.0001, min_qty=min_qty)
                except:
                    # Fallback if money_engine not available
                    notional = balance * risk_per_trade
                    qty = notional / entry_price
                
                trades.append({
                    'type': 'entry',
                    'time': timestamp,
                    'price': entry_price,
                    'qty': qty,
                    'notional': qty * entry_price
                })
                
                detailed_trades.append({
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'qty': qty,
                    'ml_prob': prob,
                    'features': features.copy()
                })
                
                holding = 0
                peak_price = entry_price
                trailing_stop_price = None
                last_features = features
                stats['entries_executed'] += 1
                
                if enable_logging and len(trades) <= 5:  # Log first few trades in detail
                    logger.info(f"ENTRY #{len(detailed_trades)}: {timestamp} @ {entry_price:.4f}, qty={qty:.4f}, ML prob={prob:.3f}")
        
        else:  # In position
            holding += 1
            
            # Update trailing stop
            if trailing_stop_pct is not None:
                if row['high'] > peak_price:
                    peak_price = row['high']
                trailing_stop_price = peak_price * (1 - trailing_stop_pct)
            
            # Check exit conditions
            exit_triggered = False
            exit_type = None
            exit_price = None
            
            # Take profit
            if row['high'] >= entry_price * (1 + take_profit_pct):
                exit_price = entry_price * (1 + take_profit_pct)
                exit_type = 'exit_tp'
                stats['exits_tp'] += 1
                last_outcome = 1  # Win
                exit_triggered = True
            
            # Stop loss
            elif row['low'] <= entry_price * (1 - stop_loss_pct):
                exit_price = entry_price * (1 - stop_loss_pct)
                exit_type = 'exit_sl'
                stats['exits_sl'] += 1
                last_outcome = 0  # Loss
                exit_triggered = True
            
            # Trailing stop
            elif trailing_stop_pct is not None and trailing_stop_price is not None and row['low'] <= trailing_stop_price:
                exit_price = trailing_stop_price
                exit_type = 'exit_trail'
                stats['exits_trailing'] += 1
                last_outcome = 1 if exit_price > entry_price else 0
                exit_triggered = True
            
            # Time-based exit
            elif holding >= max_holding_bars:
                exit_price = row['close']
                exit_type = 'exit_time'
                stats['exits_time'] += 1
                last_outcome = 1 if exit_price > entry_price else 0
                exit_triggered = True
            
            if exit_triggered:
                # Record exit
                trades.append({
                    'type': exit_type,
                    'time': timestamp,
                    'price': exit_price,
                    'qty': qty
                })
                
                # Calculate trade P&L with fees and slippage
                entry_px_adj = entry_price * (1 + slippage_pct)
                exit_px_adj = exit_price * (1 - slippage_pct)
                
                # Volume-based slippage adjustment
                if slippage_vs_volume and 'volume' in row:
                    try:
                        recent_vol = float(row['vol_mean20']) if not pd.isna(row['vol_mean20']) else 1.0
                        extra_slippage = min(slippage_k * (qty / max(recent_vol, 1e-8)), slippage_cap)
                        entry_px_adj *= (1 + extra_slippage)
                        exit_px_adj *= (1 - extra_slippage)
                    except:
                        pass
                
                # Calculate P&L
                trade_pnl_price = exit_px_adj - entry_px_adj
                trade_pnl = trade_pnl_price * qty
                
                # Apply fees
                fee_cost = (entry_px_adj * qty + exit_px_adj * qty) * fee_pct
                trade_pnl -= fee_cost
                
                # Update balance and equity curve
                balance += trade_pnl
                equity_curve.append(balance)
                
                # Complete trade record
                if detailed_trades:
                    detailed_trades[-1].update({
                        'exit_time': timestamp,
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'holding_bars': holding,
                        'pnl': trade_pnl,
                        'pnl_pct': (trade_pnl / (qty * entry_price)) * 100,
                        'outcome': last_outcome
                    })
                
                # ML learning
                if last_features is not None and last_outcome is not None:
                    trainer.learn_one(last_features, last_outcome)
                
                # Reset position
                position = None
                entry_idx = None
                entry_price = None
                holding = 0
                last_features = None
                last_outcome = None
                
                if enable_logging and len(detailed_trades) <= 5:  # Log first few trades in detail
                    logger.info(f"EXIT #{len(detailed_trades)}: {timestamp} @ {exit_price:.4f}, {exit_type}, PnL={trade_pnl:.2f} ({trade_pnl/(qty*entry_price)*100:.2f}%), hold={holding} bars")
    
    # Calculate final performance metrics
    execution_time = time.time() - start_time
    
    # Ensure we have equity curve
    if len(equity_curve) == 1:
        equity_curve.append(balance)
    
    # Calculate comprehensive metrics
    performance_metrics = calculate_performance_metrics(equity_curve, detailed_trades)
    
    # Compile results
    results = {
        'trades': len(detailed_trades),
        'wins': stats['exits_tp'] + sum(1 for t in detailed_trades if t.get('pnl', 0) > 0),
        'win_rate_pct': (stats['exits_tp'] / len(detailed_trades) * 100) if detailed_trades else 0,
        'pnl': balance - starting_balance,
        'final_balance': balance,
        'starting_balance': starting_balance,
        'execution_time_seconds': execution_time,
        'trading_stats': stats,
        'performance_metrics': performance_metrics,
        'equity_curve': equity_curve,
        'trade_list': detailed_trades[:100],  # Limit to first 100 trades for memory efficiency
        'details': {
            'entries': len([t for t in trades if t['type'] == 'entry']),
            'exits': len([t for t in trades if not t['type'] == 'entry']),
            'bars_processed': len(df),
            'avg_holding_period': np.mean([t.get('holding_bars', 0) for t in detailed_trades]) if detailed_trades else 0
        }
    }
    
    if enable_logging:
        logger.info("="*60)
        logger.info("BACKTEST COMPLETED")
        logger.info(f"Execution time: {execution_time:.2f}s")
        logger.info(f"Total trades: {results['trades']}")
        logger.info(f"Win rate: {results['win_rate_pct']:.1f}%")
        logger.info(f"Total PnL: {results['pnl']:.2f} ({(results['pnl']/starting_balance)*100:.1f}%)")
        if performance_metrics:
            logger.info(f"Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Max drawdown: {performance_metrics.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"Profit factor: {performance_metrics.get('profit_factor', 0):.2f}")
        logger.info("="*60)
    
    return results
    rs = roll_up / roll_down
    df['rsi14'] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(len(df)):
        row = df.iloc[i]
        features = {col: float(row[col]) for col in ['close','high','low','volume','ret1','ma3','mom5','mom10','vol20','vol_ratio','atr14','rsi14'] if col in row}
        if position is None:
            # entry condition: breakout
            if not pd.isna(row['rolling_high']) and row['close'] > row['rolling_high']:
                # Adaptive filter: only enter if model predicts high probability
                prob = trainer.predict_proba(features)
                if prob < 0.6:
                    continue
                # optional trend filter: only enter if ema_fast > ema_slow
                if trend_filter:
                    if pd.isna(row.get('ema_fast')) or pd.isna(row.get('ema_slow')):
                        continue
                    if row['ema_fast'] <= row['ema_slow']:
                        continue
                if vol_filter:
                    vol = row.get('vol', None)
                    if vol is None or pd.isna(vol):
                        continue
                    med = df['vol'].median()
                    if med == 0 or vol < med * vol_multiplier:
                        continue

                position = 'long'
                entry_idx = row.name
                entry_price = row['close']
                raw_qty, notional = choose_position_size(balance, risk_per_trade, entry_price, entry_price * (1 - stop_loss_pct), leverage=leverage, min_qty=min_qty)
                # round quantity to exchange lot and enforce minimums
                qty = round_qty(raw_qty, step=0.0001, min_qty=min_qty)
                notional = qty * entry_price
                trades.append({'type': 'entry', 'time': entry_idx, 'price': entry_price, 'qty': qty, 'notional': notional})
                holding = 0
                peak_price = entry_price
                trailing_stop_price = None
                last_features = features
        else:
            holding += 1
            if trailing_stop_pct is not None:
                if row['high'] > peak_price:
                    peak_price = row['high']
                trailing_stop_price = peak_price * (1 - trailing_stop_pct)

            # check TP/SL/trailing
            trade_closed = False
            if row['high'] >= entry_price * (1 + take_profit_pct):
                exit_price = entry_price * (1 + take_profit_pct)
                trades.append({'type': 'exit_tp', 'time': row.name, 'price': exit_price, 'qty': qty})
                last_outcome = 1  # Win
                trade_closed = True
            elif row['low'] <= entry_price * (1 - stop_loss_pct):
                exit_price = entry_price * (1 - stop_loss_pct)
                trades.append({'type': 'exit_sl', 'time': row.name, 'price': exit_price, 'qty': qty})
                last_outcome = 0  # Loss
                trade_closed = True
            elif trailing_stop_pct is not None and row['low'] <= trailing_stop_price:
                exit_price = trailing_stop_price
                trades.append({'type': 'exit_trail', 'time': row.name, 'price': exit_price, 'qty': qty})
                last_outcome = 1 if exit_price > entry_price else 0
                trade_closed = True
            elif holding >= max_holding_bars:
                exit_price = row['close']
                trades.append({'type': 'exit_hold', 'time': row.name, 'price': exit_price, 'qty': qty})
                last_outcome = 1 if exit_price > entry_price else 0
                trade_closed = True

            if trade_closed:
                # Learn from the outcome
                if last_features is not None and last_outcome is not None:
                    trainer.learn_one(last_features, last_outcome)
                position = None
                entry_idx = None
                entry_price = None
                holding = 0
                last_features = None
                last_outcome = None
                position = None
            elif row['low'] <= entry_price * (1 - stop_loss_pct):
                exit_price = entry_price * (1 - stop_loss_pct)
                trades.append({'type': 'exit_sl', 'time': row.name, 'price': exit_price, 'qty': qty})
                position = None
            elif trailing_stop_pct is not None and trailing_stop_price is not None and row['low'] <= trailing_stop_price:
                exit_price = trailing_stop_price
                trades.append({'type': 'exit_trail', 'time': row.name, 'price': exit_price, 'qty': qty})
                position = None
            elif holding >= max_holding_bars:
                exit_price = row['close']
                trades.append({'type': 'exit_time', 'time': row.name, 'price': exit_price, 'qty': qty})
                position = None

    # pair entries and exits into trade-level results and build equity curve
    entries = [t for t in trades if t['type'] == 'entry']
    exits = [t for t in trades if t['type'].startswith('exit')]
    n = min(len(entries), len(exits))
    wins = 0
    total = n
    pnl = 0.0
    trade_pairs = []
    equity = []
    bal = balance

    for i in range(n):
        e = entries[i]
        x = exits[i]
        qty = e.get('qty', 0.0)
        # ensure qty is rounded and non-negative
        qty = round_qty(qty, step=0.0001, min_qty=min_qty)
        # apply slippage: assume worse execution on entry and exit
        entry_px = e['price'] * (1 + slippage_pct)
        exit_px = x['price'] * (1 - slippage_pct)
        # optionally increase slippage when trade notional relative to recent volume is large
        if slippage_vs_volume:
            # attempt to read nearby vol_mean20 if present in df
            try:
                # use time index to lookup volume mean; fall back to simple average
                idx_time = pd.to_datetime(e['time'])
                if 'volume' in df.columns:
                    # compute a simple recent avg volume per bar (20-bar) if not present
                    vol_mean20 = df['volume'].rolling(20).mean()
                    if idx_time in vol_mean20.index:
                        recent_vol = float(vol_mean20.loc[idx_time]) if not pd.isna(vol_mean20.loc[idx_time]) else float(vol_mean20.mean())
                    else:
                        recent_vol = float(vol_mean20.mean())
                else:
                    recent_vol = 1.0
            except Exception:
                recent_vol = 1.0
            # avoid div by zero
            eps = 1e-8
            extra = slippage_k * (qty / max(recent_vol, eps))
            extra = min(extra, slippage_cap)
            entry_px = e['price'] * (1 + slippage_pct + extra)
            exit_px = x['price'] * (1 - slippage_pct - extra)
        trade_pnl_price = exit_px - entry_px
        # trade PnL before fees
        trade_pnl = trade_pnl_price * qty
        # fees: assume fee_pct applied to notional on both sides
        fee_cost = (entry_px * qty + exit_px * qty) * fee_pct
        trade_pnl = trade_pnl - fee_cost
        pnl += trade_pnl
        bal += trade_pnl
        if trade_pnl > 0:
            wins += 1
        trade_pairs.append({
            'entry_time': str(e['time']),
            'exit_time': str(x['time']),
            'entry_price': e['price'],
            'exit_price': x['price'],
            'qty': qty,
            'pnl': trade_pnl
        })
        equity.append({'time': str(x['time']), 'balance': bal})

    win_rate = (wins / total * 100) if total > 0 else 0.0
    return {
        'trades': total,
        'wins': wins,
        'win_rate_pct': win_rate,
        'pnl': pnl,
        'details': {'entries': len(entries), 'exits': len(exits)},
        'trade_list': trade_pairs,
        'equity_curve': equity,
    }

def main():
    """
    Main entry point for CCXT backtesting demonstration.
    
    This function demonstrates the basic usage of the enhanced backtesting framework:
    1. Fetches OHLCV data from the configured exchange
    2. Runs the aggressive strategy backtest with default parameters
    3. Displays comprehensive performance metrics and analytics
    
    The aggressive strategy uses:
    - Breakout entry signals (price closes above rolling high)
    - Tight take-profit and stop-loss exits
    - Time-based exit after maximum holding period
    - Adaptive machine learning filtering
    - Professional-grade performance analytics
    
    Example usage:
        python backtest_ccxt.py
    
    Environment variables used:
        EXCHANGE: Exchange name (default: binance)
        PAPER: Paper trading mode (default: true)
        
    Returns:
        None: Results are printed to console and logged to backtest.log
        
    Raises:
        Exception: If data fetching or backtesting fails
        
    Performance Metrics Displayed:
        - Total number of trades executed
        - Win count and win rate percentage  
        - Total profit/loss in price units
        - Sharpe ratio for risk-adjusted returns
        - Maximum drawdown percentage
        - Profit factor (average win / average loss)
        - Additional risk and performance metrics
        
    Note:
        This is a demonstration function. For production use, consider:
        - Using larger datasets for more robust results
        - Adjusting parameters based on market conditions
        - Implementing proper risk management
        - Adding position sizing based on account balance
    """
    symbol = 'BTC/USDT'
    print('Fetching', symbol)
    df = fetch_ohlcv(symbol)
    print('Running aggressive strategy backtest...')
    stats = aggressive_strategy_backtest(df)
    print('Trades:', stats['trades'])
    print('Wins:', stats['wins'])
    print(f"Win rate: {stats['win_rate_pct']:.2f}%")
    print('PnL (price units):', stats['pnl'])

if __name__ == '__main__':
    main()
