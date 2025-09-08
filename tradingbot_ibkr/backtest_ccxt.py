"""Enhanced CCXT backtest runner with comprehensive logging and vectorized operations.

This script provides a robust backtesting framework with:
- Comprehensive trade execution logging
- Vectorized operations for improved performance
- Enhanced statistical analysis and reporting
- Detailed trade analytics and performance metrics
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("python-dotenv not available, skipping .env file loading")

try:
    import ccxt
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("ccxt not available, some functionality will be limited")
    ccxt = None

# Use our pandas utility for better compatibility
try:
    from .utils.pandas_utils import get_pandas, is_using_real_pandas, safe_date_range, safe_to_datetime
    pd, using_real_pandas = get_pandas()
except ImportError:
    # Fallback to direct pandas import
    try:
        import pandas as pd
        using_real_pandas = True
    except ImportError:
        from . import pandas as pd
        using_real_pandas = False

import numpy as np

# Use package-relative imports with fallbacks
try:
    from .money_engine import choose_position_size, fixed_fractional, round_qty, round_price
except ImportError:
    logger.warning("money_engine not available - using fallback implementations")
    # Fallback implementations
    def choose_position_size(balance, risk_pct, entry_price, stop_price, leverage=1.0, min_qty=0.0):
        risk_amount = balance * risk_pct
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit == 0:
            return 0.0, 0.0
        qty = risk_amount / risk_per_unit
        return max(qty, min_qty), qty * entry_price
    
    def fixed_fractional(balance, risk_pct):
        return balance * risk_pct
    
    def round_qty(qty, step=0.001, min_qty=0.0):
        return max(round(qty / step) * step, min_qty)
    
    def round_price(price, step=0.01):
        return round(price / step) * step

# Adaptive learning - make optional
try:
    from .models.online_trainer import OnlineTrainer
except ImportError:
    logger.warning("OnlineTrainer not available - ML features will be disabled")
    # Create a dummy OnlineTrainer for compatibility
    class OnlineTrainer:
        def load(self): pass
        def predict_proba(self, features): return 0.6  # Default confidence
        def learn_one(self, features, outcome): pass

EXCHANGE = os.getenv('EXCHANGE', 'binance')
PAPER = os.getenv('PAPER', 'true').lower() == 'true'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BacktestLogger:
    """Comprehensive logging for backtest execution details."""
    
    def __init__(self, strategy_name: str = "backtest"):
        self.strategy_name = strategy_name
        self.trade_logs = []
        self.performance_logs = []
        self.signal_logs = []
        self.start_time = time.time()
        
    def log_signal(self, timestamp: str, signal_type: str, price: float, 
                  features: Dict[str, float], probability: float = None):
        """Log trading signal generation."""
        self.signal_logs.append({
            'timestamp': timestamp,
            'signal_type': signal_type,
            'price': price,
            'features': features,
            'probability': probability
        })
        logger.debug(f"Signal {signal_type} at {timestamp}: price={price:.4f}, prob={probability:.3f}")
    
    def log_trade_entry(self, timestamp: str, price: float, qty: float, 
                       notional: float, stop_loss: float, take_profit: float):
        """Log trade entry with full details."""
        entry_log = {
            'type': 'entry',
            'timestamp': timestamp,
            'price': price,
            'quantity': qty,
            'notional': notional,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': notional * abs(price - stop_loss) / price
        }
        self.trade_logs.append(entry_log)
        logger.info(f"ENTRY: {timestamp} - Price: ${price:.4f}, Qty: {qty:.4f}, "
                   f"Notional: ${notional:.2f}, SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
    
    def log_trade_exit(self, timestamp: str, exit_type: str, price: float, 
                      qty: float, pnl: float, fees: float, holding_periods: int):
        """Log trade exit with performance details."""
        exit_log = {
            'type': 'exit',
            'exit_type': exit_type,
            'timestamp': timestamp,
            'price': price,
            'quantity': qty,
            'pnl_gross': pnl + fees,
            'fees': fees,
            'pnl_net': pnl,
            'holding_periods': holding_periods
        }
        self.trade_logs.append(exit_log)
        logger.info(f"EXIT ({exit_type}): {timestamp} - Price: ${price:.4f}, "
                   f"PnL: ${pnl:.2f}, Fees: ${fees:.2f}, Held: {holding_periods} bars")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics during backtest."""
        self.performance_logs.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        execution_time = time.time() - self.start_time
        return {
            'strategy_name': self.strategy_name,
            'execution_time_seconds': execution_time,
            'total_signals': len(self.signal_logs),
            'total_entries': len([log for log in self.trade_logs if log['type'] == 'entry']),
            'total_exits': len([log for log in self.trade_logs if log['type'] == 'exit']),
            'signal_logs': self.signal_logs,
            'trade_logs': self.trade_logs,
            'performance_logs': self.performance_logs
        }


def vectorized_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators using vectorized operations for performance.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    logger.debug("Calculating technical indicators")
    
    # Check pandas capabilities
    has_copy = hasattr(df, 'copy')
    has_rolling = hasattr(df, 'rolling') if hasattr(df, '__class__') else False
    
    # Make a copy if possible, otherwise work with original
    if has_copy:
        data = df.copy()
    else:
        logger.warning("DataFrame.copy() not available - working with original data")
        data = df
    
    # Use real pandas methods when available, fallback for custom pandas
    if not using_real_pandas or not has_rolling:
        logger.warning("Limited pandas functionality - returning basic data with minimal indicators")
        
        # Only add very basic indicators that work with custom pandas
        try:
            # Simple price-based features
            close_data = [row.get('close', 0) for row in data._rows] if hasattr(data, '_rows') else []
            if close_data:
                # Simple return calculation
                data._rows[0]['ret1'] = 0.0
                for i in range(1, len(data._rows)):
                    if close_data[i-1] != 0:
                        data._rows[i]['ret1'] = (close_data[i] - close_data[i-1]) / close_data[i-1]
                    else:
                        data._rows[i]['ret1'] = 0.0
                        
                logger.debug("Added basic return calculation")
        except Exception as e:
            logger.warning(f"Could not add basic indicators: {e}")
        
        return data
    
    # Price-based indicators
    data['ret1'] = data['close'].pct_change().fillna(0.0)
    data['ma3'] = data['close'].rolling(3).mean()
    data['ma10'] = data['close'].rolling(10).mean()
    data['ma20'] = data['close'].rolling(20).mean()
    data['ma50'] = data['close'].rolling(50).mean()
    
    # Momentum indicators
    data['mom5'] = data['close'].pct_change(5).fillna(0.0)
    data['mom10'] = data['close'].pct_change(10).fillna(0.0)
    data['mom20'] = data['close'].pct_change(20).fillna(0.0)
    
    # Volatility indicators
    data['vol20'] = data['ret1'].rolling(20).std().fillna(0.0)
    data['vol_mean20'] = data['volume'].rolling(20).mean() if 'volume' in data.columns else pd.Series(index=data.index).fillna(1.0)
    data['vol_ratio'] = data['volume'] / data['vol_mean20'].replace(0, 1) if 'volume' in data.columns else 1.0
    
    # ATR (Average True Range) - vectorized calculation
    high_low = data['high'] - data['low']
    high_pc = (data['high'] - data['close'].shift(1)).abs()
    low_pc = (data['low'] - data['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    data['atr14'] = true_range.rolling(14).mean()
    
    # RSI (Relative Strength Index) - vectorized calculation
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    data['rsi14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # Support/Resistance levels using rolling min/max
    data['support_5'] = data['low'].rolling(5).min()
    data['resistance_5'] = data['high'].rolling(5).max()
    data['support_20'] = data['low'].rolling(20).min()
    data['resistance_20'] = data['high'].rolling(20).max()
    
    # Fill NaN values using forward fill and backward fill
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    # Handle fillna method differences between pandas versions
    try:
        data[numeric_columns] = data[numeric_columns].fillna(method='bfill').fillna(method='ffill').fillna(0)
    except (TypeError, AttributeError):
        # Newer pandas versions
        data[numeric_columns] = data[numeric_columns].bfill().ffill().fillna(0)
    
    logger.debug(f"Technical indicators calculated for {len(data)} bars")
    return data


def enhanced_trade_analysis(trades: List[Dict[str, Any]], 
                           equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform comprehensive trade analysis with compatibility for both pandas implementations.
    
    Args:
        trades: List of trade dictionaries
        equity_curve: List of equity curve points
        
    Returns:
        Dictionary with detailed analytics
    """
    if not trades:
        return {'error': 'No trades to analyze'}
    
    # Basic statistics using simple calculations for compatibility
    total_trades = len(trades)
    pnl_values = [trade['pnl'] for trade in trades]
    
    winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
    losing_trades = sum(1 for pnl in pnl_values if pnl <= 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # PnL statistics
    total_pnl = sum(pnl_values)
    winning_pnls = [pnl for pnl in pnl_values if pnl > 0]
    losing_pnls = [pnl for pnl in pnl_values if pnl <= 0]
    
    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
    
    # Risk metrics
    if losing_trades > 0 and avg_loss != 0:
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
    else:
        profit_factor = float('inf')
    
    # Consecutive wins/losses analysis
    consecutive_wins = []
    consecutive_losses = []
    current_streak = 0
    current_type = None
    
    for pnl in pnl_values:
        is_win = pnl > 0
        if current_type is None or current_type != is_win:
            if current_type is not None:
                if current_type:
                    consecutive_wins.append(current_streak)
                else:
                    consecutive_losses.append(current_streak)
            current_streak = 1
            current_type = is_win
        else:
            current_streak += 1
    
    # Add final streak
    if current_type is not None:
        if current_type:
            consecutive_wins.append(current_streak)
        else:
            consecutive_losses.append(current_streak)
    
    max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
    max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
    
    # Basic drawdown calculation
    max_drawdown_pct = 0
    if equity_curve:
        balances = [point['balance'] for point in equity_curve]
        running_max = balances[0]
        max_drawdown = 0
        
        for balance in balances:
            if balance > running_max:
                running_max = balance
            else:
                drawdown = (running_max - balance) / running_max
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        max_drawdown_pct = max_drawdown * 100
    
    # Standard deviation calculation
    if len(pnl_values) > 1:
        mean_pnl = total_pnl / len(pnl_values)
        variance = sum((pnl - mean_pnl) ** 2 for pnl in pnl_values) / (len(pnl_values) - 1)
        std_deviation = variance ** 0.5
    else:
        std_deviation = 0
    
    # Return comprehensive analysis
    analysis = {
        'trade_count': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_pct': round(win_rate, 2),
        'total_pnl': round(total_pnl, 2),
        'average_win': round(avg_win, 2),
        'average_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'largest_win': round(max(pnl_values), 2) if pnl_values else 0,
        'largest_loss': round(min(pnl_values), 2) if pnl_values else 0,
        'std_deviation': round(std_deviation, 2),
        'sharpe_ratio': round(total_pnl / std_deviation, 2) if std_deviation != 0 else 0
    }
    
    logger.info(f"Trade analysis completed: {total_trades} trades, "
               f"{win_rate:.1f}% win rate, ${total_pnl:.2f} total PnL")
    
    return analysis

def fetch_ohlcv(symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data from exchange with error handling and logging.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Candle timeframe
        limit: Number of candles to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    if ccxt is None:
        raise ImportError("ccxt is not available - cannot fetch live data")
        
    logger.info(f"Fetching {limit} {timeframe} candles for {symbol} from {EXCHANGE}")
    
    try:
        ex = getattr(ccxt, EXCHANGE)()
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not data:
            raise ValueError("No data returned from exchange")
            
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        
        # Use safe datetime conversion
        if hasattr(pd, 'to_datetime'):
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        else:
            df['ts'] = safe_to_datetime([d / 1000 for d in df['ts']])
            
        df.set_index('ts', inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        raise


def simple_backtest(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 30) -> Tuple[float, List[Tuple]]:
    """Simple moving average crossover strategy with enhanced logging.
    
    Args:
        df: OHLCV DataFrame
        fast_period: Fast moving average period
        slow_period: Slow moving average period
        
    Returns:
        Tuple of (total_pnl, trades_list)
    """
    logger.info(f"Running simple MA crossover backtest (fast={fast_period}, slow={slow_period})")
    
    # Calculate indicators using vectorized operations
    df = df.copy()
    df['ma_fast'] = df['close'].rolling(fast_period).mean()
    df['ma_slow'] = df['close'].rolling(slow_period).mean()
    df.dropna(inplace=True)
    
    position = 0
    entry_price = 0
    pnl = 0
    trades = []
    
    for idx, row in df.iterrows():
        if row['ma_fast'] > row['ma_slow'] and position == 0:
            position = 1
            entry_price = row['close']
            trades.append(('buy', idx, entry_price))
            logger.debug(f"BUY signal at {idx}: price={entry_price:.4f}")
        elif row['ma_fast'] < row['ma_slow'] and position == 1:
            position = 0
            exit_price = row['close']
            trade_pnl = exit_price - entry_price
            pnl += trade_pnl
            trades.append(('sell', idx, exit_price))
            logger.debug(f"SELL signal at {idx}: price={exit_price:.4f}, PnL={trade_pnl:.4f}")
    
    logger.info(f"Simple backtest completed: {len(trades)//2} trades, ${pnl:.2f} total PnL")
    return pnl, trades


def aggressive_strategy_backtest(df: pd.DataFrame, take_profit_pct: float = 0.004, stop_loss_pct: float = 0.002, 
                                max_holding_bars: int = 12, fee_pct: float = 0.0, slippage_pct: float = 0.0, 
                                starting_balance: float = 10000.0, trend_filter: bool = False, 
                                ema_fast: int = 50, ema_slow: int = 200, vol_filter: bool = False, 
                                vol_lookback: int = 20, vol_multiplier: float = 1.0,
                                trailing_stop_pct: Optional[float] = None, risk_per_trade: Optional[float] = None,
                                leverage: float = 1.0, min_qty: float = 0.0, slippage_vs_volume: bool = False,
                                slippage_k: float = 0.0, slippage_cap: float = 0.05, 
                                enable_logging: bool = True) -> Dict[str, Any]:
    """Enhanced aggressive intraday strategy with comprehensive logging and vectorized operations.

    Features:
    - Comprehensive trade execution logging
    - Vectorized technical indicator calculations
    - Enhanced performance analytics
    - Detailed trade-by-trade reporting
    
    Strategy Rules:
    - Entry: Price closes above rolling high (breakout) with optional filters
    - Exit: Take-profit, stop-loss, trailing stop, or time-based exit
    - Risk management with position sizing and adaptive learning
    
    Args:
        df: OHLCV DataFrame with price data
        take_profit_pct: Take profit percentage (e.g., 0.004 = 0.4%)
        stop_loss_pct: Stop loss percentage (e.g., 0.002 = 0.2%)
        max_holding_bars: Maximum bars to hold position
        fee_pct: Trading fee percentage
        slippage_pct: Slippage percentage
        starting_balance: Initial balance in quote currency
        trend_filter: Enable trend filter using EMAs
        ema_fast: Fast EMA period for trend filter
        ema_slow: Slow EMA period for trend filter
        vol_filter: Enable volatility filter
        vol_lookback: Volatility lookback period
        vol_multiplier: Volatility threshold multiplier
        trailing_stop_pct: Trailing stop percentage (optional)
        risk_per_trade: Risk per trade as fraction of balance
        leverage: Leverage multiplier
        min_qty: Minimum quantity for trades
        slippage_vs_volume: Enable volume-based slippage calculation
        slippage_k: Volume slippage factor
        slippage_cap: Maximum additional slippage
        enable_logging: Enable comprehensive trade logging
        
    Returns:
        Dictionary with backtest results and detailed analytics
    """
    start_time = time.time()
    logger.info(f"Starting aggressive strategy backtest on {len(df)} bars")
    logger.info(f"Parameters: TP={take_profit_pct:.1%}, SL={stop_loss_pct:.1%}, "
               f"Hold={max_holding_bars}, Fee={fee_pct:.1%}, Slippage={slippage_pct:.1%}")
    
    # Initialize logging
    backtest_logger = BacktestLogger("aggressive_strategy") if enable_logging else None
    
    # Prepare data with vectorized technical indicators
    logger.debug("Calculating technical indicators...")
    df = vectorized_technical_indicators(df)
    
    # Strategy-specific indicators
    lookback = 5
    df['rolling_high'] = df['high'].shift(1).rolling(lookback).max()
    
    # Trend filter indicators
    if trend_filter:
        logger.debug(f"Applying trend filter: EMA({ema_fast}) vs EMA({ema_slow})")
        df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    
    # Volatility filter
    if vol_filter:
        logger.debug(f"Applying volatility filter: {vol_lookback}-period lookback, {vol_multiplier}x multiplier")
        df['vol_filter'] = df['vol20'] > df['vol20'].rolling(vol_lookback).median() * vol_multiplier
    
    # Initialize trading variables
    trades = []
    position = None
    entry_idx = None
    entry_price = None
    balance = starting_balance
    risk_per_trade = risk_per_trade or 0.01  # Default 1% risk per trade
    holding = 0
    peak_price = 0
    trailing_stop_price = None
    
    # Initialize adaptive trainer
    trainer = OnlineTrainer()
    trainer.load()
    last_features = None
    last_outcome = None
    
    # Feature columns for machine learning
    feature_columns = ['close', 'high', 'low', 'volume', 'ret1', 'ma3', 'mom5', 'mom10', 
                      'vol20', 'vol_ratio', 'atr14', 'rsi14', 'bb_position', 'macd']
    
    # Main trading loop - optimized for performance
    signal_count = 0
    entry_signals = 0
    
    logger.debug("Starting main trading loop...")
    
    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = str(row.name)
        
        # Extract features for ML prediction
        features = {}
        for col in feature_columns:
            if col in df.columns and not pd.isna(row[col]):
                features[col] = float(row[col])
            else:
                features[col] = 0.0
        
        if position is None:
            # Entry logic
            entry_condition = (not pd.isna(row['rolling_high']) and row['close'] > row['rolling_high'])
            
            if entry_condition:
                signal_count += 1
                
                # ML-based entry filter
                prob = trainer.predict_proba(features) if features else 0.5
                
                if backtest_logger:
                    backtest_logger.log_signal(timestamp, 'entry_candidate', row['close'], features, prob)
                
                if prob < 0.6:  # Require high confidence
                    continue
                
                # Trend filter
                if trend_filter:
                    if pd.isna(row.get('ema_fast')) or pd.isna(row.get('ema_slow')):
                        continue
                    if row['ema_fast'] <= row['ema_slow']:
                        continue
                
                # Volatility filter
                if vol_filter and not row.get('vol_filter', True):
                    continue
                
                # Execute entry
                position = 'long'
                entry_idx = row.name
                entry_price = row['close']
                
                # Position sizing
                stop_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
                
                raw_qty, notional = choose_position_size(
                    balance, risk_per_trade, entry_price, stop_price, 
                    leverage=leverage, min_qty=min_qty
                )
                qty = round_qty(raw_qty, step=0.0001, min_qty=min_qty)
                notional = qty * entry_price
                
                trades.append({
                    'type': 'entry',
                    'time': entry_idx,
                    'price': entry_price,
                    'qty': qty,
                    'notional': notional
                })
                
                # Initialize position tracking
                holding = 0
                peak_price = entry_price
                trailing_stop_price = None
                last_features = features
                entry_signals += 1
                
                if backtest_logger:
                    backtest_logger.log_trade_entry(timestamp, entry_price, qty, notional, 
                                                   stop_price, take_profit_price)
        else:
            # Position management
            holding += 1
            
            # Update trailing stop
            if trailing_stop_pct is not None:
                if row['high'] > peak_price:
                    peak_price = row['high']
                trailing_stop_price = peak_price * (1 - trailing_stop_pct)
            
            # Exit conditions - vectorized where possible
            exit_type = None
            exit_price = None
            
            if row['high'] >= entry_price * (1 + take_profit_pct):
                exit_type = 'take_profit'
                exit_price = entry_price * (1 + take_profit_pct)
                last_outcome = 1  # Win
            elif row['low'] <= entry_price * (1 - stop_loss_pct):
                exit_type = 'stop_loss'
                exit_price = entry_price * (1 - stop_loss_pct)
                last_outcome = 0  # Loss
            elif trailing_stop_pct is not None and trailing_stop_price is not None and row['low'] <= trailing_stop_price:
                exit_type = 'trailing_stop'
                exit_price = trailing_stop_price
                last_outcome = 1 if exit_price > entry_price else 0
            elif holding >= max_holding_bars:
                exit_type = 'time_exit'
                exit_price = row['close']
                last_outcome = 1 if exit_price > entry_price else 0
            
            if exit_type:
                # Execute exit
                trades.append({
                    'type': f'exit_{exit_type}',
                    'time': row.name,
                    'price': exit_price,
                    'qty': qty
                })
                
                # Learn from outcome
                if last_features and last_outcome is not None:
                    trainer.learn_one(last_features, last_outcome)
                
                if backtest_logger:
                    # Calculate preliminary PnL for logging
                    gross_pnl = (exit_price - entry_price) * qty
                    fees = (entry_price * qty + exit_price * qty) * fee_pct
                    net_pnl = gross_pnl - fees
                    backtest_logger.log_trade_exit(timestamp, exit_type, exit_price, qty, 
                                                  net_pnl, fees, holding)
                
                # Reset position
                position = None
                entry_idx = None
                entry_price = None
                holding = 0
                last_features = None
                last_outcome = None
    
    execution_time = time.time() - start_time
    logger.info(f"Trading loop completed in {execution_time:.2f}s - "
               f"{signal_count} signals, {entry_signals} entries")
    
    # Process trades and calculate performance using vectorized operations
    logger.debug("Processing trades and calculating performance...")
    
    entries = [t for t in trades if t['type'] == 'entry']
    exits = [t for t in trades if t['type'].startswith('exit')]
    n_completed_trades = min(len(entries), len(exits))
    
    if n_completed_trades == 0:
        logger.warning("No completed trades found")
        return {
            'trades': 0,
            'wins': 0,
            'win_rate_pct': 0.0,
            'pnl': 0.0,
            'execution_time': execution_time,
            'details': {'entries': len(entries), 'exits': len(exits)},
            'trade_list': [],
            'equity_curve': [],
            'analytics': {},
            'execution_log': backtest_logger.get_execution_summary() if backtest_logger else None
        }
    
    # Vectorized trade processing
    trade_pairs = []
    equity_curve = []
    current_balance = balance
    wins = 0
    total_pnl = 0.0
    
    for i in range(n_completed_trades):
        entry = entries[i]
        exit = exits[i]
        qty = entry.get('qty', 0.0)
        
        # Apply slippage
        entry_px = entry['price'] * (1 + slippage_pct)
        exit_px = exit['price'] * (1 - slippage_pct)
        
        # Volume-based slippage adjustment
        if slippage_vs_volume and 'volume' in df.columns:
            try:
                vol_mean = df['vol_mean20'].loc[pd.to_datetime(entry['time'])]
                if not pd.isna(vol_mean) and vol_mean > 0:
                    extra_slippage = min(slippage_k * (qty / vol_mean), slippage_cap)
                    entry_px *= (1 + extra_slippage)
                    exit_px *= (1 - extra_slippage)
            except (KeyError, IndexError):
                pass
        
        # Calculate trade PnL
        gross_pnl = (exit_px - entry_px) * qty
        fees = (entry_px * qty + exit_px * qty) * fee_pct
        net_pnl = gross_pnl - fees
        
        total_pnl += net_pnl
        current_balance += net_pnl
        
        if net_pnl > 0:
            wins += 1
        
        # Store trade details
        trade_pairs.append({
            'entry_time': str(entry['time']),
            'exit_time': str(exit['time']),
            'entry_price': entry['price'],
            'exit_price': exit['price'],
            'qty': qty,
            'pnl': net_pnl,
            'exit_type': exit['type']
        })
        
        equity_curve.append({
            'time': str(exit['time']),
            'balance': current_balance
        })
    
    # Calculate performance metrics
    win_rate = (wins / n_completed_trades * 100) if n_completed_trades > 0 else 0.0
    
    # Enhanced analytics
    analytics = enhanced_trade_analysis(trade_pairs, equity_curve) if trade_pairs else {}
    
    # Log performance metrics
    if backtest_logger:
        performance_metrics = {
            'total_trades': n_completed_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_balance': current_balance,
            'return_pct': (current_balance - balance) / balance * 100
        }
        backtest_logger.log_performance_metrics(performance_metrics)
    
    logger.info(f"Backtest completed: {n_completed_trades} trades, "
               f"{win_rate:.2f}% win rate, ${total_pnl:.2f} PnL")
    
    # Return comprehensive results
    return {
        'trades': n_completed_trades,
        'wins': wins,
        'win_rate_pct': win_rate,
        'pnl': total_pnl,
        'final_balance': current_balance,
        'return_pct': (current_balance - balance) / balance * 100,
        'execution_time': execution_time,
        'signals_generated': signal_count,
        'entries_executed': entry_signals,
        'details': {
            'entries': len(entries), 
            'exits': len(exits),
            'parameters': {
                'take_profit_pct': take_profit_pct,
                'stop_loss_pct': stop_loss_pct,
                'max_holding_bars': max_holding_bars,
                'fee_pct': fee_pct,
                'slippage_pct': slippage_pct,
                'risk_per_trade': risk_per_trade
            }
        },
        'trade_list': trade_pairs,
        'equity_curve': equity_curve,
        'analytics': analytics,
        'execution_log': backtest_logger.get_execution_summary() if backtest_logger else None
    }
def main():
    """Main function demonstrating enhanced backtesting capabilities."""
    symbol = 'BTC/USDT'
    
    logger.info(f"Starting enhanced backtest demonstration for {symbol}")
    
    try:
        # Check if we can fetch live data
        if ccxt is not None:
            # Fetch data from exchange
            df = fetch_ohlcv(symbol, timeframe='1h', limit=1000)
        else:
            # Create sample data for demonstration when ccxt is not available
            logger.warning("CCXT not available - creating sample data for demonstration")
            
            # Create realistic sample OHLCV data
            dates = safe_date_range(start='2023-01-01', periods=1000, freq='1h')
            np.random.seed(42)  # For reproducible results
            
            # Generate realistic price data with trend and volatility
            base_price = 30000
            returns = np.random.normal(0.0001, 0.02, 1000).cumsum()
            prices = base_price * np.exp(returns)
            
            data = {
                'open': prices + np.random.normal(0, prices * 0.001),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 1000))),
                'close': prices,
                'volume': np.random.lognormal(8, 0.5, 1000)  # Realistic volume distribution
            }
            
            df = pd.DataFrame(data, index=dates)
            # Ensure OHLC consistency
            df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
            df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        # Run simple backtest first
        logger.info("Running simple MA crossover backtest...")
        simple_pnl, simple_trades = simple_backtest(df, fast_period=10, slow_period=30)
        logger.info(f"Simple backtest: {len(simple_trades)//2} trades, ${simple_pnl:.2f} PnL")
        
        # Run enhanced aggressive strategy backtest
        logger.info("Running enhanced aggressive strategy backtest...")
        stats = aggressive_strategy_backtest(
            df,
            take_profit_pct=0.006,
            stop_loss_pct=0.003,
            max_holding_bars=24,
            fee_pct=0.001,
            slippage_pct=0.0005,
            starting_balance=10000.0,
            trend_filter=True,
            vol_filter=True,
            trailing_stop_pct=0.005,
            enable_logging=True
        )
        
        # Display comprehensive results
        print("\n" + "="*60)
        print("ENHANCED BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {stats['trades']}")
        print(f"Winning Trades: {stats['wins']}")
        print(f"Win Rate: {stats['win_rate_pct']:.2f}%")
        print(f"Total PnL: ${stats['pnl']:.2f}")
        print(f"Final Balance: ${stats['final_balance']:.2f}")
        print(f"Return: {stats['return_pct']:.2f}%")
        print(f"Execution Time: {stats['execution_time']:.2f}s")
        print(f"Signals Generated: {stats['signals_generated']}")
        print(f"Entries Executed: {stats['entries_executed']}")
        
        # Display analytics if available
        if stats.get('analytics'):
            analytics = stats['analytics']
            print("\nADVANCED ANALYTICS:")
            print("-" * 30)
            print(f"Profit Factor: {analytics.get('profit_factor', 'N/A')}")
            print(f"Max Drawdown: {analytics.get('max_drawdown_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {analytics.get('sharpe_ratio', 'N/A')}")
            print(f"Average Win: ${analytics.get('average_win', 0):.2f}")
            print(f"Average Loss: ${analytics.get('average_loss', 0):.2f}")
            print(f"Max Consecutive Wins: {analytics.get('max_consecutive_wins', 0)}")
            print(f"Max Consecutive Losses: {analytics.get('max_consecutive_losses', 0)}")
        
        # Show sample trades
        if stats.get('trade_list'):
            print(f"\nSAMPLE TRADES (first 5):")
            print("-" * 50)
            for i, trade in enumerate(stats['trade_list'][:5]):
                print(f"{i+1}. {trade['entry_time']} -> {trade['exit_time']}")
                print(f"   {trade['entry_price']:.4f} -> {trade['exit_price']:.4f} "
                      f"(${trade['pnl']:.2f}) [{trade['exit_type']}]")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == '__main__':
    main()
