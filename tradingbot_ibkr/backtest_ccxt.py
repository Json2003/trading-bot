"""Minimal CCXT backtest runner (example).

This script loads OHLCV via ccxt (historical) and runs a tiny moving-average crossover
backtest for demonstration.
"""

import os
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
# Use package-relative imports
from .money_engine import choose_position_size, fixed_fractional, round_qty, round_price
# Adaptive learning
from .models.online_trainer import OnlineTrainer

load_dotenv()
EXCHANGE = os.getenv('EXCHANGE', 'binance')
PAPER = os.getenv('PAPER', 'true').lower() == 'true'

def fetch_ohlcv(symbol, timeframe='1h', limit=500):
    ex = getattr(ccxt, EXCHANGE)()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def simple_backtest(df):
    df['ma_fast'] = df['close'].rolling(10).mean()
    df['ma_slow'] = df['close'].rolling(30).mean()
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
        elif row['ma_fast'] < row['ma_slow'] and position == 1:
            position = 0
            exit_price = row['close']
            pnl += (exit_price - entry_price)
            trades.append(('sell', idx, exit_price))
    return pnl, trades


def aggressive_strategy_backtest(df, take_profit_pct=0.004, stop_loss_pct=0.002, max_holding_bars=12,
                                fee_pct=0.0, slippage_pct=0.0, starting_balance=10000.0,
                                trend_filter=False, ema_fast=50, ema_slow=200,
                                vol_filter=False, vol_lookback=20, vol_multiplier=1.0,
                                trailing_stop_pct=None,
                                # new sizing/execution params
                                risk_per_trade: float | None = None,
                                leverage: float = 1.0,
                                min_qty: float = 0.0,
                                # optional volume-based slippage
                                slippage_vs_volume: bool = False,
                                slippage_k: float = 0.0,
                                slippage_cap: float = 0.05):
    """Aggressive intraday-style strategy: enter on breakout and use tight TP/SL.

    Rules (example aggressive setup):
    - Entry: price closes above the rolling high of the last N bars (breakout)
    - Exit: take-profit at take_profit_pct, stop-loss at stop_loss_pct, or exit after max_holding_bars

    This is intentionally aggressive: many small trades, higher frequency.
    """
    lookback = 5
    df = df.copy()
    df['rolling_high'] = df['high'].shift(1).rolling(lookback).max()
    # trend filter: compute EMAs
    if trend_filter:
        df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    # volatility filter: rolling std of returns
    if vol_filter:
        df['ret'] = df['close'].pct_change()
        df['vol'] = df['ret'].rolling(vol_lookback).std()
    trades = []
    position = None
    entry_idx = None
    entry_price = None
    balance = starting_balance  # starting balance in quote currency (e.g., USDT / USD)
    # allow caller to override the per-trade risk; default to 1% if not provided
    if risk_per_trade is None:
        risk_per_trade = 0.01
    holding = 0
    # Initialize adaptive trainer
    trainer = OnlineTrainer()
    trainer.load()
    # Feature columns for learning (simple example: close, high, low, volume)
    feature_cols = ['close', 'high', 'low', 'volume']
    last_features = None
    last_outcome = None

    # compute matching features for online predictions
    # create columns used by trainer
    df['ret1'] = df['close'].pct_change().fillna(0.0)
    df['ma3'] = df['close'].rolling(3).mean().fillna(method='bfill')
    df['mom5'] = df['close'].pct_change(5).fillna(0.0)
    df['mom10'] = df['close'].pct_change(10).fillna(0.0)
    df['vol_mean20'] = df['volume'].rolling(20).mean().fillna(method='bfill') if 'volume' in df.columns else 0.0
    df['vol_ratio'] = df['volume'] / (df['vol_mean20'].replace(0, 1))
    df['vol20'] = df['ret1'].rolling(20).std().fillna(0.0)
    high_low = (df['high'] - df['low']).abs()
    high_pc = (df['high'] - df['close'].shift(1)).abs()
    low_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean().fillna(method='bfill')
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean().replace(0, 1e-8)
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
    """Main backtest runner - now supports multiple strategies"""
    symbol = 'BTC/USDT'
    print('Fetching', symbol)
    
    try:
        df = fetch_ohlcv(symbol)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using sample data for demonstration...")
        # Create simple sample data if fetch fails
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        periods = 100
        dates = [datetime.now() - timedelta(hours=i) for i in range(periods)]
        dates.reverse()
        
        # Simple random walk
        prices = [50000]  # Starting price
        for i in range(1, periods):
            change = np.random.normal(0, 0.02)  # 2% volatility
            prices.append(prices[-1] * (1 + change))
            
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(100, 1000) for _ in range(periods)]
        }, index=dates)
    
    print('Running aggressive strategy backtest...')
    stats = aggressive_strategy_backtest(df)
    print('=== Aggressive Strategy Results ===')
    print('Trades:', stats['trades'])
    print('Wins:', stats['wins'])
    print(f"Win rate: {stats['win_rate_pct']:.2f}%")
    print('PnL (price units):', stats['pnl'])
    
    # Try to run advanced strategies if available
    try:
        from advanced_strategies import run_all_strategies_backtest
        print('\n' + '='*50)
        print('Running Advanced Trading Strategies...')
        print('='*50)
        
        # Convert DataFrame to list of dicts for advanced strategies
        df_data = []
        for i in range(len(df)):
            row = df.iloc[i]
            df_data.append({
                'timestamp': row.name,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
            
        advanced_results = run_all_strategies_backtest(df_data)
        
        if 'comparison' in advanced_results:
            comparison = advanced_results['comparison']
            summary = comparison.get('summary', {})
            if summary:
                print(f"\n=== Advanced Strategies Summary ===")
                print(f"Best Performing: {summary.get('best_performing', 'N/A')}")
                print(f"Most Consistent: {summary.get('most_consistent', 'N/A')}")
                print(f"Average PnL: {summary.get('avg_pnl', 0):.2f}")
                
    except ImportError as e:
        print(f"\nAdvanced strategies not available: {e}")
        print("Run 'python advanced_strategies.py' separately to test new strategies")
    except Exception as e:
        print(f"\nError running advanced strategies: {e}")

if __name__ == '__main__':
    main()
