"""Minimal CCXT live paper/live trading example.

This script demonstrates placing a market order (paper) or printing the order
when PAPER=true. It's intentionally minimal and requires you to set the
API keys in .env.
"""
import os
from dotenv import load_dotenv
import ccxt
from money_engine import choose_position_size, round_qty
from data.store import append_trade_record
from datetime import datetime, timezone
from asset_classes import AssetClass

load_dotenv()
EXCHANGE = os.getenv('EXCHANGE', 'binance')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
PAPER = os.getenv('PAPER', 'true').lower() == 'true'
ALLOW_LIVE_RISK = os.getenv('ALLOW_LIVE_RISK', 'false').lower() == 'true'
ASSET_CLASS = AssetClass(os.getenv('ASSET_CLASS', 'crypto'))

RISK_PER_TRADE = {
    AssetClass.CRYPTO: 0.02,
    AssetClass.FOREX: 0.01,
    AssetClass.STOCKS: 0.02,
    AssetClass.FUTURES: 0.03,
    AssetClass.OPTIONS: 0.05,
}


def build_exchange():
    ex_class = getattr(ccxt, EXCHANGE)
    ex = ex_class({'apiKey': API_KEY, 'secret': API_SECRET})
    return ex


def place_market_order(ex, symbol, side, amount):
    if PAPER or not ALLOW_LIVE_RISK:
        print(f"SAFE MODE: PAPER={PAPER}, ALLOW_LIVE_RISK={ALLOW_LIVE_RISK} -> would {side} {amount} {symbol}")
        return None
    print('Placing real order')
    order = ex.create_market_order(symbol, side, amount)
    return order


def main():
    ex = build_exchange()
    symbol = 'BTC/USDT'
    side = 'buy'
    balance = 1000.0
    entry_price = 30000.0
    stop_loss_price = 29700.0
    risk_pct = RISK_PER_TRADE.get(ASSET_CLASS, 0.02)
    qty, notional = choose_position_size(balance, risk_pct, entry_price, stop_loss_price)
    qty = round_qty(qty, step=0.0001, min_qty=0.0001)
    print('Exchange:', EXCHANGE, 'PAPER:', PAPER, 'ALLOW_LIVE_RISK:', ALLOW_LIVE_RISK, 'ASSET_CLASS:', ASSET_CLASS.value)
    print(f'Would place {side} {qty} {symbol} (notional {notional})')
    res = place_market_order(ex, symbol, side, qty)
    print('Result:', res)

    # Simulate a closed trade record for evaluation when in PAPER mode
    if res is None:
        entry_ts = datetime.now(timezone.utc).isoformat()
        # simulate a short hold then exit
        exit_ts = datetime.now(timezone.utc).isoformat()
        trade = {
            'trade_id': f'sim-{entry_ts}',
            'symbol': symbol,
            'entry_ts': entry_ts,
            'exit_ts': exit_ts,
            'entry_price': entry_price,
            'exit_price': entry_price * 1.002,  # small simulated move
            'size': float(qty),
            'side': side,
            'fees': 0.0,
            'pnl': float((entry_price * 1.002 - entry_price) * qty),
            'pnl_pct': float(0.002),
            'model_version': 'sim-0',
            'entry_reason': 'paper-test',
            'exit_reason': 'simulated'
        }
        append_trade_record(trade)
        print('Appended simulated trade to trades.csv')
    else:
        # if exchange returned an order response, attempt to log fills
        try:
            order_id = res.get('id') if isinstance(res, dict) else str(res)
            filled = 0.0
            avg_price = None
            fees = 0.0
            # CCXT order structure may contain 'trades' or 'filled' fields
            if isinstance(res, dict):
                filled = float(res.get('filled', 0.0))
                avg_price = float(res.get('average')) if res.get('average') else None
                trades = res.get('trades') or []
                for t in trades:
                    fees += float(t.get('fee', {}).get('cost', 0.0) or 0.0)

            entry_ts = datetime.now(timezone.utc).isoformat()
            exit_ts = datetime.now(timezone.utc).isoformat()
            trade = {
                'trade_id': order_id,
                'symbol': symbol,
                'entry_ts': entry_ts,
                'exit_ts': exit_ts,
                'entry_price': entry_price,
                'exit_price': avg_price or entry_price,
                'size': float(filled or qty),
                'side': side,
                'fees': fees,
                'pnl': float(( (avg_price or entry_price) - entry_price) * (filled or qty)),
                'pnl_pct': float(((avg_price or entry_price) / entry_price) - 1.0),
                'model_version': os.getenv('MODEL_VERSION', 'live-unknown'),
                'entry_reason': 'live-order',
                'exit_reason': 'filled'
            }
            append_trade_record(trade)
            print(f'Logged real order {order_id} to trades.csv')
        except Exception as e:
            print('Failed to log live order:', e)


if __name__ == '__main__':
    main()
