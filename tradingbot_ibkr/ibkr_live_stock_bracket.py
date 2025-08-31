"""Example IBKR live stock bracket order using ib_insync.

This script connects to TWS/IB Gateway and demonstrates placing a bracket
order (parent + take-profit + stop-loss). WARNING: live trading risks real money.
Set PAPER=false in .env if you want to run real orders.
"""
import os
from dotenv import load_dotenv
from ib_insync import IB, Stock, MarketOrder, Order

load_dotenv()
IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.getenv('IBKR_PORT', '7496'))
CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
PAPER = os.getenv('PAPER', 'true').lower() == 'true'
ALLOW_LIVE_RISK = os.getenv('ALLOW_LIVE_RISK', 'false').lower() == 'true'
from money_engine import round_qty

def bracket_order(parent, takeProfitPrice, stopLossPrice, quantity):
    # parent is a MarketOrder
    parent.orderType = 'MKT'
    parent.totalQuantity = quantity
    parent.transmit = False

    tp = Order(orderType='LMT', action='SELL', lmtPrice=takeProfitPrice, totalQuantity=quantity)
    tp.parentId = parent.orderId
    tp.transmit = False

    sl = Order(orderType='STP', action='SELL', auxPrice=stopLossPrice, totalQuantity=quantity)
    sl.parentId = parent.orderId
    sl.transmit = True

    return [parent, tp, sl]

def main():
    ib = IB()
    print('Connecting to IBKR', IBKR_HOST, IBKR_PORT)
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID)

    symbol = 'AAPL'
    exchange = 'SMART'
    qty = 1
    market_price = 170.0
    tp = market_price * 1.02
    sl = market_price * 0.98

    contract = Stock(symbol, exchange, 'USD')
    ib.qualifyContracts(contract)

    parent = MarketOrder('BUY', qty)
    orders = bracket_order(parent, tp, sl, qty)

    if PAPER or not ALLOW_LIVE_RISK:
        print('SAFE MODE: not transmitting; orders prepared:')
        for o in orders:
            print(o)
    else:
        # ensure qty respects minimums
        qty = round_qty(qty, step=1, min_qty=1)
        for o in orders:
            o.totalQuantity = qty
            ib.placeOrder(contract, o)
        print('Orders placed')

    ib.disconnect()

if __name__ == '__main__':
    main()
