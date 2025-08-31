"""Example IBKR forex bracket trade (no PDT) using ib_insync.

Connects to IBKR and places a simple bracket market order for a forex pair.
"""
import os
from dotenv import load_dotenv
from ib_insync import IB, Forex, MarketOrder, Order

load_dotenv()
IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.getenv('IBKR_PORT', '7496'))
CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
PAPER = os.getenv('PAPER', 'true').lower() == 'true'
ALLOW_LIVE_RISK = os.getenv('ALLOW_LIVE_RISK', 'false').lower() == 'true'
from money_engine import round_qty

def bracket_order(parent, takeProfitPrice, stopLossPrice, quantity):
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
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID)

    pair = 'EURUSD'
    qty = 1000  # contract size depends on IB conventions
    market_price = 1.10
    tp = market_price + 0.0020
    sl = market_price - 0.0020

    contract = Forex(pair)
    ib.qualifyContracts(contract)

    parent = MarketOrder('BUY', qty)
    orders = bracket_order(parent, tp, sl, qty)

    if PAPER or not ALLOW_LIVE_RISK:
        print('SAFE MODE: not transmitting; orders prepared:')
        for o in orders:
            print(o)
    else:
        qty = round_qty(qty, step=1000, min_qty=1000)
        for o in orders:
            o.totalQuantity = qty
            ib.placeOrder(contract, o)
        print('Orders placed')

    ib.disconnect()

if __name__ == '__main__':
    main()
