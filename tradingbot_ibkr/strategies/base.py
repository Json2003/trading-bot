"""
Base Strategy Classes

Provides common functionality for all trading strategies.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass 
class Trade:
    """Represents a single trade"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell' 
    quantity: float
    price: float
    timestamp: datetime
    trade_type: str  # 'entry', 'exit', 'maker', 'taker'
    fees: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """Represents a current position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    
    @property 
    def market_value(self) -> float:
        return self.quantity * self.current_price
        
    @property
    def realized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity


@dataclass
class MarketData:
    """Market data for a single bar/tick"""
    symbol: str
    timestamp: datetime
    open: float
    high: float  
    low: float
    close: float
    volume: float
    
    
class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.balance = self.config.get('starting_balance', 10000.0)
        self.total_fees = 0.0
        self.trade_count = 0
        
    def on_data(self, data: MarketData) -> List[Dict[str, Any]]:
        """
        Process new market data and return list of orders to place.
        
        Args:
            data: Market data for current bar/tick
            
        Returns:
            List of order dictionaries with keys: side, quantity, price, order_type
        """
        raise NotImplementedError("Subclasses must implement on_data()")
        
    def update_position(self, symbol: str, trade: Trade):
        """Update position based on executed trade"""
        if symbol not in self.positions:
            if trade.side == 'buy':
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity, 
                    entry_price=trade.price,
                    current_price=trade.price,
                    entry_time=trade.timestamp
                )
            return
            
        pos = self.positions[symbol]
        
        if trade.side == 'buy':
            # Add to position
            total_cost = pos.quantity * pos.entry_price + trade.quantity * trade.price
            pos.quantity += trade.quantity
            pos.entry_price = total_cost / pos.quantity if pos.quantity > 0 else trade.price
        else:
            # Reduce position
            pos.quantity -= trade.quantity
            if pos.quantity <= 0:
                del self.positions[symbol]
                
    def calculate_position_size(self, symbol: str, price: float, risk_pct: float = 0.02) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = self.balance * risk_pct
        return risk_amount / price
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.trades:
            return {'total_pnl': 0.0, 'win_rate': 0.0, 'trade_count': 0}
            
        total_pnl = sum(t.quantity * t.price * (-1 if t.side == 'buy' else 1) for t in self.trades) - self.total_fees
        winning_trades = 0
        
        # Simple PnL calculation - could be enhanced with proper entry/exit pairing
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        paired_trades = min(len(buy_trades), len(sell_trades))
        for i in range(paired_trades):
            pnl = (sell_trades[i].price - buy_trades[i].price) * min(buy_trades[i].quantity, sell_trades[i].quantity)
            if pnl > 0:
                winning_trades += 1
                
        win_rate = (winning_trades / paired_trades * 100) if paired_trades > 0 else 0.0
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate, 
            'trade_count': len(self.trades),
            'paired_trades': paired_trades,
            'total_fees': self.total_fees
        }
        
    def reset(self):
        """Reset strategy state"""
        self.positions.clear()
        self.trades.clear()
        self.balance = self.config.get('starting_balance', 10000.0)
        self.total_fees = 0.0
        self.trade_count = 0