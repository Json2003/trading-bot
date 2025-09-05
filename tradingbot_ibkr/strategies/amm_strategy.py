"""
Automated Market Maker (AMM) Strategy

This strategy provides liquidity to the market by placing both bid and ask orders
around the current market price, capturing the bid-ask spread.
"""

from typing import Dict, List, Any
from .base import BaseStrategy, MarketData, Trade
from datetime import datetime
import math


class AMMStrategy(BaseStrategy):
    """
    Automated Market Maker Strategy
    
    Places bid and ask orders around the current price to capture spread.
    Manages inventory risk and adjusts spreads based on market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("AMM Strategy", config)
        
        # AMM-specific configuration
        self.spread_bps = config.get('spread_bps', 20)  # 20 basis points spread
        self.order_size = config.get('order_size', 100.0)  # Base order size
        self.max_inventory = config.get('max_inventory', 1000.0)  # Max inventory per side
        self.inventory_skew_factor = config.get('inventory_skew_factor', 0.1)  # Skew spread based on inventory
        self.min_spread_bps = config.get('min_spread_bps', 5)  # Minimum spread
        self.max_spread_bps = config.get('max_spread_bps', 100)  # Maximum spread
        
        # State tracking
        self.current_bid_orders = []
        self.current_ask_orders = []
        self.inventory = 0.0  # Current inventory (positive = long, negative = short)
        self.last_mid_price = 0.0
        self.volatility_estimate = 0.0
        self.price_history = []
        
    def calculate_optimal_spread(self, data: MarketData) -> float:
        """Calculate optimal spread based on market conditions"""
        # Update volatility estimate
        self.price_history.append(data.close)
        if len(self.price_history) > 20:
            self.price_history.pop(0)
            
        if len(self.price_history) >= 2:
            returns = [(self.price_history[i] / self.price_history[i-1] - 1) 
                      for i in range(1, len(self.price_history))]
            if returns:
                variance = sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)
                self.volatility_estimate = math.sqrt(variance) * math.sqrt(len(self.price_history))
        
        # Base spread
        base_spread_bps = self.spread_bps
        
        # Adjust for volatility (higher vol = wider spreads)
        vol_adjustment = self.volatility_estimate * 1000  # Scale factor
        adjusted_spread = base_spread_bps + vol_adjustment
        
        # Adjust for inventory (more inventory = wider spreads on that side)
        inventory_ratio = abs(self.inventory) / self.max_inventory
        inventory_adjustment = inventory_ratio * self.inventory_skew_factor * 100
        adjusted_spread += inventory_adjustment
        
        # Clamp to min/max bounds
        return max(self.min_spread_bps, min(self.max_spread_bps, adjusted_spread))
        
    def calculate_skew(self) -> float:
        """Calculate price skew based on inventory"""
        if self.max_inventory == 0:
            return 0.0
        inventory_ratio = self.inventory / self.max_inventory
        return inventory_ratio * self.inventory_skew_factor
        
    def on_data(self, data: MarketData) -> List[Dict[str, Any]]:
        """Process market data and generate AMM orders"""
        mid_price = (data.high + data.low) / 2  # Approximate mid price
        self.last_mid_price = mid_price
        
        orders = []
        
        # Cancel existing orders (in real implementation, would check if they need updating)
        self.current_bid_orders.clear()
        self.current_ask_orders.clear()
        
        # Calculate optimal spread and skew
        spread_bps = self.calculate_optimal_spread(data)
        skew = self.calculate_skew()
        
        # Convert basis points to price
        spread_amount = mid_price * (spread_bps / 10000)
        skew_amount = mid_price * skew
        
        # Calculate bid and ask prices
        bid_price = mid_price - spread_amount/2 - skew_amount
        ask_price = mid_price + spread_amount/2 - skew_amount
        
        # Adjust order sizes based on inventory
        inventory_factor = 1.0 - abs(self.inventory) / self.max_inventory
        bid_size = self.order_size * inventory_factor
        ask_size = self.order_size * inventory_factor
        
        # Only place orders if we haven't exceeded inventory limits
        if self.inventory < self.max_inventory and bid_size > 0:
            bid_order = {
                'side': 'buy',
                'quantity': bid_size,
                'price': bid_price,
                'order_type': 'limit',
                'strategy_info': {'type': 'amm_bid', 'spread_bps': spread_bps}
            }
            orders.append(bid_order)
            self.current_bid_orders.append(bid_order)
            
        if self.inventory > -self.max_inventory and ask_size > 0:
            ask_order = {
                'side': 'sell', 
                'quantity': ask_size,
                'price': ask_price,
                'order_type': 'limit',
                'strategy_info': {'type': 'amm_ask', 'spread_bps': spread_bps}
            }
            orders.append(ask_order)
            self.current_ask_orders.append(ask_order)
            
        return orders
        
    def update_position(self, symbol: str, trade: Trade):
        """Update AMM-specific position tracking"""
        super().update_position(symbol, trade)
        
        # Update inventory
        if trade.side == 'buy':
            self.inventory += trade.quantity
        else:
            self.inventory -= trade.quantity
            
    def get_amm_metrics(self) -> Dict[str, Any]:
        """Get AMM-specific performance metrics"""
        base_metrics = self.get_performance_metrics()
        
        # Add AMM-specific metrics
        amm_metrics = {
            'current_inventory': self.inventory,
            'inventory_utilization': abs(self.inventory) / self.max_inventory if self.max_inventory > 0 else 0,
            'spread_efficiency': self.calculate_spread_efficiency(),
            'market_making_trades': len([t for t in self.trades if 'amm' in t.reason.lower()]),
            'avg_spread_captured': self.calculate_avg_spread_captured(),
        }
        
        return {**base_metrics, **amm_metrics}
        
    def calculate_spread_efficiency(self) -> float:
        """Calculate how efficiently we're capturing spreads"""
        if not self.trades:
            return 0.0
            
        # Simple efficiency metric - could be enhanced
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        if not buy_trades or not sell_trades:
            return 0.0
            
        avg_buy_price = sum(t.price for t in buy_trades) / len(buy_trades)
        avg_sell_price = sum(t.price for t in sell_trades) / len(sell_trades)
        
        if avg_buy_price == 0:
            return 0.0
            
        return (avg_sell_price - avg_buy_price) / avg_buy_price * 100
        
    def calculate_avg_spread_captured(self) -> float:
        """Calculate average spread captured per trade pair"""
        efficiency = self.calculate_spread_efficiency()
        return efficiency * 10000  # Convert to basis points
        
    def reset(self):
        """Reset AMM strategy state"""
        super().reset()
        self.current_bid_orders.clear()
        self.current_ask_orders.clear()
        self.inventory = 0.0
        self.price_history.clear()
        self.volatility_estimate = 0.0


def amm_strategy_backtest(df, config: Dict[str, Any] = None):
    """
    Run AMM strategy backtest on OHLCV data
    
    Args:
        df: DataFrame with OHLCV data (requires columns: open, high, low, close, volume)
        config: AMM strategy configuration
        
    Returns:
        Dictionary with backtest results and metrics
    """
    if config is None:
        config = {
            'starting_balance': 10000.0,
            'spread_bps': 20,
            'order_size': 100.0,
            'max_inventory': 500.0,
            'fee_pct': 0.001
        }
        
    strategy = AMMStrategy(config)
    trades = []
    
    # Convert DataFrame rows to MarketData objects and process
    for i in range(len(df)):
        try:
            row = df.iloc[i]
            data = MarketData(
                symbol='BTC/USDT',  # Default symbol
                timestamp=row.name if hasattr(row, 'name') else datetime.now(),
                open=float(row.get('open', row.get('Open', 0))),
                high=float(row.get('high', row.get('High', 0))),
                low=float(row.get('low', row.get('Low', 0))),
                close=float(row.get('close', row.get('Close', 0))),
                volume=float(row.get('volume', row.get('Volume', 0)))
            )
            
            # Get orders from strategy
            orders = strategy.on_data(data)
            
            # Simulate order execution (simplified)
            for order in orders:
                # Assume all limit orders get filled at their price (simplified)
                trade = Trade(
                    trade_id=f"amm_{strategy.trade_count}",
                    symbol=data.symbol,
                    side=order['side'],
                    quantity=order['quantity'],
                    price=order['price'],
                    timestamp=data.timestamp,
                    trade_type='maker',
                    fees=order['quantity'] * order['price'] * config.get('fee_pct', 0.001),
                    reason=f"AMM {order['strategy_info']['type']}"
                )
                
                strategy.trades.append(trade)
                strategy.update_position(data.symbol, trade)
                strategy.trade_count += 1
                strategy.total_fees += trade.fees
                trades.append(trade)
                
        except Exception as e:
            # Skip problematic rows
            continue
            
    # Get final metrics
    metrics = strategy.get_amm_metrics()
    
    return {
        'strategy': 'AMM',
        'trades': len(trades),
        'trade_list': [
            {
                'entry_time': str(t.timestamp),
                'side': t.side,
                'price': t.price,
                'qty': t.quantity,
                'fees': t.fees,
                'reason': t.reason
            } for t in trades
        ],
        'metrics': metrics,
        'final_inventory': strategy.inventory,
        'pnl': metrics['total_pnl'],
        'win_rate_pct': metrics['win_rate']
    }