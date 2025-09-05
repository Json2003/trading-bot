"""
Multi-pool Staking Strategy

This strategy manages staking across multiple pools/assets to optimize yield
while managing risk through diversification and rebalancing.
"""

from typing import Dict, List, Any, Tuple
from .base import BaseStrategy, MarketData, Trade
from datetime import datetime, timedelta
import math


class Pool:
    """Represents a staking pool"""
    def __init__(self, name: str, apy: float, risk_score: float, min_stake: float = 0.0):
        self.name = name
        self.apy = apy  # Annual percentage yield
        self.risk_score = risk_score  # 1-10, 1 being safest
        self.min_stake = min_stake
        self.current_stake = 0.0
        self.rewards_earned = 0.0
        self.last_reward_time = datetime.now()
        
    def calculate_rewards(self, current_time: datetime) -> float:
        """Calculate rewards earned since last update"""
        if self.current_stake == 0:
            return 0.0
            
        time_diff = (current_time - self.last_reward_time).total_seconds()
        days_elapsed = time_diff / (24 * 3600)
        
        # Simple compound interest calculation
        daily_rate = (1 + self.apy / 100) ** (1/365) - 1
        rewards = self.current_stake * daily_rate * days_elapsed
        
        self.rewards_earned += rewards
        self.last_reward_time = current_time
        return rewards
        
    def stake(self, amount: float) -> bool:
        """Stake additional amount"""
        if amount >= self.min_stake:
            self.current_stake += amount
            return True
        return False
        
    def unstake(self, amount: float) -> float:
        """Unstake amount, returns actual amount unstaked"""
        unstaked = min(amount, self.current_stake)
        self.current_stake -= unstaked
        return unstaked


class MultipoolStakingStrategy(BaseStrategy):
    """
    Multi-pool Staking Strategy
    
    Allocates capital across multiple staking pools based on:
    - Risk-adjusted returns
    - Diversification requirements
    - Rebalancing triggers
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Multipool Staking Strategy", config)
        
        # Configuration
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)  # 10% deviation trigger
        self.max_pool_allocation = config.get('max_pool_allocation', 0.4)  # Max 40% per pool
        self.min_pool_allocation = config.get('min_pool_allocation', 0.05)  # Min 5% per pool
        self.rebalance_frequency_days = config.get('rebalance_frequency_days', 7)  # Weekly rebalancing
        self.risk_tolerance = config.get('risk_tolerance', 5)  # 1-10 scale
        
        # Initialize pools
        self.pools = self._initialize_pools(config.get('pools', []))
        self.target_allocations = {}  # Will be calculated based on optimization
        self.last_rebalance = datetime.now()
        self.total_rewards_earned = 0.0
        
    def _initialize_pools(self, pool_configs: List[Dict[str, Any]]) -> Dict[str, Pool]:
        """Initialize staking pools from configuration"""
        pools = {}
        
        # Default pools if none provided
        if not pool_configs:
            pool_configs = [
                {'name': 'BTC_Staking', 'apy': 4.5, 'risk_score': 3, 'min_stake': 0.001},
                {'name': 'ETH_Staking', 'apy': 6.0, 'risk_score': 4, 'min_stake': 0.01},
                {'name': 'USDC_Lending', 'apy': 8.0, 'risk_score': 2, 'min_stake': 10.0},
                {'name': 'DeFi_Pool', 'apy': 15.0, 'risk_score': 7, 'min_stake': 50.0},
                {'name': 'Stable_Farm', 'apy': 12.0, 'risk_score': 5, 'min_stake': 20.0},
            ]
            
        for config in pool_configs:
            pool = Pool(
                name=config['name'],
                apy=config['apy'],
                risk_score=config['risk_score'],
                min_stake=config.get('min_stake', 0.0)
            )
            pools[pool.name] = pool
            
        return pools
        
    def calculate_target_allocations(self) -> Dict[str, float]:
        """Calculate optimal portfolio allocations using risk-adjusted returns"""
        if not self.pools:
            return {}
            
        # Calculate risk-adjusted scores for each pool
        pool_scores = {}
        for name, pool in self.pools.items():
            # Risk adjustment factor (higher risk tolerance = less penalty for risk)
            risk_penalty = (pool.risk_score / 10) * (10 - self.risk_tolerance) / 10
            risk_adjusted_apy = pool.apy * (1 - risk_penalty)
            pool_scores[name] = max(0, risk_adjusted_apy)
            
        if not pool_scores or sum(pool_scores.values()) == 0:
            return {}
            
        # Calculate base allocations proportional to scores
        total_score = sum(pool_scores.values())
        base_allocations = {name: score / total_score for name, score in pool_scores.items()}
        
        # Apply allocation constraints
        allocations = {}
        remaining = 1.0
        
        for name, alloc in base_allocations.items():
            # Clamp to min/max bounds
            constrained_alloc = max(self.min_pool_allocation, 
                                  min(self.max_pool_allocation, alloc))
            allocations[name] = constrained_alloc
            remaining -= constrained_alloc
            
        # Redistribute any remaining allocation
        if remaining > 0 and allocations:
            adjustment_per_pool = remaining / len(allocations)
            for name in allocations:
                allocations[name] += adjustment_per_pool
                
        # Normalize to ensure sum equals 1.0
        total = sum(allocations.values())
        if total > 0:
            allocations = {name: alloc / total for name, alloc in allocations.items()}
            
        return allocations
        
    def needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing"""
        current_time = datetime.now()
        
        # Check time-based rebalancing
        if (current_time - self.last_rebalance).days >= self.rebalance_frequency_days:
            return True
            
        # Check threshold-based rebalancing
        current_allocations = self.get_current_allocations()
        target_allocations = self.calculate_target_allocations()
        
        for pool_name in current_allocations:
            current = current_allocations.get(pool_name, 0)
            target = target_allocations.get(pool_name, 0)
            if abs(current - target) > self.rebalance_threshold:
                return True
                
        return False
        
    def get_current_allocations(self) -> Dict[str, float]:
        """Get current allocation percentages"""
        total_value = sum(pool.current_stake for pool in self.pools.values())
        if total_value == 0:
            return {}
            
        return {name: pool.current_stake / total_value 
                for name, pool in self.pools.items()}
                
    def rebalance_portfolio(self, total_capital: float) -> List[Dict[str, Any]]:
        """Rebalance portfolio to target allocations"""
        target_allocations = self.calculate_target_allocations()
        current_allocations = self.get_current_allocations()
        
        rebalance_orders = []
        
        for pool_name, target_pct in target_allocations.items():
            if pool_name not in self.pools:
                continue
                
            pool = self.pools[pool_name]
            target_amount = total_capital * target_pct
            current_amount = pool.current_stake
            difference = target_amount - current_amount
            
            if abs(difference) > pool.min_stake:  # Only rebalance if difference is significant
                if difference > 0:
                    # Need to stake more
                    order = {
                        'action': 'stake',
                        'pool': pool_name,
                        'amount': difference,
                        'reason': f'Rebalance to {target_pct:.1%}'
                    }
                else:
                    # Need to unstake
                    order = {
                        'action': 'unstake', 
                        'pool': pool_name,
                        'amount': abs(difference),
                        'reason': f'Rebalance to {target_pct:.1%}'
                    }
                rebalance_orders.append(order)
                
        self.last_rebalance = datetime.now()
        return rebalance_orders
        
    def on_data(self, data: MarketData) -> List[Dict[str, Any]]:
        """Process market data and generate staking/unstaking orders"""
        orders = []
        
        # Update rewards for all pools
        current_time = data.timestamp
        total_new_rewards = 0.0
        
        for pool in self.pools.values():
            new_rewards = pool.calculate_rewards(current_time)
            total_new_rewards += new_rewards
            
        self.total_rewards_earned += total_new_rewards
        self.balance += total_new_rewards  # Add rewards to available balance
        
        # Check if rebalancing is needed
        if self.needs_rebalancing():
            total_capital = self.balance + sum(pool.current_stake for pool in self.pools.values())
            rebalance_orders = self.rebalance_portfolio(total_capital)
            
            # Execute rebalancing orders
            for order in rebalance_orders:
                pool = self.pools[order['pool']]
                
                if order['action'] == 'stake' and self.balance >= order['amount']:
                    if pool.stake(order['amount']):
                        self.balance -= order['amount']
                        
                        # Create a trade record
                        trade = Trade(
                            trade_id=f"stake_{self.trade_count}",
                            symbol=f"{order['pool']}/STAKE", 
                            side='buy',  # Staking is like buying
                            quantity=order['amount'],
                            price=1.0,  # Stake at par value
                            timestamp=current_time,
                            trade_type='stake',
                            reason=order['reason']
                        )
                        self.trades.append(trade)
                        self.trade_count += 1
                        
                elif order['action'] == 'unstake':
                    unstaked_amount = pool.unstake(order['amount'])
                    if unstaked_amount > 0:
                        self.balance += unstaked_amount
                        
                        # Create a trade record  
                        trade = Trade(
                            trade_id=f"unstake_{self.trade_count}",
                            symbol=f"{order['pool']}/STAKE",
                            side='sell',  # Unstaking is like selling
                            quantity=unstaked_amount,
                            price=1.0,  # Unstake at par value
                            timestamp=current_time,
                            trade_type='unstake', 
                            reason=order['reason']
                        )
                        self.trades.append(trade)
                        self.trade_count += 1
                        
        return orders
        
    def get_staking_metrics(self) -> Dict[str, Any]:
        """Get staking-specific performance metrics"""
        base_metrics = self.get_performance_metrics()
        
        total_staked = sum(pool.current_stake for pool in self.pools.values())
        total_portfolio = self.balance + total_staked
        
        # Calculate APY metrics
        pool_apys = {name: pool.apy for name, pool in self.pools.items() if pool.current_stake > 0}
        current_allocations = self.get_current_allocations()
        
        weighted_apy = sum(pool_apys.get(name, 0) * alloc 
                          for name, alloc in current_allocations.items())
        
        staking_metrics = {
            'total_staked': total_staked,
            'total_rewards_earned': self.total_rewards_earned,
            'staking_utilization': total_staked / total_portfolio if total_portfolio > 0 else 0,
            'weighted_portfolio_apy': weighted_apy,
            'number_of_pools': len([p for p in self.pools.values() if p.current_stake > 0]),
            'pool_allocations': current_allocations,
            'pool_details': {
                name: {
                    'stake': pool.current_stake,
                    'rewards': pool.rewards_earned,
                    'apy': pool.apy,
                    'risk_score': pool.risk_score
                } for name, pool in self.pools.items()
            }
        }
        
        return {**base_metrics, **staking_metrics}
        

def multipool_staking_backtest(df, config: Dict[str, Any] = None):
    """
    Run multi-pool staking strategy backtest
    
    Args:
        df: DataFrame with OHLCV data (used for timestamps)
        config: Staking strategy configuration
        
    Returns:
        Dictionary with backtest results and metrics
    """
    if config is None:
        config = {
            'starting_balance': 10000.0,
            'rebalance_threshold': 0.1,
            'max_pool_allocation': 0.4,
            'rebalance_frequency_days': 7,
            'risk_tolerance': 5
        }
        
    strategy = MultipoolStakingStrategy(config)
    
    # Initially allocate all capital to pools
    initial_allocations = strategy.calculate_target_allocations()
    for pool_name, target_pct in initial_allocations.items():
        if pool_name in strategy.pools:
            amount = strategy.balance * target_pct
            pool = strategy.pools[pool_name]
            if pool.stake(amount):
                strategy.balance -= amount
    
    # Process each time period
    for i in range(len(df)):
        try:
            row = df.iloc[i]
            data = MarketData(
                symbol='STAKING',
                timestamp=row.name if hasattr(row, 'name') else datetime.now() + timedelta(days=i),
                open=1.0,  # Staking doesn't have traditional OHLCV
                high=1.0,
                low=1.0,
                close=1.0,
                volume=0.0
            )
            
            # Process the data (updates rewards and rebalances if needed)
            strategy.on_data(data)
            
        except Exception as e:
            continue
            
    # Get final metrics
    metrics = strategy.get_staking_metrics()
    
    return {
        'strategy': 'Multipool Staking',
        'trades': len(strategy.trades),
        'trade_list': [
            {
                'entry_time': str(t.timestamp),
                'side': t.side,
                'symbol': t.symbol,
                'qty': t.quantity,
                'price': t.price,
                'reason': t.reason
            } for t in strategy.trades
        ],
        'metrics': metrics,
        'total_rewards': strategy.total_rewards_earned,
        'final_balance': strategy.balance,
        'total_staked': sum(pool.current_stake for pool in strategy.pools.values()),
        'pnl': metrics['total_pnl'] + strategy.total_rewards_earned,  # Include staking rewards in PnL
        'apy_achieved': metrics['weighted_portfolio_apy']
    }