# Advanced Trading Strategies

This module implements new trading strategies for the trading bot, including:

## 1. Automated Market Maker (AMM)
**Purpose**: Provides liquidity by placing bid and ask orders around current price  
**Expected Returns**: 5-15% annually  
**Risk Level**: Medium

**Features**:
- Dynamic spread adjustment based on volatility
- Inventory risk management 
- Spread skewing based on position
- Real-time bid/ask order placement

**Configuration**: See `strategies_config.json` under `strategies.amm`

## 2. Multi-Pool Staking System  
**Purpose**: Diversified staking across multiple pools with automatic rebalancing  
**Expected Returns**: 6-20% annually  
**Risk Level**: Low-Medium

**Features**:
- Risk-adjusted pool allocation
- Automatic rebalancing triggers
- Yield optimization algorithms
- Multiple pool support (BTC, ETH, USDC, DeFi, Stablecoins)

**Configuration**: See `strategies_config.json` under `strategies.multipool_staking`

## 3. Intelligent Trading Strategies

### Momentum Strategy
**Purpose**: Trades in direction of strong price trends  
**Expected Returns**: 10-30% annually (high volatility)  
**Risk Level**: High

**Features**:
- Multiple momentum indicators (price, volume, RSI)
- Dynamic position sizing
- Stop loss and take profit management
- Trend confirmation filters

### Mean Reversion Strategy  
**Purpose**: Trades oversold/overbought conditions expecting price to revert  
**Expected Returns**: 8-25% annually  
**Risk Level**: Medium-High

**Features**:
- Bollinger Bands for overbought/oversold detection
- RSI confirmation
- Maximum holding period limits
- Mean reversion targeting

### Arbitrage Strategy
**Purpose**: Captures price differences between exchanges for low-risk profits  
**Expected Returns**: 3-12% annually  
**Risk Level**: Very Low

**Features**:
- Multi-exchange price monitoring
- Simultaneous buy/sell execution
- Transaction cost optimization
- Risk-free profit capture

## Usage

### Running Individual Strategy Backtests

```python
from strategies.amm_strategy import amm_strategy_backtest
from strategies.multipool_strategy import multipool_staking_backtest
from strategies.intelligent_strategies import momentum_strategy_backtest

# Run AMM backtest
config = {'starting_balance': 10000, 'spread_bps': 20}
result = amm_strategy_backtest(ohlcv_data, config)

# Run staking backtest  
result = multipool_staking_backtest(ohlcv_data)

# Run momentum backtest
result = momentum_strategy_backtest(ohlcv_data)
```

### Running All Strategies

```python
from advanced_strategies import run_all_strategies_backtest

# Test all strategies
results = run_all_strategies_backtest(ohlcv_data)

# Get performance comparison
comparison = results['comparison']
print(f"Best performer: {comparison['summary']['best_performing']}")
```

### Command Line Usage

```bash
# Test all strategies with sample data
python advanced_strategies.py

# Run integrated backtest
python backtest_ccxt.py

# Run strategy validation tests
python test_strategies.py
```

## Portfolio Allocation Presets

The configuration includes three portfolio allocation presets:

### Conservative
- 50% Multipool Staking
- 30% Arbitrage  
- 20% AMM

### Balanced
- 30% Multipool Staking
- 25% AMM
- 20% Mean Reversion
- 15% Arbitrage
- 10% Momentum

### Aggressive  
- 35% Momentum
- 25% Mean Reversion
- 20% AMM
- 15% Multipool Staking
- 5% Arbitrage

## Configuration

All strategy parameters can be customized in `strategies_config.json`. Key sections:

- `strategies.<strategy_name>.config`: Strategy-specific parameters
- `portfolio_allocation`: Preset allocation percentages
- `risk_management`: Global risk limits
- `backtest_settings`: Default backtest parameters

## Files Structure

```
tradingbot_ibkr/
├── strategies/
│   ├── __init__.py              # Strategy module exports
│   ├── base.py                  # Base strategy classes
│   ├── amm_strategy.py          # Automated Market Maker
│   ├── multipool_strategy.py    # Multi-pool staking
│   └── intelligent_strategies.py # Momentum, Mean Reversion, Arbitrage
├── advanced_strategies.py       # Integration and testing
├── strategies_config.json       # Configuration file
├── test_strategies.py          # Validation tests
└── README_strategies.md        # This file
```

## Integration with Existing System

The new strategies integrate seamlessly with the existing trading bot:

1. **Backtest Framework**: Uses same OHLCV data format as existing `aggressive_strategy_backtest()`
2. **Configuration**: Follows same `.env` and config patterns  
3. **Trade Recording**: Compatible with existing trade logging in `data/store.py`
4. **Performance Metrics**: Extends existing metrics with strategy-specific KPIs

## Development Notes

- All strategies inherit from `BaseStrategy` for consistent interface
- Modular design allows easy addition of new strategies
- Comprehensive error handling and fallbacks
- Extensive configuration options for customization
- Built-in performance comparison and recommendations

## Future Enhancements

1. **Real-time Data Integration**: Connect to live market data feeds
2. **Machine Learning**: Integrate with existing ML models in `models/`
3. **Risk Management**: Enhanced drawdown protection and position sizing
4. **Multi-asset Support**: Extend beyond crypto to forex, stocks, commodities
5. **Strategy Combinations**: Implement strategy mixing and ensemble methods