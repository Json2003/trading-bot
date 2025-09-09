# Enhanced Trading Bot - Setup Guide

## Overview

This trading bot provides comprehensive backtesting, optimization, and live trading capabilities with advanced features including:

- **Grid Search Optimization** with parallel processing and adaptive pruning
- **Multiple ML Models** with hyperparameter optimization (Random Forest, Gradient Boosting, Neural Networks)
- **Professional Backtesting** with comprehensive analytics (Sharpe ratio, profit factor, drawdown analysis)  
- **Enhanced WebSocket Server** with JWT authentication, rate limiting, and real-time monitoring
- **Robust Data Fetching** with retry logic and error handling
- **Async Market Data Crawling** for efficient large-scale data processing

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Quick Start

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd trading-bot
./install.sh  # Linux/macOS
# OR
install.bat   # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r tradingbot_ibkr/requirements.txt
```

### Enhanced Dependencies (Optional)

For full functionality, install these additional packages:

```bash
# For hyperparameter optimization
pip install optuna

# For async data crawling  
pip install aiohttp aiofiles

# For progress bars
pip install tqdm

# For advanced data formats
pip install pyarrow  # Parquet support

# For password hashing (production)
pip install passlib[bcrypt]

# For JWT authentication (production) 
pip install pyjwt[crypto]
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Trading Configuration
EXCHANGE=binance
PAPER=true
API_KEY=your_exchange_api_key
API_SECRET=your_exchange_api_secret

# FRED Data API
FRED_API_KEY=your_fred_api_key

# Server Configuration (Production)
SECRET_KEY=your-super-secret-jwt-key
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Database (Optional)
DATABASE_URL=postgresql://user:pass@localhost/trading_db
```

## Usage Guide

### 1. Grid Search Optimization

#### Basic optimization:
```bash
cd tradingbot_ibkr
python aggressive_optimize.py
```

#### Advanced optimization with parallel processing:
```bash
python aggressive_optimize.py --symbol ETH/USDT --workers 8 --patience 30

python aggressive_optimize_expanded.py --workers 16 --batch-size 100
```

**Key Features:**
- Parallel processing across multiple CPU cores
- Early stopping to prevent overfitting
- Adaptive parameter pruning for efficiency
- Intermediate result saving to prevent data loss
- Comprehensive logging and progress tracking

### 2. Model Training

#### Train multiple ML models with hyperparameter optimization:
```bash
cd tradingbot_ibkr/models

# Basic training
python -c "
from train_batch import train_and_evaluate_models
import pandas as pd

# Load your OHLCV data
df = pd.read_csv('../datafiles/BTC_USDT_bars.csv')
results = train_and_evaluate_models(df, optimize_hyperparams=True, use_optuna=True)
print('Best model:', results)
"
```

**Supported Models:**
- Random Forest Classifier
- Gradient Boosting Classifier  
- Multi-Layer Perceptron (Neural Network)
- Logistic Regression

**Features:**
- Automatic hyperparameter tuning with Optuna (50+ trials)
- Time series cross-validation
- Feature importance analysis
- Model comparison and automatic selection
- Comprehensive performance metrics

### 3. Enhanced Backtesting

```bash
cd tradingbot_ibkr
python backtest_ccxt.py
```

**Professional Analytics Included:**
- Sharpe Ratio
- Sortino Ratio  
- Calmar Ratio
- Maximum Drawdown
- Profit Factor
- Win Rate Analysis
- Risk-Reward Ratios
- Comprehensive trade logging

### 4. WebSocket Server

#### Start enhanced server:
```bash
python server.py
```

#### Test endpoints:
```bash
# Public endpoints
curl http://localhost:8000/health
curl http://localhost:8000/status  
curl http://localhost:8000/metrics

# Authentication
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Protected endpoints (requires JWT token)
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/positions

# WebSocket connection
# Connect to ws://localhost:8000/ws for real-time updates
```

**Server Features:**
- JWT-based authentication with role-based permissions
- Rate limiting (60 requests/minute per IP)
- WebSocket connection management
- Real-time health monitoring
- Comprehensive error handling and logging

### 5. Data Fetching

#### Enhanced FRED data fetching:
```bash
cd tradingbot_ibkr

# Basic usage
python run_fetch_one.py YOUR_FRED_API_KEY

# Advanced usage with date range and format
python run_fetch_one.py YOUR_FRED_API_KEY \
  --series GDPC1 \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --format json \
  --max-retries 5 \
  --verbose
```

**Features:**
- Robust retry logic with exponential backoff
- Data validation and cleaning
- Multiple output formats (CSV, JSON, Parquet)
- Comprehensive error handling
- Progress tracking and logging

### 6. Binance Market Data Crawling

#### Enhanced async crawling:
```bash
# Fast async crawling with progress tracking
python binance_vision_size.py --concurrent 20 --progress

# Specific market data with detailed logging
python binance_vision_size.py \
  --prefix data/spot/ \
  --concurrent 15 \
  --progress \
  --verbose
```

**Features:**
- Async HTTP requests for 10x faster crawling
- Real-time progress tracking with ETA
- Configurable concurrency limits
- Memory-efficient processing
- Comprehensive statistics and error handling

## Architecture

### Project Structure
```
trading-bot/
├── server.py                          # Enhanced WebSocket server
├── tradingbot_ibkr/
│   ├── aggressive_optimize.py         # Basic grid search optimization  
│   ├── aggressive_optimize_expanded.py # Advanced parallel optimization
│   ├── backtest_ccxt.py              # Professional backtesting framework
│   ├── run_fetch_one.py              # Enhanced data fetching
│   ├── models/
│   │   └── train_batch.py            # Multi-model ML training
│   ├── datafiles/                    # Data storage
│   └── model_store/                  # Trained model persistence
├── binance_vision_size.py            # Async market data crawler
├── requirements.txt                   # Core dependencies
└── docs/                             # Documentation
```

### Key Components

1. **Optimization Engine**: Parallel grid search with adaptive pruning
2. **ML Pipeline**: Multi-model training with hyperparameter tuning  
3. **Backtesting Engine**: Professional-grade analytics and risk metrics
4. **Data Layer**: Robust fetching with comprehensive error handling
5. **API Server**: Secure WebSocket server with authentication
6. **Monitoring**: Real-time progress tracking and health monitoring

## Performance Optimizations

### Grid Search
- **Parallel Processing**: Utilizes multiple CPU cores
- **Early Stopping**: Prevents overfitting and saves time
- **Adaptive Pruning**: Focuses on promising parameter ranges
- **Memory Management**: Efficient handling of large parameter spaces

### Machine Learning  
- **Vectorized Operations**: Fast pandas computations
- **Feature Engineering**: Comprehensive technical indicators
- **Hyperparameter Tuning**: Advanced optimization with Optuna
- **Model Persistence**: Efficient model storage and loading

### Data Processing
- **Async I/O**: Non-blocking network operations
- **Connection Pooling**: Efficient HTTP connection reuse
- **Retry Logic**: Robust error handling with exponential backoff
- **Memory Efficiency**: Streaming processing for large datasets

## Monitoring and Logging

### Log Files Generated
- `optimization.log` - Grid search progress and results
- `model_training.log` - ML model training details
- `backtest.log` - Backtesting execution logs
- `server.log` - WebSocket server operations
- `data_fetch.log` - Data fetching operations
- `binance_crawl.log` - Market data crawling logs

### Metrics Collection
- Performance metrics (Sharpe ratio, drawdown, etc.)
- System metrics (CPU, memory usage)
- Network metrics (request rates, error rates)
- Trading metrics (win rate, profit factor)

## Production Deployment

### Security Considerations
1. **Change Default Passwords**: Update default user credentials
2. **Use Strong JWT Secret**: Generate cryptographically secure secret key
3. **Enable HTTPS**: Use reverse proxy (nginx) with SSL certificates
4. **Rate Limiting**: Configure appropriate limits for your use case
5. **Firewall**: Restrict access to necessary ports only

### Scaling
- **Horizontal Scaling**: Deploy multiple server instances behind load balancer
- **Database**: Use PostgreSQL/MySQL for production data storage
- **Caching**: Implement Redis for session and data caching
- **Monitoring**: Use Prometheus/Grafana for metrics collection

### Example Production Setup
```bash
# Use production ASGI server
pip install gunicorn uvicorn[standard]

# Start server with multiple workers
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   pip install -r tradingbot_ibkr/requirements.txt
   ```

2. **Permission Errors**: Ensure proper file permissions
   ```bash
   chmod +x install.sh
   chmod -R 755 tradingbot_ibkr/
   ```

3. **API Key Issues**: Verify API keys in `.env` file
4. **Memory Issues**: Reduce batch sizes or concurrent requests
5. **Network Timeouts**: Increase timeout values in configuration

### Getting Help

- Check log files for detailed error information
- Enable verbose logging with `--verbose` flag
- Review configuration settings in `.env` file
- Ensure all required dependencies are installed

## Contributing

When contributing to this project:

1. Follow existing code style and naming conventions
2. Add comprehensive docstrings to all functions
3. Include error handling and logging
4. Update documentation for new features
5. Add tests for critical functionality

## License

This project is licensed under the terms specified in the LICENSE file.