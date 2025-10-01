# Trading Readiness Guide

This guide helps you verify that your trading bot is ready for safe operation.

## Quick Start

1. **Run the readiness checker:**
   ```bash
   python check_trading_readiness.py --verbose
   ```

2. **Fix any critical issues identified**

3. **Run quick setup if needed:**
   ```bash
   python quick_setup.py --install-deps
   ```

4. **Need help fixing common issues automatically?**
   ```bash
   python check_trading_readiness.py --fix-issues --verbose
   ```
   This attempts to install missing Python packages, create a `.env` file from the
   template, and generate placeholder market data so you can get to a deployable
   state faster. Review the output carefully and re-run the checker without the
   flag to verify everything looks correct.

## Readiness Checker

The `check_trading_readiness.py` script performs comprehensive validation of your trading setup:

### What It Checks

#### 🐍 Environment Setup
- Python version compatibility (3.11+ recommended)
- Virtual environment usage
- Required dependencies installation

#### ⚙️ Configuration
- `.env` file presence and configuration
- Safety settings (PAPER mode, ALLOW_LIVE_RISK)
- API key configuration

#### 📊 Data Availability
- Market data files presence
- Data directory structure
- Sample data validation

#### 🤖 Model Validation
- Strategy validation reports
- Performance metrics analysis
- Risk assessment results

#### 🛡️ Safety Mechanisms
- Paper trading mode enforcement
- Live risk protection
- Confirmation file checks

#### 💰 Risk Management
- Position sizing modules
- Risk management systems
- Money management validation

#### 🌐 Server Configuration
- API authentication systems
- Rate limiting mechanisms
- Security configurations

### Usage Options

```bash
# Basic readiness check
python check_trading_readiness.py

# Verbose output with detailed information
python check_trading_readiness.py --verbose

# Save report to custom file
python check_trading_readiness.py --output my_report.json

# Show help
python check_trading_readiness.py --help
```

### Understanding Results

#### Status Levels

- **✅ PASS**: Component is properly configured
- **⚠️ WARN**: Component has issues but is not critical
- **❌ FAIL**: Critical issue that must be fixed
- **ℹ️ INFO**: Informational status

#### Overall Status

- **READY**: System is ready for trading (score ≥85%)
- **MOSTLY_READY**: System is mostly ready with minor warnings (score ≥70%)
- **NEEDS_WORK**: System needs significant work (score <70%)
- **NOT_READY**: Critical failures prevent trading

## Common Issues and Fixes

### Missing Dependencies

**Issue**: `Dependency: pandas: Data manipulation library missing`

**Fix**:
```bash
pip install pandas numpy ccxt python-dotenv
```

Or install all requirements:
```bash
pip install -r tradingbot_ibkr/requirements.txt
```

### Missing .env File

**Issue**: `.env file missing but .env.example found`

**Fix**:
```bash
cp tradingbot_ibkr/.env.example .env
# Then edit .env with your settings
```

### Missing Market Data

**Issue**: `Market data file missing`

**Fix**:
- Run data fetching scripts to download market data
- Or create sample data files for testing

### High Validation Drawdown

**Issue**: `Candidate X: High drawdown (XX%)`

**Fix**:
- Review and adjust strategy parameters
- Run stress tests with different market conditions
- Consider reducing position sizes or improving risk management

### No Trades Generated

**Issue**: `Candidate X: No trades generated`

**Fix**:
- Check data quality and timeframe
- Verify strategy entry conditions
- Adjust strategy parameters to be less restrictive

## Safety Checklist

Before enabling live trading, ensure:

- [ ] **PAPER=true** in .env file for initial testing
- [ ] **ALLOW_LIVE_RISK=false** until ready for live trading
- [ ] All critical dependencies installed
- [ ] Strategy validation shows acceptable performance
- [ ] Risk management parameters properly set
- [ ] Stop-loss and take-profit levels configured
- [ ] Position sizing appropriate for account balance
- [ ] API keys configured with appropriate permissions
- [ ] Extensive testing in paper mode completed

## Production Readiness

For production deployment:

1. **Security**:
   - Use strong API keys with minimal required permissions
   - Enable rate limiting and authentication
   - Use HTTPS for all API communications
   - Regularly rotate API keys

2. **Monitoring**:
   - Set up logging and alerting
   - Monitor system performance
   - Track trading metrics
   - Implement health checks

3. **Risk Management**:
   - Set maximum daily/weekly loss limits
   - Implement circuit breakers
   - Use position sizing limits
   - Monitor drawdown levels

4. **Backup and Recovery**:
   - Backup trading logs and configurations
   - Have rollback procedures ready
   - Test disaster recovery scenarios

## Support

If you encounter issues:

1. Check the verbose output for detailed error messages
2. Review fix suggestions in the readiness report
3. Ensure all dependencies are properly installed
4. Verify configuration files are correctly set up
5. Test individual components separately

## Integration with CI/CD

You can integrate the readiness checker with your CI/CD pipeline:

```yaml
# Example GitHub Actions step
- name: Check Trading Readiness
  run: |
    python check_trading_readiness.py --verbose
    if [ $? -eq 1 ]; then
      echo "Critical issues found - deployment blocked"
      exit 1
    fi
```

Exit codes:
- `0`: Ready for trading
- `1`: Not ready (critical failures)
- `2`: Needs work (warnings present)