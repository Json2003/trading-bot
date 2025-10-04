#!/usr/bin/env python3
"""
Test script for trading readiness checker

This script tests the basic functionality of the trading readiness checker
without requiring all dependencies to be installed.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path to import our checker
sys.path.insert(0, str(Path(__file__).parent))

from check_trading_readiness import TradingReadinessChecker, ReadinessReport


def test_basic_functionality():
    """Test basic readiness checker functionality"""
    print("🧪 Testing Trading Readiness Checker...")

    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create basic structure
        tradingbot_dir = temp_path / "tradingbot_ibkr"
        tradingbot_dir.mkdir()

        # Create requirements.txt
        requirements_file = tradingbot_dir / "requirements.txt"
        requirements_file.write_text("pandas==2.2.2\nnumpy==2.2.4\nccxt==3.0.72\n")

        # Create .env.example
        env_example = tradingbot_dir / ".env.example"
        env_example.write_text("PAPER=true\nEXCHANGE=binance\n")

        # Create live trade script with safety checks
        live_script = tradingbot_dir / "live_trade_ccxt.py"
        live_script.write_text("""
import os
PAPER = os.getenv('PAPER', 'true')
ALLOW_LIVE_RISK = os.getenv('ALLOW_LIVE_RISK', 'false')
""")

        # Create money engine
        money_engine = tradingbot_dir / "money_engine.py"
        money_engine.write_text("def choose_position_size(): pass")

        # Initialize checker
        checker = TradingReadinessChecker(temp_path, verbose=False)

        # Run subset of checks that don't require dependencies
        checker.check_python_environment()
        checker.check_configuration_files()
        checker.check_safety_mechanisms()
        checker.check_risk_management()

        # Generate report
        report = checker.run_all_checks()

        # Basic assertions
        assert isinstance(report, ReadinessReport)
        assert report.timestamp is not None
        assert report.overall_status in ["READY", "MOSTLY_READY", "NEEDS_WORK", "NOT_READY"]
        assert 0 <= report.score <= 100
        assert len(report.checks) > 0
        assert len(report.recommendations) > 0

        # Check that we have some basic checks
        check_names = [check.name for check in report.checks]
        assert "Python Version" in check_names
        assert "Requirements File" in check_names

        print("✅ Basic functionality test passed")
        print(f"   - Found {len(report.checks)} checks")
        print(f"   - Overall status: {report.overall_status}")
        print(f"   - Readiness score: {report.score:.1f}/100")

        return True


def test_json_serialization():
    """Test that reports can be serialized to JSON"""
    print("🧪 Testing JSON serialization...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tradingbot_dir = temp_path / "tradingbot_ibkr"
        tradingbot_dir.mkdir()

        checker = TradingReadinessChecker(temp_path, verbose=False)
        checker.check_python_environment()

        report = checker.run_all_checks()

        # Test JSON serialization
        json_str = json.dumps(
            {
                "timestamp": report.timestamp,
                "overall_status": report.overall_status,
                "score": report.score,
                "summary": report.summary,
                "recommendations": report.recommendations,
            },
            indent=2,
        )

        assert len(json_str) > 0

        # Test deserialization
        data = json.loads(json_str)
        assert data["overall_status"] == report.overall_status
        assert data["score"] == report.score

        print("✅ JSON serialization test passed")
        return True


def main():
    """Run all tests"""
    print("🚀 Running Trading Readiness Checker Tests")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_json_serialization()

        print("\n🎉 All tests passed!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r tradingbot_ibkr/requirements.txt")
        print("2. Run full readiness check: python check_trading_readiness.py --verbose")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
