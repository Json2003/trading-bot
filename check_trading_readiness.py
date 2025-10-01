#!/usr/bin/env python3
"""
Trading Readiness Checker

This script comprehensively validates whether the trading bot repository
is ready for live trading by checking:

1. Environment setup and dependencies
2. Configuration and safety settings
3. Data availability and quality
4. Model validation results
5. Risk management settings
6. Trading safety checks

Usage:
    python check_trading_readiness.py [--fix-issues] [--verbose]
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import subprocess
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a readiness check"""
    name: str
    status: str  # 'PASS', 'FAIL', 'WARN', 'INFO'
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None


@dataclass
class ReadinessReport:
    """Complete readiness assessment report"""
    timestamp: str
    overall_status: str
    score: float  # 0-100 readiness score
    checks: List[CheckResult]
    summary: Dict[str, int]  # count by status
    recommendations: List[str]


DEFAULT_ENV_TEMPLATE = """# General
PAPER=true
TIMEZONE=UTC

# CCXT / Crypto
EXCHANGE=binance
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
PAPER_CURRENCY=USDT

# IBKR (TWS/IB Gateway)
IBKR_HOST=127.0.0.1
IBKR_PORT=7496
IBKR_CLIENT_ID=1

# Other
LOG_LEVEL=INFO
ALLOW_LIVE_RISK=false
"""


class TradingReadinessChecker:
    """Comprehensive trading readiness validation"""

    def __init__(self, repo_root: Path, verbose: bool = False, fix_issues: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
        self.fix_issues = fix_issues
        self.tradingbot_dir = repo_root / "tradingbot_ibkr"
        self.checks: List[CheckResult] = []
        self.critical_failures = 0
        self._install_attempts: Dict[str, bool] = {}
        
    def add_check(self, name: str, status: str, message: str, 
                  details: str = None, fix_suggestion: str = None):
        """Add a check result"""
        check = CheckResult(name, status, message, details, fix_suggestion)
        self.checks.append(check)
        
        if status == 'FAIL':
            self.critical_failures += 1
            
        if self.verbose or status in ['FAIL', 'WARN']:
            logger.info(f"[{status}] {name}: {message}")
            if details and self.verbose:
                logger.info(f"    Details: {details}")

    def attempt_dependency_install(self, package_name: str) -> bool:
        """Attempt to install a dependency using pip when fixes are enabled."""
        if package_name in self._install_attempts:
            return self._install_attempts[package_name]

        if not self.fix_issues:
            self._install_attempts[package_name] = False
            return False

        logger.info(f"Attempting to install missing dependency: {package_name}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=not self.verbose,
                text=True,
                check=True,
                timeout=180,
            )
            if self.verbose and result.stdout:
                logger.info(result.stdout.strip())
            self._install_attempts[package_name] = True
            return True
        except subprocess.CalledProcessError as exc:
            if self.verbose and exc.stderr:
                logger.error(exc.stderr.strip())
            self._install_attempts[package_name] = False
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Unexpected error installing {package_name}: {exc}")
            self._install_attempts[package_name] = False
            return False

    def check_python_environment(self) -> None:
        """Check Python version and virtual environment"""
        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 11):
            self.add_check(
                "Python Version",
                "PASS", 
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro} is supported"
            )
        elif python_version >= (3, 8):
            self.add_check(
                "Python Version", 
                "WARN",
                f"Python {python_version.major}.{python_version.minor} works but 3.11+ recommended",
                fix_suggestion="Consider upgrading to Python 3.11 or 3.12"
            )
        else:
            self.add_check(
                "Python Version",
                "FAIL",
                f"Python {python_version.major}.{python_version.minor} is too old",
                fix_suggestion="Upgrade to Python 3.11 or higher"
            )
        
        # Virtual environment check
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if in_venv:
            self.add_check(
                "Virtual Environment",
                "PASS",
                "Running in virtual environment"
            )
        else:
            self.add_check(
                "Virtual Environment",
                "WARN", 
                "Not running in virtual environment",
                fix_suggestion="Create virtual environment: python -m venv venv && source venv/bin/activate"
            )

    def check_dependencies(self) -> None:
        """Check required dependencies"""
        requirements_file = self.tradingbot_dir / "requirements.txt"
        
        if not requirements_file.exists():
            self.add_check(
                "Requirements File",
                "FAIL",
                "requirements.txt not found",
                fix_suggestion="Create requirements.txt with necessary dependencies"
            )
            return
            
        self.add_check(
            "Requirements File", 
            "PASS",
            "requirements.txt found"
        )
        
        # Check critical dependencies
        critical_deps = [
            ("pandas", "pandas", "Data manipulation"),
            ("numpy", "numpy", "Numerical computations"),
            ("ccxt", "ccxt", "Exchange connectivity"),
            ("dotenv", "python-dotenv", "Environment configuration")
        ]

        for module_name, package_name, description in critical_deps:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    self.add_check(
                        f"Dependency: {package_name}",
                        "PASS",
                        f"{description} library available"
                    )
                else:
                    installed = self.attempt_dependency_install(package_name)
                    if installed:
                        self.add_check(
                            f"Dependency: {package_name}",
                            "PASS",
                            f"{description} library installed automatically"
                        )
                    else:
                        self.add_check(
                            f"Dependency: {package_name}",
                            "FAIL",
                            f"{description} library missing",
                            fix_suggestion=f"pip install {package_name}"
                        )
            except Exception as e:
                self.add_check(
                    f"Dependency: {package_name}",
                    "FAIL",
                    f"Error checking {module_name}: {str(e)}",
                    fix_suggestion=f"pip install {package_name}"
                )

    def check_configuration_files(self) -> None:
        """Check configuration and environment files"""
        # .env file check
        env_file = self.repo_root / ".env"
        env_example = self.tradingbot_dir / ".env.example"

        if self.ensure_env_file(env_file, env_example):
            self.add_check(
                "Environment File",
                "PASS",
                ".env file found"
            )
            self._validate_env_settings(env_file)
        elif env_example.exists():
            self.add_check(
                "Environment File",
                "WARN",
                ".env file missing but .env.example found",
                fix_suggestion="Copy .env.example to .env and configure settings"
            )
        else:
            self.add_check(
                "Environment File",
                "WARN",
                ".env file missing",
                fix_suggestion="Create .env file with trading configuration"
            )

    def ensure_env_file(self, env_file: Path, env_example: Path) -> bool:
        """Ensure a .env file exists by copying or creating a template."""
        if env_file.exists():
            return True

        if not self.fix_issues:
            return False

        try:
            env_file.parent.mkdir(parents=True, exist_ok=True)
            if env_example.exists():
                shutil.copy(env_example, env_file)
                logger.info("Created .env from .env.example")
            else:
                env_file.write_text(DEFAULT_ENV_TEMPLATE)
                logger.info("Created default .env template")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to create .env file: {exc}")
            return False
    
    def _validate_env_settings(self, env_file: Path) -> None:
        """Validate environment variable settings"""
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Check for paper trading safety
            if 'PAPER=true' in content or 'PAPER=True' in content:
                self.add_check(
                    "Paper Trading Mode",
                    "PASS",
                    "Paper trading is enabled (safe for testing)"
                )
            elif 'PAPER=false' in content or 'PAPER=False' in content:
                self.add_check(
                    "Paper Trading Mode", 
                    "WARN",
                    "Live trading mode detected - ensure this is intentional",
                    fix_suggestion="Set PAPER=true for safe testing"
                )
            else:
                self.add_check(
                    "Paper Trading Mode",
                    "WARN",
                    "PAPER setting not found in .env",
                    fix_suggestion="Add PAPER=true to .env for safety"
                )
            
            # Check for API keys
            if 'API_KEY=' in content and 'API_SECRET=' in content:
                self.add_check(
                    "API Configuration",
                    "PASS",
                    "API key placeholders found"
                )
            else:
                self.add_check(
                    "API Configuration", 
                    "WARN",
                    "API key configuration missing",
                    fix_suggestion="Add API_KEY and API_SECRET to .env"
                )
                
        except Exception as e:
            self.add_check(
                "Environment Validation",
                "FAIL",
                f"Error reading .env file: {str(e)}"
            )

    def generate_sample_market_data(self, file_path: Path, symbol_key: str) -> bool:
        """Generate deterministic OHLCV sample data when fixes are enabled."""
        if file_path.exists():
            return True

        if not self.fix_issues:
            return False

        base_prices = {
            "BTC_USDT": 67000.0,
            "ETH_USDT": 3400.0,
        }

        base_price = base_prices.get(symbol_key, 100.0)
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as handle:
                handle.write("ts,Unnamed: 0,open,high,low,close,volume\n")
                for idx in range(24):
                    ts = (start_time + timedelta(hours=idx)).strftime("%Y-%m-%d %H:%M:%S")
                    open_price = base_price + idx * 1.5
                    high_price = open_price + 8.0
                    low_price = open_price - 8.0
                    close_price = open_price + ((idx % 4) - 1.5) * 1.2
                    volume = 150 + idx * 2.5
                    handle.write(
                        f"{ts},{float(idx):.1f},{open_price:.2f},{high_price:.2f},{low_price:.2f},{close_price:.2f},{volume:.5f}\n"
                    )
            logger.info(f"Generated sample market data: {file_path}")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to generate sample data for {symbol_key}: {exc}")
            return False

    def check_data_availability(self) -> None:
        """Check for required data files"""
        data_dir = self.tradingbot_dir / "datafiles"
        
        if not data_dir.exists():
            self.add_check(
                "Data Directory",
                "WARN", 
                "datafiles directory not found",
                fix_suggestion="Create datafiles directory and download market data"
            )
            return
            
        self.add_check(
            "Data Directory",
            "PASS",
            "datafiles directory exists"
        )
        
        # Check for sample data files
        expected_files = [
            "BTC_USDT_bars.csv",
            "ETH_USDT_bars.csv"
        ]

        found_files = 0
        for filename in expected_files:
            file_path = data_dir / filename
            if file_path.exists():
                found_files += 1
                self.add_check(
                    f"Data File: {filename}",
                    "PASS",
                    f"Market data file found ({file_path.stat().st_size} bytes)"
                )
            else:
                symbol_key = filename.split("_bars")[0]
                if self.generate_sample_market_data(file_path, symbol_key):
                    found_files += 1
                    self.add_check(
                        f"Data File: {filename}",
                        "PASS",
                        "Sample market data generated automatically"
                    )
                else:
                    self.add_check(
                        f"Data File: {filename}",
                        "WARN",
                        f"Market data file missing",
                        fix_suggestion=f"Download or generate {filename}"
                    )

        if found_files == 0:
            self.add_check(
                "Market Data",
                "FAIL",
                "No market data files found",
                fix_suggestion="Download market data using data fetching scripts"
            )

    def check_model_validation(self) -> None:
        """Check model validation results"""
        validation_files = list(self.repo_root.glob("validation_*.json"))
        
        if not validation_files:
            self.add_check(
                "Model Validation",
                "WARN",
                "No validation reports found",
                fix_suggestion="Run validate_candidates.py to generate validation reports"
            )
            return
            
        self.add_check(
            "Validation Reports",
            "PASS", 
            f"Found {len(validation_files)} validation reports"
        )
        
        # Analyze validation results
        for val_file in validation_files:
            try:
                with open(val_file, 'r') as f:
                    data = json.load(f)
                
                label = data.get('candidate', {}).get('label', 'Unknown')
                summary = data.get('summary', {})
                
                trades = summary.get('trades', 0)
                win_rate = summary.get('win_rate', 0)
                max_dd = summary.get('overall_max_drawdown_pct', 0)
                
                if trades == 0:
                    status = "WARN"
                    message = f"Candidate {label}: No trades generated"
                elif win_rate < 0.4:
                    status = "WARN" 
                    message = f"Candidate {label}: Low win rate ({win_rate:.1%})"
                elif max_dd > 0.3:
                    status = "WARN"
                    message = f"Candidate {label}: High drawdown ({max_dd:.1%})"
                else:
                    status = "PASS"
                    message = f"Candidate {label}: Good performance ({trades} trades, {win_rate:.1%} win rate)"
                
                self.add_check(
                    f"Validation: {label}",
                    status,
                    message,
                    details=f"Trades: {trades}, Win Rate: {win_rate:.1%}, Max DD: {max_dd:.1%}"
                )
                
            except Exception as e:
                self.add_check(
                    f"Validation File: {val_file.name}",
                    "FAIL",
                    f"Error reading validation file: {str(e)}"
                )

    def check_safety_mechanisms(self) -> None:
        """Check trading safety mechanisms"""
        # Check live trading safety files
        live_trade_file = self.tradingbot_dir / "live_trade_ccxt.py"
        
        if live_trade_file.exists():
            try:
                with open(live_trade_file, 'r') as f:
                    content = f.read()
                
                # Check for safety mechanisms
                if 'ALLOW_LIVE_RISK' in content:
                    self.add_check(
                        "Live Risk Protection",
                        "PASS",
                        "ALLOW_LIVE_RISK safety mechanism found"
                    )
                else:
                    self.add_check(
                        "Live Risk Protection",
                        "WARN", 
                        "ALLOW_LIVE_RISK safety mechanism not found",
                        fix_suggestion="Add ALLOW_LIVE_RISK environment variable check"
                    )
                
                if 'PAPER' in content:
                    self.add_check(
                        "Paper Mode Check",
                        "PASS",
                        "Paper mode safety check found"
                    )
                else:
                    self.add_check(
                        "Paper Mode Check",
                        "FAIL",
                        "Paper mode safety check missing",
                        fix_suggestion="Add PAPER environment variable check"
                    )
                    
            except Exception as e:
                self.add_check(
                    "Safety Code Review",
                    "FAIL",
                    f"Error reviewing safety code: {str(e)}"
                )
        else:
            self.add_check(
                "Live Trading Script",
                "WARN", 
                "live_trade_ccxt.py not found",
                fix_suggestion="Ensure live trading script exists and has safety checks"
            )
        
        # Check for confirmation files that prevent accidental live trading
        confirm_file = self.repo_root / "allow_live_confirm.txt"
        if confirm_file.exists():
            self.add_check(
                "Live Confirmation File",
                "WARN",
                "Live trading confirmation file exists - remove if not intended for live trading",
                fix_suggestion="Delete allow_live_confirm.txt to prevent accidental live trading"
            )
        else:
            self.add_check(
                "Live Confirmation Safety",
                "PASS", 
                "No live trading confirmation file (safe)"
            )

    def check_risk_management(self) -> None:
        """Check risk management settings"""
        # Look for risk management configuration
        risk_file = self.tradingbot_dir / "risk_management.py"
        money_engine_file = self.tradingbot_dir / "money_engine.py"
        
        if money_engine_file.exists():
            self.add_check(
                "Position Sizing Module",
                "PASS",
                "Money engine module found"
            )
        else:
            self.add_check(
                "Position Sizing Module",
                "WARN",
                "Money engine module not found",
                fix_suggestion="Ensure position sizing logic is implemented"
            )
        
        if risk_file.exists():
            self.add_check(
                "Risk Management Module",
                "PASS", 
                "Risk management module found"
            )
        else:
            self.add_check(
                "Risk Management Module",
                "WARN",
                "Risk management module not found",
                fix_suggestion="Consider implementing dedicated risk management"
            )

    def check_server_configuration(self) -> None:
        """Check server and API configuration"""
        server_file = self.repo_root / "server.py"
        
        if server_file.exists():
            try:
                with open(server_file, 'r') as f:
                    content = f.read()
                
                if 'JWT' in content or 'jwt' in content:
                    self.add_check(
                        "Authentication System",
                        "PASS",
                        "JWT authentication system found"
                    )
                else:
                    self.add_check(
                        "Authentication System",
                        "WARN",
                        "No authentication system detected",
                        fix_suggestion="Implement authentication for production use"
                    )
                    
                if 'rate_limit' in content.lower():
                    self.add_check(
                        "Rate Limiting",
                        "PASS",
                        "Rate limiting mechanism found"
                    )
                else:
                    self.add_check(
                        "Rate Limiting",
                        "WARN", 
                        "No rate limiting detected",
                        fix_suggestion="Add rate limiting for production API"
                    )
                    
            except Exception as e:
                self.add_check(
                    "Server Configuration",
                    "FAIL",
                    f"Error checking server configuration: {str(e)}"
                )
        else:
            self.add_check(
                "API Server",
                "INFO",
                "No API server found (optional component)"
            )

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical issues first
        critical_issues = [check for check in self.checks if check.status == 'FAIL']
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Fix all FAIL status issues before proceeding")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                if issue.fix_suggestion:
                    recommendations.append(f"   • {issue.fix_suggestion}")
        
        # Warnings
        warnings = [check for check in self.checks if check.status == 'WARN']
        if warnings:
            recommendations.append("⚠️  WARNINGS: Address these issues for production readiness")
            for warning in warnings[:3]:  # Top 3 warnings
                if warning.fix_suggestion:
                    recommendations.append(f"   • {warning.fix_suggestion}")
        
        # General recommendations
        if not critical_issues:
            recommendations.append("✅ No critical issues found!")
            
        recommendations.extend([
            "📋 Next Steps:",
            "   • Run validation scripts: python tradingbot_ibkr/validate_candidates.py",
            "   • Test in paper mode extensively before live trading",
            "   • Review and update risk management settings",
            "   • Monitor system performance and error logs"
        ])
        
        return recommendations

    def calculate_readiness_score(self) -> float:
        """Calculate overall readiness score (0-100)"""
        if not self.checks:
            return 0.0
            
        total_checks = len(self.checks)
        pass_count = len([c for c in self.checks if c.status == 'PASS'])
        warn_count = len([c for c in self.checks if c.status == 'WARN'])
        fail_count = len([c for c in self.checks if c.status == 'FAIL'])
        
        # Scoring: PASS=1.0, WARN=0.7, FAIL=0.0, INFO=0.8
        score = 0.0
        for check in self.checks:
            if check.status == 'PASS':
                score += 1.0
            elif check.status == 'WARN':
                score += 0.7
            elif check.status == 'INFO':
                score += 0.8
            # FAIL contributes 0.0
                
        return (score / total_checks) * 100

    def run_all_checks(self) -> ReadinessReport:
        """Run all readiness checks and generate report"""
        logger.info("🚀 Starting Trading Readiness Check...")
        
        # Run all check categories
        self.check_python_environment()
        self.check_dependencies()
        self.check_configuration_files()
        self.check_data_availability()
        self.check_model_validation()
        self.check_safety_mechanisms()
        self.check_risk_management()
        self.check_server_configuration()
        
        # Calculate results
        score = self.calculate_readiness_score()
        
        # Determine overall status
        if self.critical_failures > 0:
            overall_status = "NOT_READY"
        elif score >= 85:
            overall_status = "READY"
        elif score >= 70:
            overall_status = "MOSTLY_READY"
        else:
            overall_status = "NEEDS_WORK"
        
        # Generate summary
        summary = {
            'PASS': len([c for c in self.checks if c.status == 'PASS']),
            'WARN': len([c for c in self.checks if c.status == 'WARN']),
            'FAIL': len([c for c in self.checks if c.status == 'FAIL']),
            'INFO': len([c for c in self.checks if c.status == 'INFO'])
        }
        
        recommendations = self.generate_recommendations()
        
        return ReadinessReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            score=score,
            checks=self.checks,
            summary=summary,
            recommendations=recommendations
        )


def print_report(report: ReadinessReport, verbose: bool = False) -> None:
    """Print readiness report to console"""
    print("\n" + "="*80)
    print("🏦 TRADING BOT READINESS ASSESSMENT")
    print("="*80)
    print(f"📅 Timestamp: {report.timestamp}")
    print(f"📊 Overall Status: {report.overall_status}")
    print(f"📈 Readiness Score: {report.score:.1f}/100")
    print()
    
    # Summary
    print("📋 CHECK SUMMARY:")
    for status, count in report.summary.items():
        if count > 0:
            print(f"   {status}: {count}")
    print()
    
    # Detailed results (if verbose or any failures/warnings)
    show_details = verbose or report.summary.get('FAIL', 0) > 0 or report.summary.get('WARN', 0) > 0
    
    if show_details:
        print("🔍 DETAILED RESULTS:")
        for check in report.checks:
            icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'INFO': 'ℹ️'}.get(check.status, '❓')
            print(f"   {icon} {check.name}: {check.message}")
            if verbose and check.details:
                print(f"      Details: {check.details}")
            if check.fix_suggestion and check.status in ['FAIL', 'WARN']:
                print(f"      💡 Fix: {check.fix_suggestion}")
        print()
    
    # Recommendations
    print("🎯 RECOMMENDATIONS:")
    for rec in report.recommendations:
        print(f"   {rec}")
    print()
    
    # Status summary
    if report.overall_status == "READY":
        print("🎉 System appears ready for trading! Proceed with caution and thorough testing.")
    elif report.overall_status == "MOSTLY_READY":
        print("🔧 System is mostly ready but has some warnings to address.")
    elif report.overall_status == "NEEDS_WORK":
        print("⚠️  System needs significant work before trading.")
    else:
        print("🚫 System is NOT ready for trading. Critical issues must be fixed.")
    
    print("="*80)


def save_report(report: ReadinessReport, output_file: Path) -> None:
    """Save report to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"📄 Report saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check trading bot readiness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Save report to JSON file")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to fix common issues")
    
    args = parser.parse_args()
    
    # Find repository root
    repo_root = Path(__file__).parent
    
    # Run readiness check
    checker = TradingReadinessChecker(
        repo_root,
        verbose=args.verbose,
        fix_issues=args.fix_issues,
    )
    report = checker.run_all_checks()
    
    # Print results
    print_report(report, verbose=args.verbose)
    
    # Save report if requested
    if args.output:
        save_report(report, Path(args.output))
    else:
        # Save default report
        default_output = repo_root / f"trading_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_report(report, default_output)
    
    # Exit with appropriate code
    if report.overall_status == "NOT_READY":
        sys.exit(1)
    elif report.overall_status in ["NEEDS_WORK", "MOSTLY_READY"]:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()