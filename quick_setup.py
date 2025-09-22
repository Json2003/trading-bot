#!/usr/bin/env python3
"""
Quick Setup Script for Trading Bot

This script helps set up the trading bot environment by:
1. Creating .env file from template
2. Installing dependencies
3. Creating necessary directories
4. Running initial readiness check

Usage:
    python quick_setup.py [--install-deps]
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def create_env_file(repo_root: Path) -> None:
    """Create .env file from .env.example if needed"""
    env_file = repo_root / ".env"
    env_example = repo_root / "tradingbot_ibkr" / ".env.example"
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
        
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("📝 Please edit .env file with your API keys and settings")
    else:
        # Create basic .env file
        env_content = """# General
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
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created basic .env file")
        print("📝 Please edit .env file with your API keys and settings")


def create_directories(repo_root: Path) -> None:
    """Create necessary directories"""
    dirs_to_create = [
        repo_root / "tradingbot_ibkr" / "datafiles",
        repo_root / "tradingbot_ibkr" / "model_store",
        repo_root / "data",
        repo_root / "artifacts",
        repo_root / "logs"
    ]
    
    for directory in dirs_to_create:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")


def install_dependencies(repo_root: Path) -> bool:
    """Try to install dependencies"""
    requirements_file = repo_root / "tradingbot_ibkr" / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing dependencies...")
        # Install core dependencies first
        core_deps = ["python-dotenv", "pandas", "numpy", "ccxt"]
        for dep in core_deps:
            print(f"   Installing {dep}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"   ✅ {dep} installed successfully")
            else:
                print(f"   ❌ Failed to install {dep}: {result.stderr}")
                return False
        
        print("✅ Core dependencies installed")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {str(e)}")
        return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick setup for trading bot")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Attempt to install dependencies")
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent
    
    print("🚀 Trading Bot Quick Setup")
    print("="*50)
    
    # Create .env file
    create_env_file(repo_root)
    
    # Create directories
    create_directories(repo_root)
    
    # Install dependencies if requested
    if args.install_deps:
        success = install_dependencies(repo_root)
        if not success:
            print("⚠️  Some dependencies failed to install. Try manual installation.")
    
    print("\n🎯 Next Steps:")
    print("1. Edit .env file with your API keys")
    print("2. If dependencies installation failed, run: pip install -r tradingbot_ibkr/requirements.txt")
    print("3. Download market data or run data fetching scripts")
    print("4. Run readiness check: python check_trading_readiness.py")
    print("5. Start with paper trading mode (PAPER=true)")
    
    print("\n✅ Quick setup completed!")


if __name__ == "__main__":
    main()