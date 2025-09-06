#!/usr/bin/env bash
# Simple script to set up a Python virtual environment and install dependencies for the trading bot.
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
. venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r tradingbot_ibkr/requirements.txt

echo "\nInstallation complete. Activate the environment with 'source venv/bin/activate'."
