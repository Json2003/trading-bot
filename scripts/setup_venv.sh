#!/bin/bash

# Create venv
python -m venv venv

# Activate and install deps
source venv/bin/activate
pip install -r tradingbot_ibkr/requirements.txt

echo "Installation complete. Activate with: source venv/bin/activate"
