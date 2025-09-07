@echo off
REM Simple script to set up a Python virtual environment and install dependencies for the trading bot.

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r tradingbot_ibkr\requirements.txt

echo.
echo Installation complete. Activate the environment with "call venv\Scripts\activate".
