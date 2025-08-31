# tradingbot_ibkr

Minimal example folder showing how to run CCXT backtests/live and IBKR live scripts.

Setup (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:
- For crypto live: set `PAPER=false`, `EXCHANGE`, `API_KEY`, `API_SECRET`.
- For IBKR live: run TWS or IB Gateway (LIVE), enable API (port 7496), set `IBKR_PORT=7496` in `.env`.

Run:

```powershell
python backtest_ccxt.py
python live_trade_ccxt.py           # live crypto
python ibkr_live_stock_bracket.py   # live stocks (PDT if margin < $25k)
python ibkr_live_forex_bracket.py   # live FX (no PDT)
```

Notes:
- These scripts are minimal examples for demonstration. Review and test in paper mode before enabling live trading.

Roman Bot desktop app
---------------------

You can launch a lightweight desktop app that hosts the dashboard using `pywebview`:

```powershell
pip install -r requirements.txt
python roman_bot.py
```

This starts the Flask dashboard and opens a native window titled "Roman Bot" with the Centurion icon.

Building a single EXE (Windows)
--------------------------------

You can bundle the app into a single Windows executable using PyInstaller.

1. Activate your venv and install PyInstaller:

```powershell
pip install pyinstaller
```

2. Run the included build script from the `tradingbot_ibkr` folder:

```powershell
cd tradingbot_ibkr
.\build_exe.ps1
```

If the build succeeds you'll find `RomanBot.exe` under `tradingbot_ibkr\dist`.

Notes:
- The build script includes templates, static assets, models, and datafiles. If you add new folders, update `build_exe.ps1` and `roman_bot.spec`.
- Building on Windows is recommended; cross-building (Linux -> Windows) may fail for some native packages.


Hybrid model training and safety
--------------------------------

This repo includes a hybrid learning pipeline (online + batch) under `models/`.

Files:
- `models/online_trainer.py` — River-based incremental trainer (updates per bar)
- `models/train_batch.py` — scikit-learn batch trainer with time-series CV
- `models/promote.py` — safety-promote helper; promotion requires env `ALLOW_MODEL_PROMOTE=true` and an on-disk file `allow_live_confirm.txt` in the project root.

How to run a batch training (example):

1. Collect bars into `data/datafiles/<symbol>_bars.csv` (the `data/store.py` helper can append bars)
2. Run batch training interactively (example):

```powershell
python -c "from models.train_batch import train_and_evaluate; import pandas as pd; df=pd.read_csv('datafiles/BTC_USDT_bars.csv', parse_dates=['ts']); print(train_and_evaluate(df))"
```

Promotion safety:
- To promote a candidate model to production you must set `ALLOW_MODEL_PROMOTE=true` in `.env` and create an empty file `allow_live_confirm.txt` in the repo root. This prevents accidental promotions.

Dashboard and demo training
---------------------------

Start the dashboard (runs on port 5001):

```powershell
python dashboard.py
```

Open http://localhost:5001 in your browser. Click "Run Demo Training" to fetch recent CCXT bars, append them to the data store, and run the batch trainer.

Notes:
- The demo training runs synchronously and may take a short time; check console logs.
- For real usage, wire the online trainer into your live data pipeline and view model status here.


