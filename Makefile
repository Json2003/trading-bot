.PHONY: daily-trends show-trends

daily-trends:
@python -m src.pipelines.daily_trends_pipeline

show-trends:
@python - <<'PY'
import duckdb, os
db = "data/daily/market_trends.duckdb"
if not os.path.exists(db):
    raise SystemExit("No DB yet. Run `make daily-trends` first.")
con = duckdb.connect(db)
print(con.execute("SELECT date, symbol, asset_class, close, rsi_14, macd_hist, vol_20d FROM market_trends ORDER BY ts DESC LIMIT 30").df())
PY
