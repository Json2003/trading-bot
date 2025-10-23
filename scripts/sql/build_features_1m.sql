PRAGMA threads=system;
INSTALL httpfs; LOAD httpfs; -- ignore if local files only

ATTACH 'file:data/duckdb/featurestore.duckdb' AS fs (READ_ONLY FALSE);
CREATE SCHEMA IF NOT EXISTS fs.market;

-- Create a view over partitioned Parquet
CREATE OR REPLACE VIEW fs.ohlcv_1m AS
SELECT *
FROM read_parquet('data/parquet/ohlcv_1m/symbol=*/date=*/part.parquet');

-- Build features (right-closed windows exclude current row via 1 PRECEDING)
CREATE OR REPLACE TABLE fs.market_features_1m AS
WITH base AS (
  SELECT
    symbol,
    ts,
    CAST(ts AS DATE) AS date,
    open, high, low, close, volume,
    LN(close / LAG(close) OVER (PARTITION BY symbol ORDER BY ts)) AS ret_1m
  FROM fs.ohlcv_1m
),
feat AS (
  SELECT
    symbol, ts, date, open, high, low, close, volume, ret_1m,
    -- rolling sums (include up to 1 PRECEDING to avoid leakage)
    SUM(ret_1m) OVER w5 AS ret_5m,
    SUM(ret_1m) OVER w30 AS ret_30m,
    SQRT(SUM(ret_1m*ret_1m) OVER w30) AS vol_realized_30m,
    (close - AVG(close) OVER w30) / NULLIF(STDDEV_SAMP(close) OVER w30,0) AS zscore_30m,
    -- RSI(14) approximation
    100.0 * SAFE_DIVIDE(
      SUM(GREATEST(ret_1m,0)) OVER w14,
      NULLIF(SUM(ABS(ret_1m)) OVER w14,0)
    ) AS rsi_14
  FROM base
  WINDOW
    w5  AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 5 PRECEDING  AND 1 PRECEDING),
    w14 AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING),
    w30 AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING)
)
SELECT * FROM feat;

-- Optional: also write back to partitioned parquet for training
COPY (SELECT * FROM fs.market_features_1m)
TO 'data/parquet/features_1m/symbol={symbol}/date={date}/part.parquet' 
(WITH PARTITION_BY (symbol, date), FORMAT PARQUET, OVERWRITE_OR_IGNORE);
