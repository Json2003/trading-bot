import polars as pl
from pathlib import Path
import numpy as np

FEATURES = "data/parquet/features_1m"


def triple_barrier(df, up=0.003, down=0.003, max_h=30):
    px = df["close"].to_numpy()
    y = np.zeros(len(px), dtype=np.int8)
    for i in range(len(px) - 1):
        p0 = px[i]
        horizon = px[i + 1 : i + 1 + max_h]
        upper_hits = np.flatnonzero(horizon >= p0 * (1 + up))
        lower_hits = np.flatnonzero(horizon <= p0 * (1 - down))
        first_upper = upper_hits[0] if upper_hits.size else None
        first_lower = lower_hits[0] if lower_hits.size else None
        if first_upper is not None and (first_lower is None or first_upper <= first_lower):
            y[i] = 1
        elif first_lower is not None:
            y[i] = -1
        else:
            y[i] = 0
    return pl.Series("y", y)


def process_symbol(symbol):
    files = sorted(Path(FEATURES).glob(f"symbol={symbol}/date=*/part.parquet"))
    dfs = [pl.read_parquet(fp) for fp in files]
    if not dfs:
        return
    df = pl.concat(dfs).sort("ts")
    df = df.with_columns(
        [
            triple_barrier(df, up=0.003, down=0.003, max_h=30),
            pl.col("ret_1m").cast(pl.Float64).fill_null(0.0),
        ]
    )
    out_dir = Path("data/parquet/labels_1m") / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.select(["symbol", "ts", "y"]).write_parquet(out_dir / "labels.parquet")


if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        process_symbol(sym)
