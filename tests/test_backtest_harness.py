from __future__ import annotations

import json
import platform
from pathlib import Path

from tradingbot_core.backtest_harness import BacktestContext, BacktestHarness


def test_backtest_harness_persists_metrics(tmp_path: Path) -> None:
    harness = BacktestHarness(
        output_dir=tmp_path,
        metadata=BacktestContext(
            strategy="mean_reversion",
            market="BTC/USDT",
            timeframe="1h",
            seed=1337,
            broker_fees={"maker": 0.0005},
            tags=("regression",),
        ),
        time_provider=lambda: 123.0,
    )

    def runner() -> dict[str, object]:
        return {
            "returns": [0.01, -0.005, 0.007],
            "equity_curve": [1.0, 1.01, 1.00495, 1.01201465],
            "trades": [
                {"symbol": "BTC", "qty": 1, "price": 25000},
                {"symbol": "BTC", "qty": -1, "price": 25500},
            ],
            "fees_paid": 12.5,
        }

    output_path = harness.run(runner)

    assert output_path.exists()

    payload = json.loads(output_path.read_text())

    assert payload["meta"]["git_sha"] == "<local>"
    assert payload["meta"]["python"] == platform.python_version()
    assert isinstance(payload["meta"]["deps"], str)
    assert payload["meta"]["timestamp"].endswith("Z") or payload["meta"]["timestamp"].endswith(
        "+00:00"
    )

    results = payload["results"]

    assert results["metadata"]["strategy"] == "mean_reversion"
    assert results["metadata"]["fees"] == {"maker": 0.0005}
    assert results["metadata"]["seed"] == 1337
    assert results["metadata"]["captured_at"] == 123.0

    metrics = results["result"]["metrics"]
    assert set(metrics.keys()) == {"sharpe", "sortino", "max_drawdown", "cvar_95"}
    assert metrics["max_drawdown"] >= 0

    assert results["fees_paid"] == 12.5
