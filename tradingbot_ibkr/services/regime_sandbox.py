"""Simulation sandbox to replay stress regimes or synthetic shocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class RegimeScenario:
    name: str
    features: torch.Tensor
    targets: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None


class RegimeSandbox:
    """Loads replay windows and applies user-provided evaluation callbacks."""

    def __init__(self, scenarios: Sequence[RegimeScenario]) -> None:
        self.scenarios = list(scenarios)
        if not self.scenarios:
            raise ValueError("RegimeSandbox requires at least one scenario")

    @classmethod
    def from_parquet(cls, paths: Iterable[Path], feature_cols: Optional[Sequence[str]] = None) -> "RegimeSandbox":
        import pandas as pd  # local import to avoid mandatory dependency elsewhere

        scenarios: List[RegimeScenario] = []
        for path in paths:
            df = pd.read_parquet(path)
            cols = feature_cols or [c for c in df.columns if c not in ("target", "label")]
            features = torch.tensor(df[cols].values, dtype=torch.float32)
            targets = None
            for candidate in ("target", "label"):
                if candidate in df.columns:
                    targets = torch.tensor(df[candidate].values, dtype=torch.float32)
                    break
            scenarios.append(RegimeScenario(name=path.stem, features=features, targets=targets))
        return cls(scenarios)

    def run(self, evaluator: Callable[[RegimeScenario], dict]) -> List[dict]:
        """Apply evaluator to each scenario and collect metrics."""
        reports: List[dict] = []
        for scenario in self.scenarios:
            report = evaluator(scenario)
            report["scenario"] = scenario.name
            reports.append(report)
        return reports

    def inject_shock(self, scale: float = 3.0) -> "RegimeSandbox":
        """Return a new sandbox with synthetic volatility shocks applied."""
        shocked: List[RegimeScenario] = []
        for idx, scenario in enumerate(self.scenarios):
            noise = torch.randn_like(scenario.features) * scale
            shocked_features = scenario.features + noise
            shocked.append(
                RegimeScenario(
                    name=f"{scenario.name}_shock{idx}",
                    features=shocked_features,
                    targets=scenario.targets,
                    metadata={"base": scenario.name, "shock_scale": scale},
                )
            )
        return RegimeSandbox(shocked)

    def to_json(self, path: Path) -> None:
        payload = []
        for scenario in self.scenarios:
            payload.append(
                {
                    "name": scenario.name,
                    "features": scenario.features.tolist(),
                    "targets": scenario.targets.tolist() if scenario.targets is not None else None,
                    "metadata": scenario.metadata,
                }
            )
        path.write_text(json.dumps(payload, indent=2))
