"""Reconciliation helpers for validating backtest performance envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping, Sequence

import yaml


class BacktestProfileNotFoundError(KeyError):
    """Raised when a requested backtest profile is not present in metadata."""


@dataclass(frozen=True)
class MetricExpectation:
    """Expectation for a single metric captured in a backtest profile."""

    name: str
    target: float
    tolerance: float
    comparison: str
    description: str | None = None

    def bounds(self) -> tuple[float | None, float | None]:
        """Return the lower and upper bounds implied by the expectation."""

        if self.comparison == "min":
            return self.target - self.tolerance, None
        if self.comparison == "max":
            return None, self.target + self.tolerance
        raise ValueError(f"Unsupported comparison operator: {self.comparison!r}")

    def evaluate(self, actual: float) -> "MetricEvaluation":
        """Evaluate *actual* against the configured expectation."""

        lower, upper = self.bounds()
        within_bounds = True
        if lower is not None and actual < lower:
            within_bounds = False
        if upper is not None and actual > upper:
            within_bounds = False

        return MetricEvaluation(
            name=self.name,
            actual=actual,
            target=self.target,
            tolerance=self.tolerance,
            comparison=self.comparison,
            description=self.description,
            lower_bound=lower,
            upper_bound=upper,
            within_bounds=within_bounds,
        )


@dataclass(frozen=True)
class MetricEvaluation:
    """Result of reconciling an observed metric against its expectation."""

    name: str
    actual: float
    target: float
    tolerance: float
    comparison: str
    description: str | None
    lower_bound: float | None
    upper_bound: float | None
    within_bounds: bool


@dataclass(frozen=True)
class BacktestProfile:
    """Metadata describing an expected performance envelope for a strategy."""

    name: str
    strategy: str
    market: str
    timeframe: str
    metrics: tuple[MetricExpectation, ...]
    notes: str | None
    tags: tuple[str, ...]

    def evaluate(self, metrics: Mapping[str, float]) -> "BacktestEvaluation":
        """Evaluate the provided *metrics* against the profile expectations."""

        evaluations = []
        lookup = {expectation.name: expectation for expectation in self.metrics}
        for name, expectation in lookup.items():
            if name not in metrics:
                raise KeyError(
                    f"Metric '{name}' missing from evaluation payload for profile '{self.name}'"
                )
            evaluations.append(expectation.evaluate(metrics[name]))

        return BacktestEvaluation(
            profile=self,
            metrics=tuple(evaluations),
        )


@dataclass(frozen=True)
class BacktestEvaluation:
    """Aggregate evaluation outcome for a backtest profile."""

    profile: BacktestProfile
    metrics: tuple[MetricEvaluation, ...]

    @property
    def passed(self) -> bool:
        """Return ``True`` if all evaluated metrics satisfied their bounds."""

        return all(metric.within_bounds for metric in self.metrics)

    @property
    def breaches(self) -> tuple[MetricEvaluation, ...]:
        """Return the subset of metric evaluations that failed validation."""

        return tuple(metric for metric in self.metrics if not metric.within_bounds)


def load_backtest_profiles(path: str | Path) -> dict[str, BacktestProfile]:
    """Load backtest profiles from *path*.

    The file format is a YAML document containing a ``profiles`` mapping.  Each
    profile may optionally define ``notes`` and ``tags``.  Unknown keys are
    ignored which allows teams to extend the metadata without code changes.
    """

    raw = _load_yaml(path)
    profiles_data = raw.get("profiles") or {}

    profiles: dict[str, BacktestProfile] = {}
    for name, payload in profiles_data.items():
        metrics_payload = payload.get("metrics") or {}
        metrics = tuple(_parse_metric(name, metric_name, metric_payload) for metric_name, metric_payload in metrics_payload.items())

        notes = payload.get("notes")
        tags_payload = payload.get("tags") or []
        tags = tuple(str(tag) for tag in tags_payload)

        profiles[name] = BacktestProfile(
            name=name,
            strategy=str(payload.get("strategy", name)),
            market=str(payload.get("market", "<unknown>")),
            timeframe=str(payload.get("timeframe", "<unknown>")),
            metrics=metrics,
            notes=str(notes) if notes is not None else None,
            tags=tags,
        )

    return profiles


def _load_yaml(path: str | Path) -> MutableMapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise TypeError("Backtest metadata root element must be a mapping")
    return data


def _parse_metric(profile: str, name: str, payload: Mapping[str, object]) -> MetricExpectation:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Metric '{name}' in profile '{profile}' must be a mapping")

    try:
        target = float(payload["target"])
        tolerance = float(payload.get("tolerance", 0.0))
        comparison = str(payload.get("comparison", "min"))
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Metric '{name}' missing required key: {exc.args[0]}") from exc

    description_value = payload.get("description")
    description = str(description_value) if description_value is not None else None

    expectation = MetricExpectation(
        name=name,
        target=target,
        tolerance=tolerance,
        comparison=comparison,
        description=description,
    )

    # Validate comparison early so configuration errors surface immediately.
    expectation.bounds()
    return expectation


class BacktestReconciler:
    """Helper that validates observed metrics against stored expectations."""

    def __init__(self, profiles: Mapping[str, BacktestProfile] | None = None):
        self._profiles: dict[str, BacktestProfile] = dict(profiles or {})

    @classmethod
    def from_path(cls, path: str | Path) -> "BacktestReconciler":
        """Construct a reconciler by loading profiles from *path*."""

        return cls(load_backtest_profiles(path))

    def register_profile(self, profile: BacktestProfile) -> None:
        """Register or override a profile in the reconciler."""

        self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> BacktestProfile:
        """Return the profile named *name* or raise :class:`BacktestProfileNotFoundError`."""

        try:
            return self._profiles[name]
        except KeyError as exc:
            raise BacktestProfileNotFoundError(name) from exc

    def evaluate(self, name: str, metrics: Mapping[str, float]) -> BacktestEvaluation:
        """Evaluate *metrics* for the profile identified by *name*."""

        profile = self.get_profile(name)
        return profile.evaluate(metrics)

    def profiles(self) -> Sequence[BacktestProfile]:
        """Return the registered profiles sorted by name for deterministic output."""

        return tuple(self._profiles[name] for name in sorted(self._profiles))

    def __iter__(self) -> Iterator[BacktestProfile]:
        return iter(self.profiles())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._profiles)
