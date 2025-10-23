"""Adaptive utilities package.

This module re-exports regime classification and parameter helpers from
backtest.regime to provide stable import paths:

- backtest.adaptive.regimes.classify_regime
- backtest.adaptive.param_policy.params_for_regime
- backtest.adaptive.candidate_gen.around
"""

from .regimes import classify_regime  # re-export for convenience
from .param_policy import params_for_regime, ATRExit, TrendATRPolicy
from .candidate_gen import around
from .dqn_agent import DQNAgent, Transition, build_transition_stream

__all__ = [
    "classify_regime",
    "params_for_regime",
    "ATRExit",
    "TrendATRPolicy",
    "around",
    "DQNAgent",
    "Transition",
    "build_transition_stream",
]
