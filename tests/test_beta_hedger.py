from math import isclose

from engine.beta_hedger import BetaHedgeCfg, BetaHedger


def test_returns_zero_when_equity_is_non_positive() -> None:
    hedger = BetaHedger(BetaHedgeCfg())
    exposures = {"ETHUSDT": 1000.0}
    betas = {"ETHUSDT": 1.5}

    assert hedger.hedge_notional(exposures, betas, 0.0) == 0.0
    assert hedger.hedge_notional(exposures, betas, -100.0) == 0.0


def test_hedge_notional_short_when_beta_above_target() -> None:
    hedger = BetaHedger(BetaHedgeCfg(target_beta=0.15, rebalance_thresh=0.01))
    exposures = {"ETHUSDT": 5000.0}
    betas = {"ETHUSDT": 1.0}

    assert isclose(hedger.hedge_notional(exposures, betas, 10000.0), -3500.0)


def test_hedge_notional_long_when_beta_below_target() -> None:
    hedger = BetaHedger(BetaHedgeCfg(target_beta=0.2, rebalance_thresh=0.01))
    exposures = {"ETHUSDT": 100.0}
    betas = {"ETHUSDT": 0.5}

    # current beta = 0.05, gap = -0.15 -> hedge long 0.15 * equity
    assert isclose(hedger.hedge_notional(exposures, betas, 1000.0), 150.0)


def test_triggers_rebalance_when_gap_exceeds_threshold() -> None:
    hedger = BetaHedger(BetaHedgeCfg(target_beta=0.15, rebalance_thresh=0.05))
    exposures = {"ETHUSDT": 1400.0}
    betas = {"ETHUSDT": 0.18}

    # current beta = 0.0252, gap = -0.1248 -> outside threshold -> positive hedge
    assert isclose(hedger.hedge_notional(exposures, betas, 10000.0), 1248.0)


def test_returns_zero_when_gap_within_threshold() -> None:
    hedger = BetaHedger(BetaHedgeCfg(target_beta=0.15, rebalance_thresh=0.2))
    exposures = {"ETHUSDT": 2000.0}
    betas = {"ETHUSDT": 0.16}

    # current beta = 0.032, gap = -0.118 -> within threshold -> no action
    assert isclose(hedger.hedge_notional(exposures, betas, 10000.0), 0.0)


def test_beta_is_clipped_before_contributing() -> None:
    hedger = BetaHedger(BetaHedgeCfg(beta_clip=1.5, target_beta=0.0, rebalance_thresh=0.0))
    exposures = {"ETHUSDT": 1000.0}
    betas = {"ETHUSDT": 10.0}

    # clipped beta = 1.5 -> gap = 1.5 -> hedge short 1500
    assert isclose(hedger.hedge_notional(exposures, betas, 1000.0), -1500.0)
