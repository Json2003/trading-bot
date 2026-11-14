"""Explain stock indicators and research evidence without changing trade rules."""
from __future__ import annotations

import math

import pandas as pd


def number(value):
    return float(value) if math.isfinite(float(value)) else None


def stock_context(frame, *, symbol, as_of_session, received_at, calculated_at, risk_latched):
    data = frame.loc[frame.session <= as_of_session].copy()
    base = {"symbol": symbol, "observation_session": as_of_session, "input_received_at": received_at,
            "calculated_at": calculated_at, "research_only": True, "execution_authorized": False}
    if len(data) < 200:
        return {**base, "status": "unknown", "action": "WAIT_INSUFFICIENT_DATA", "indicators": {}}
    close = data.close.astype(float)
    sma = {n: close.rolling(n).mean().iloc[-1] for n in (5, 20, 50, 200)}
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).iloc[-1]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    std = close.rolling(20).std(ddof=0).iloc[-1]
    prev = close.shift(1)
    tr = pd.concat([data.high - data.low, (data.high - prev).abs(), (data.low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    roc = close.pct_change(10).ewm(span=3, adjust=False).mean().iloc[-1]
    volume_base = data.volume.shift(1).rolling(20).mean().iloc[-1]
    relative_volume = data.volume.iloc[-1] / volume_base if volume_base > 0 else float("nan")
    support = data.low.shift(1).rolling(20).min().iloc[-1]
    resistance = data.high.shift(1).rolling(20).max().iloc[-1]
    price = close.iloc[-1]
    checks = {"fast_5_above_slow_20": bool(sma[5] > sma[20]),
              "trend_50_above_200": bool(sma[50] > sma[200]),
              "rsi_in_existing_35_75_band": bool(35 <= rsi <= 75),
              "smoothed_10_session_momentum_nonnegative": bool(roc >= 0)}
    buy_setup = all(checks.values())
    conflicts = []
    if rsi > 70 and checks["fast_5_above_slow_20"]:
        conflicts.append("Positive short trend but RSI is above 70; extension is context, not a new veto.")
    if rsi < 30 and macd.iloc[-1] < macd_signal.iloc[-1]:
        conflicts.append("Oversold RSI with negative MACD; oversold alone does not establish a reversal.")
    if buy_setup and relative_volume < 1:
        conflicts.append("Existing buy setup has below-average volume; volume is diagnostic, not a tested gate.")
    action = "PAUSED_DRAWDOWN_LATCH" if risk_latched else ("BUY_SETUP" if buy_setup else "WAIT")
    return {**base, "status": "historical_indicator_diagnostic", "action": action,
            "if_holding_signal_exit": not buy_setup, "existing_strategy_checks": checks,
            "reason": ("The historical risk latch remains set." if risk_latched else
                       "All existing SMA/RSI/momentum conditions agree." if buy_setup else
                       "Missing conditions: " + ", ".join(k for k, passed in checks.items() if not passed)),
            "indicators": {"close": number(price), "sma_5": number(sma[5]), "sma_20": number(sma[20]),
                           "sma_50": number(sma[50]), "sma_200": number(sma[200]),
                           "rsi_14": number(rsi), "macd_12_26": number(macd.iloc[-1]),
                           "macd_signal_9": number(macd_signal.iloc[-1]),
                           "macd_histogram": number(macd.iloc[-1] - macd_signal.iloc[-1]),
                           "bollinger_lower_20_2": number(sma[20] - 2 * std),
                           "bollinger_upper_20_2": number(sma[20] + 2 * std),
                           "atr_14": number(atr), "atr_fraction": number(atr / price),
                           "relative_volume_vs_prior_20": number(relative_volume),
                           "prior_20_session_support": number(support),
                           "prior_20_session_resistance": number(resistance)},
            "indicator_conventions": "RSI uses recursive EWMA alpha=1/14, matching the existing strategy; Bollinger population std; MACD EMA adjust=False.",
            "conflicts": conflicts,
            "missing_context": ["timestamped earnings calendar", "NBBO and auction liquidity", "halt/LULD state",
                                "broad-market and sector benchmarks", "broker settled cash and actual fills"],
            "lesson_tags": ["technical_signals", "momentum", "volatility", "execution", "validation"]}


def review_cycle(report, catalog):
    blocks = []
    for result, benchmark in zip(report.get("results", []), report.get("benchmarks", [])):
        blocks.append({"block": result["block"], "net_pnl": result["net_pnl"],
                       "pnl_before_execution_costs": result["net_pnl"] + result["execution_costs"],
                       "execution_cost_drag": result["execution_costs"],
                       "excess_return_vs_cash_benchmark": result["net_return"] - benchmark["net_return"],
                       "independent_sample_sufficient": result["minimum_sample_met"],
                       "risk_latch_triggered": result["drawdown_latched_at"] is not None,
                       "drawdown_exceeded_latch_threshold": result["max_drawdown"] < -0.02})
    priorities = [
        {"skill": "quote_and_fill_quality", "status": "not_evidenced", "needed": "timestamped NBBO/auction data and IBKR fill reconciliation"},
        {"skill": "earnings_and_corporate_events", "status": "partial", "needed": "point-in-time earnings, splits, delistings and dividend payment-date ledger"},
        {"skill": "cash_settlement", "status": "not_evidenced", "needed": "broker settlement calendar and settled-cash reconciliation"},
        {"skill": "regime_and_portfolio_risk", "status": "partial", "needed": "market/sector context, correlated exposure and prospectively frozen regime tests"},
        {"skill": "neural_to_nyse_execution", "status": "not_verified", "needed": "model provenance, feature schema/freshness, account risk, broker routing and fill audit"},
        {"skill": "profitable_forward_edge", "status": "not_demonstrated", "needed": "positive net untouched forward results with enough independent events"}]
    return {"research_only": True, "model_weights_trained": False, "promotion_allowed": False,
            "catalog_source_count": len(catalog["sources"]), "book_full_texts_ingested": 0,
            "catalog_book_count": sum(source["kind"] == "book" for source in catalog["sources"]),
            "blocks": blocks, "capability_gaps": priorities,
            "lessons": ["Forecast direction and realizable net profit are separate questions.",
                        "Correlated or overlapping trades do not multiply independent evidence.",
                        "An observed two-percent latch cannot cap losses through overnight gaps.",
                        "Indicator agreement and expert agreement do not establish an executable edge."],
            "next_research_rule": "Preserve the rejected diagnostic and its hashes. Any new feature needs its own frozen hypothesis and future confirmation; do not optimize on these results."}


def select_lessons(catalog, tags):
    wanted = set(tags)
    return [{"source_id": source["source_id"], "title": source["title"], "source_url": source["source_url"],
             "kind": source["kind"], "content_scope": source["content_scope"], "lesson": source["lesson"]}
            for source in catalog["sources"] if wanted.intersection(source["skill_tags"])]


def validate_catalog(catalog):
    seen_ids, seen_urls = set(), set()
    for source in catalog["sources"]:
        if source["source_id"] in seen_ids or source["source_url"] in seen_urls:
            raise ValueError("duplicate education source")
        if not source["source_url"].startswith("https://") or not source["skill_tags"] or not source["lesson"]:
            raise ValueError("source needs provenance, tags and a bounded lesson")
        if source["kind"] not in {"book", "research_paper", "expert_opinion", "course", "official_rules"}:
            raise ValueError("unrecognized evidence class")
        seen_ids.add(source["source_id"])
        seen_urls.add(source["source_url"])
