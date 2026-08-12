# Paper arbitrage guardrails

This profile is a research and paper-trading aid. It does not enable live orders.

The bot must reject an opportunity unless:

- both venue quotes are fresh and available;
- the opportunity is pre-funded on both venues;
- the gross edge covers both trading fees, spread, slippage, and the required net buffer;
- both legs can be submitted as one atomic opportunity;
- partial fills are rejected or flattened by the paper executor;
- trade and inventory notional limits pass;
- the consecutive-loss and daily-loss limits pass.

The default profile models 10 bps per trading fee, 8 bps spread, and 8 bps slippage. Its 60-bps gross threshold leaves 24 bps after modeled costs, including a 9-bps buffer above the 15-bps minimum net edge. These are deliberately conservative placeholders, not a claim about any user's actual exchange tier.

Validate it with:

```bash
python scripts/validate_arbitrage_paper_config.py configs/arbitrage-paper.yaml
pytest -q tests/test_arbitrage_paper_config.py
```

Before any future live consideration, replace the placeholders with measured account-specific fees and fill data, then complete paper probation and explicit human approval. No credential or live-mode change belongs in this profile.
