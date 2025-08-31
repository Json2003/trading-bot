def compute_basic_metrics(trades):
    """Compute simple metrics from a list of trade dicts with 'pnl' values."""
    total = len(trades)
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    pnl = sum(t.get('pnl', 0) for t in trades)
    win_rate = (wins / total * 100) if total else 0.0
    return {'trades': total, 'wins': wins, 'win_rate_pct': win_rate, 'pnl': pnl}
