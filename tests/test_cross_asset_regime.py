from datetime import date, timedelta

from scripts.run_cross_asset_regime import regime_at


def _inputs(current: date):
    macro = {
        name: {current: 100.0}
        for name in ("SPY", "QQQ", "TLT", "UUP", "VIX")
    }
    macro["VIX"].update({current - timedelta(days=i): 30.0 for i in range(1, 21)})
    emas = {name: {current: 100.0} for name in macro}
    return macro, emas


def test_boom_requires_four_of_five_markers():
    current = date(2024, 1, 22)
    macro, emas = _inputs(current)
    macro.update({
        "SPY": {current: 110.0}, "QQQ": {current: 110.0},
        "TLT": {current: 90.0}, "UUP": {current: 90.0},
        "VIX": {**macro["VIX"], current: 15.0},
    })
    assert regime_at(macro, emas, current) == (1, 5, 0)


def test_bust_is_the_inverse_of_boom():
    current = date(2024, 1, 22)
    macro, emas = _inputs(current)
    macro.update({
        "SPY": {current: 90.0}, "QQQ": {current: 90.0},
        "TLT": {current: 110.0}, "UUP": {current: 110.0},
        "VIX": {**macro["VIX"], current: 40.0},
    })
    assert regime_at(macro, emas, current) == (-1, 0, 5)


def test_mixed_markers_are_flat_and_missing_data_is_unknown():
    current = date(2024, 1, 22)
    macro, emas = _inputs(current)
    macro.update({
        "SPY": {current: 110.0}, "QQQ": {current: 90.0},
        "TLT": {current: 90.0}, "UUP": {current: 110.0},
        "VIX": {**macro["VIX"], current: 15.0},
    })
    assert regime_at(macro, emas, current) == (0, 3, 2)
    del macro["UUP"][current]
    assert regime_at(macro, emas, current) is None
