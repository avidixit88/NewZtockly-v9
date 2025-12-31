from __future__ import annotations

import pandas as pd
import pytz
from datetime import time


ET = pytz.timezone("America/New_York")


def _to_et(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize(ET)
    return ts.tz_convert(ET)


def classify_liquidity_phase(ts: pd.Timestamp) -> str:
    """
    Returns: PREMARKET / RTH / AFTERHOURS
    Premarket: 04:00–09:30 ET
    RTH:       09:30–16:00 ET
    After:     16:00–20:00 ET (approx; AV can include extended)
    """
    t = _to_et(ts).time()
    if time(4, 0) <= t < time(9, 30):
        return "PREMARKET"
    if time(9, 30) <= t < time(16, 0):
        return "RTH"
    return "AFTERHOURS"


def classify_session(ts: pd.Timestamp) -> str:
    """
    Returns: PREMARKET / OPENING / MIDDAY / POWER / AFTERHOURS
    """
    phase = classify_liquidity_phase(ts)
    if phase == "PREMARKET":
        return "PREMARKET"
    if phase == "AFTERHOURS":
        return "AFTERHOURS"

    t = _to_et(ts).time()
    if time(9, 30) <= t < time(11, 0):
        return "OPENING"
    if time(11, 0) <= t < time(15, 0):
        return "MIDDAY"
    if time(15, 0) <= t < time(16, 0):
        return "POWER"
    return "OFF"
