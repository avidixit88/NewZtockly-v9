from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import math

from indicators import (
    vwap as calc_vwap,
    session_vwap as calc_session_vwap,
    atr as calc_atr,
    ema as calc_ema,
    adx as calc_adx,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    find_breaker_block,
    in_zone,
)
from sessions import classify_session, classify_liquidity_phase


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100 (calibrated)
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/PREMARKET/AFTERHOURS/OFF
    extras: Dict[str, Any]


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def _fib_retracement_levels(hi: float, lo: float) -> List[Tuple[str, float]]:
    ratios = [0.382, 0.5, 0.618, 0.786]
    rng = hi - lo
    if rng <= 0:
        return []
    # "pullback" levels for an up-move: hi - r*(hi-lo)
    return [(f"Fib {r:g}", hi - r * rng) for r in ratios]


def _fib_extensions(hi: float, lo: float) -> List[Tuple[str, float]]:
    # extensions above hi for longs, below lo for shorts (we'll mirror in logic)
    ratios = [1.0, 1.272, 1.618]
    rng = hi - lo
    if rng <= 0:
        return []
    return [(f"Ext {r:g}", hi + (r - 1.0) * rng) for r in ratios]


def _closest_level(price: float, levels: List[Tuple[str, float]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not levels:
        return None, None, None
    name, lvl = min(levels, key=lambda x: abs(price - x[1]))
    return name, float(lvl), float(abs(price - lvl))


def _session_liquidity_levels(df: pd.DataFrame, interval_mins: int, orb_minutes: int):
    """Compute simple liquidity levels: prior session high/low, today's premarket high/low, and ORB high/low."""
    if df is None or len(df) < 5:
        return {}
    # normalize timestamps to ET
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    else:
        ts = pd.to_datetime(df.index)

    try:
        ts = ts.dt.tz_localize("America/New_York") if getattr(ts.dt, "tz", None) is None else ts.dt.tz_convert("America/New_York")
    except Exception:
        try:
            ts = ts.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            # if tz ops fail, fall back to naive dates
            pass

    d = df.copy()
    d["_ts"] = ts
    # derive dates
    try:
        cur_date = d["_ts"].iloc[-1].date()
        dates = sorted({x.date() for x in d["_ts"] if pd.notna(x)})
    except Exception:
        cur_date = pd.to_datetime(df.index[-1]).date()
        dates = sorted({pd.to_datetime(x).date() for x in df.index})

    prev_date = dates[-2] if len(dates) >= 2 else cur_date

    def _t(x):
        try:
            return x.time()
        except Exception:
            return None

    def _is_pre(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("04:00").time()) and (t < pd.Timestamp("09:30").time())

    def _is_rth(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("09:30").time()) and (t <= pd.Timestamp("16:00").time())

    prev = d[d["_ts"].dt.date == prev_date] if "_ts" in d else df.iloc[:0]
    prev_rth = prev[prev["_ts"].apply(_is_rth)] if len(prev) else prev
    prior_high = float(prev_rth["high"].max()) if len(prev_rth) else (float(prev["high"].max()) if len(prev) else None)
    prior_low = float(prev_rth["low"].min()) if len(prev_rth) else (float(prev["low"].min()) if len(prev) else None)

    cur = d[d["_ts"].dt.date == cur_date] if "_ts" in d else df
    cur_pre = cur[cur["_ts"].apply(_is_pre)] if len(cur) else cur
    pre_hi = float(cur_pre["high"].max()) if len(cur_pre) else None
    pre_lo = float(cur_pre["low"].min()) if len(cur_pre) else None

    cur_rth = cur[cur["_ts"].apply(_is_rth)] if len(cur) else cur
    orb_bars = max(1, int(math.ceil(float(orb_minutes) / max(float(interval_mins), 1.0))))
    orb_slice = cur_rth.head(orb_bars)
    orb_hi = float(orb_slice["high"].max()) if len(orb_slice) else None
    orb_lo = float(orb_slice["low"].min()) if len(orb_slice) else None

    return {
        "prior_high": prior_high, "prior_low": prior_low,
        "premarket_high": pre_hi, "premarket_low": pre_lo,
        "orb_high": orb_hi, "orb_low": orb_lo,
    }

def _asof_slice(df: pd.DataFrame, interval_mins: int, use_last_closed_only: bool, bar_closed_guard: bool) -> pd.DataFrame:
    """Return df truncated so the last row represents the 'as-of' bar we can legally use."""
    if df is None or len(df) < 3:
        return df
    asof_idx = len(df) - 1

    # Always allow "snapshot mode" to use last fully completed bar
    if use_last_closed_only:
        asof_idx = max(0, len(df) - 2)

    if bar_closed_guard and len(df) >= 2:
        try:
            # Determine timestamp of latest bar
            if "time" in df.columns:
                last_ts = pd.to_datetime(df["time"].iloc[-1], utc=False)
            else:
                last_ts = pd.to_datetime(df.index[-1], utc=False)

            # Normalize to ET if timezone-naive
            now = pd.Timestamp.now(tz="America/New_York")
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("America/New_York")
            else:
                last_ts = last_ts.tz_convert("America/New_York")

            bar_end = last_ts + pd.Timedelta(minutes=int(interval_mins))
            # If bar hasn't ended yet, step back one candle (avoid partial)
            if now < bar_end:
                asof_idx = min(asof_idx, len(df) - 2)
        except Exception:
            # If anything goes sideways, be conservative
            asof_idx = min(asof_idx, len(df) - 2)

    asof_idx = max(0, int(asof_idx))
    return df.iloc[: asof_idx + 1].copy()

def _detect_liquidity_sweep(df: pd.DataFrame, levels: dict):
    """Simple sweep: wick through a key level then close back inside."""
    if df is None or len(df) < 2 or not levels:
        return None
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    ph = levels.get("prior_high")
    pl = levels.get("prior_low")
    if ph is not None and h > ph and c < ph:
        return {"type": "bear_sweep_prior_high", "level": float(ph)}
    if pl is not None and l < pl and c > pl:
        return {"type": "bull_sweep_prior_low", "level": float(pl)}

    pmah = levels.get("premarket_high")
    pmal = levels.get("premarket_low")
    if pmah is not None and h > pmah and c < pmah:
        return {"type": "bear_sweep_premarket_high", "level": float(pmah)}
    if pmal is not None and l < pmal and c > pmal:
        return {"type": "bull_sweep_premarket_low", "level": float(pmal)}

    return None


def _detect_rsi_divergence(
    df: pd.DataFrame,
    rsi: pd.Series,
    *,
    lookback: int = 120,
    pivot_lr: int = 3,
    min_price_delta_atr: float = 0.15,
    min_rsi_delta: float = 3.0,
) -> Optional[Dict[str, float | str]]:
    """Detect a basic RSI divergence using the last two swing points.

    Bullish divergence: price makes lower-low, RSI makes higher-low.
    Bearish divergence: price makes higher-high, RSI makes lower-high.

    Returns a dict like:
      {"type": "bull"|"bear", "strength": float, "price_a":..., "price_b":..., "rsi_a":..., "rsi_b":...}
    or None.
    """
    if df is None or len(df) < 20 or rsi is None or len(rsi) < 20:
        return None

    d = df.tail(int(min(max(40, lookback), len(df)))).copy()
    r = rsi.reindex(d.index).ffill()
    if r.isna().all():
        return None

    # ATR proxy for scaling price deltas.
    atr_last = None
    try:
        if "atr14" in d.columns and np.isfinite(d["atr14"].iloc[-1]):
            atr_last = float(d["atr14"].iloc[-1])
    except Exception:
        atr_last = None
    if not atr_last or atr_last <= 0:
        try:
            atr_last = float(np.nanmedian((d["high"].astype(float) - d["low"].astype(float)).tail(20).values))
        except Exception:
            atr_last = 0.0
    if not atr_last or not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = 0.0

    low_mask = rolling_swing_lows(d["low"].astype(float), left=int(pivot_lr), right=int(pivot_lr))
    hi_mask = rolling_swing_highs(d["high"].astype(float), left=int(pivot_lr), right=int(pivot_lr))

    lows = d.loc[low_mask, "low"].astype(float).tail(2)
    highs = d.loc[hi_mask, "high"].astype(float).tail(2)

    if len(lows) == 2:
        idx_a, idx_b = lows.index[0], lows.index[1]
        p_a, p_b = float(lows.iloc[0]), float(lows.iloc[1])
        r_a, r_b = float(r.loc[idx_a]), float(r.loc[idx_b])
        if p_b < p_a and r_b > r_a:
            price_delta = abs(p_b - p_a)
            rsi_delta = abs(r_b - r_a)
            if (atr_last <= 0 or price_delta >= float(min_price_delta_atr) * atr_last) and rsi_delta >= float(min_rsi_delta):
                strength = (rsi_delta / 10.0) + (price_delta / (atr_last + 1e-9))
                return {"type": "bull", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    if len(highs) == 2:
        idx_a, idx_b = highs.index[0], highs.index[1]
        p_a, p_b = float(highs.iloc[0]), float(highs.iloc[1])
        r_a, r_b = float(r.loc[idx_a]), float(r.loc[idx_b])
        if p_b > p_a and r_b < r_a:
            price_delta = abs(p_b - p_a)
            rsi_delta = abs(r_b - r_a)
            if (atr_last <= 0 or price_delta >= float(min_price_delta_atr) * atr_last) and rsi_delta >= float(min_rsi_delta):
                strength = (rsi_delta / 10.0) + (price_delta / (atr_last + 1e-9))
                return {"type": "bear", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    return None


def _compute_atr_pct_series(df: pd.DataFrame, period: int = 14):
    if df is None or len(df) < period + 2:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / close.replace(0, np.nan)


def _apply_atr_score_normalization(score: float, df: pd.DataFrame, lookback: int = 200, period: int = 14):
    atr_pct = _compute_atr_pct_series(df, period=period)
    if atr_pct is None:
        return score, None, None, 1.0
    cur = atr_pct.iloc[-1]
    if pd.isna(cur) or float(cur) <= 0:
        return score, (None if pd.isna(cur) else float(cur)), None, 1.0
    tail = atr_pct.dropna().tail(int(lookback))
    baseline = float(tail.median()) if len(tail) else None
    if baseline is None or baseline <= 0:
        return score, float(cur), baseline, 1.0
    scale = float(baseline / float(cur))
    scale = max(0.75, min(1.35, scale))
    return max(0.0, min(100.0, float(score) * scale)), float(cur), baseline, scale

def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,

    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",

    # VWAP / Fib / HTF
    lookback_bars: int = 180,
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    session_vwap_include_afterhours: bool = False,
    fib_lookback_bars: int = 120,
    htf_bias: Optional[Dict[str, object]] = None,   # {bias, score, details}
    htf_strict: bool = False,

    # Liquidity / ORB / execution model
    killzone_preset: str = "Custom (use toggles)",
    liquidity_weighting: float = 0.55,
    orb_minutes: int = 15,
    entry_model: str = "VWAP reclaim limit",
    slippage_mode: str = "Off",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,

    # Score normalization
    target_atr_pct: float | None = None,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    # --- Interval parsing ---
    # interval is typically like "1min", "5min", "15min", "30min", "60min"
    interval_mins = 1
    try:
        s = str(interval).lower().strip()
        if s.endswith("min"):
            interval_mins = int(float(s.replace("min", "").strip()))
        elif s.endswith("m"):
            interval_mins = int(float(s.replace("m", "").strip()))
        else:
            interval_mins = int(float(s))
    except Exception:
        interval_mins = 1

    # --- Killzone presets ---
    # Presets can optionally override the time-of-day allow toggles.
    kz = (killzone_preset or "Custom (use toggles)").strip()
    if kz == "Opening Drive":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = True, False, False, False, False
    elif kz == "Lunch Chop":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, True, False, False, False
    elif kz == "Power Hour":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, True, False, False
    elif kz == "Pre-market":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, False, True, False

    # --- Snapshot / bar-closed guards ---
    try:
        df_asof = _asof_slice(ohlcv.copy(), interval_mins=interval_mins, use_last_closed_only=use_last_closed_only, bar_closed_guard=bar_closed_guard)
    except Exception:
        df_asof = ohlcv.copy()

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = df_asof.copy().tail(int(lookback_bars)).copy()
    # Session VWAP windows are session-dependent. If the user enables scanning PM/AH but keeps
    # session VWAP restricted to RTH, VWAP-based logic becomes NaN during those windows.
    # As a product guardrail, automatically extend session VWAP to the scanned session(s).
    auto_vwap_fix = False
    if vwap_logic == "session":
        if allow_premarket and not session_vwap_include_premarket:
            session_vwap_include_premarket = True
            auto_vwap_fix = True
        if allow_afterhours and not session_vwap_include_afterhours:
            session_vwap_include_afterhours = True
            auto_vwap_fix = True

    df["vwap_cum"] = calc_vwap(df)
    df["vwap_sess"] = calc_session_vwap(
        df,
        include_premarket=session_vwap_include_premarket,
        include_afterhours=session_vwap_include_afterhours,
    )
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    # Pro: Trend strength (ADX) + direction (DI+/DI-)
    adx14 = plus_di = minus_di = None
    try:
        adx_s, pdi_s, mdi_s = calc_adx(df, 14)
        df["adx14"] = adx_s
        df["plus_di14"] = pdi_s
        df["minus_di14"] = mdi_s
        adx14 = float(adx_s.iloc[-1]) if len(adx_s) and np.isfinite(adx_s.iloc[-1]) else None
        plus_di = float(pdi_s.iloc[-1]) if len(pdi_s) and np.isfinite(pdi_s.iloc[-1]) else None
        minus_di = float(mdi_s.iloc[-1]) if len(mdi_s) and np.isfinite(mdi_s.iloc[-1]) else None
    except Exception:
        adx14 = plus_di = minus_di = None

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap_use = df["vwap_sess"] if vwap_logic == "session" else df["vwap_cum"]

    last_ts = df.index[-1]
    # Feed freshness diagnostics (ET): this helps catch the "AsOf is yesterday" case.
    try:
        now_et = pd.Timestamp.now(tz="America/New_York")
        ts_et = last_ts.tz_convert("America/New_York") if last_ts.tzinfo is not None else last_ts.tz_localize("America/New_York")
        data_age_min = float((now_et - ts_et).total_seconds() / 60.0)
        extras_feed = {"data_age_min": data_age_min, "data_date": str(ts_et.date())}
    except Exception:
        extras_feed = {"data_age_min": None, "data_date": None}
    session = classify_session(last_ts)
    phase = classify_liquidity_phase(last_ts)

    # IMPORTANT PRODUCT RULE:
    # Time-of-day toggles should NOT *block* scoring/alerts.
    # They are preference hints used for liquidity weighting and optional UI filtering.
    # A great setup is a great setup regardless of clock-time.
    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
        or (session == "PREMARKET" and allow_premarket)
        or (session == "AFTERHOURS" and allow_afterhours)
    )
    last_price = float(close.iloc[-1])

    atr_last = float(df["atr14"].iloc[-1]) if np.isfinite(df["atr14"].iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0
    atr_pct = (atr_last / last_price) if last_price else 0.0

    # Liquidity weighting: scale contributions based on the current liquidity phase.
    # liquidity_weighting in [0..1] controls how strongly we care about time-of-day liquidity.
    #  - OPENING / POWER: boost
    #  - MIDDAY: discount
    #  - PREMARKET / AFTERHOURS: heavier discount
    base = 1.0
    if phase in ("OPENING", "POWER"):
        base = 1.15
    elif phase in ("MIDDAY",):
        base = 0.85
    elif phase in ("PREMARKET", "AFTERHOURS"):
        base = 0.75
    try:
        w = max(0.0, min(1.0, float(liquidity_weighting)))
    except Exception:
        w = 0.55
    liquidity_mult = 1.0 + w * (base - 1.0)

    extras: Dict[str, Any] = {
        "vwap_logic": vwap_logic,
        "session_vwap_include_premarket": bool(session_vwap_include_premarket),
        "session_vwap_include_afterhours": bool(session_vwap_include_afterhours),
        "auto_vwap_session_fix": bool(auto_vwap_fix),
        "vwap_session": float(df["vwap_sess"].iloc[-1]) if np.isfinite(df["vwap_sess"].iloc[-1]) else None,
        "vwap_cumulative": float(df["vwap_cum"].iloc[-1]) if np.isfinite(df["vwap_cum"].iloc[-1]) else None,
        "ema20": float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None,
        "ema50": float(df["ema50"].iloc[-1]) if np.isfinite(df["ema50"].iloc[-1]) else None,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "atr14": atr_last,
        "atr_pct": atr_pct,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "liquidity_phase": phase,
        "liquidity_mult": liquidity_mult,
        "fib_lookback_bars": int(fib_lookback_bars),
        "htf_bias": htf_bias,
        "htf_strict": bool(htf_strict),
        "target_atr_pct": (float(target_atr_pct) if target_atr_pct is not None else None),
        # Diagnostics: whether the current session is inside the user's preferred windows.
        # This is NEVER used to block actionability.
        "time_filter_allowed": bool(allowed),
    }

    # Attach feed diagnostics (age/date) to every result.
    try:
        extras.update(extras_feed)
    except Exception:
        pass

    # merge feed freshness fields
    extras.update(extras_feed)

    # Do not early-return when outside preferred windows.
    # We keep scoring normally and simply annotate the result.

    # VWAP event
    was_below_vwap = (close.shift(3) < vwap_use.shift(3)).iloc[-1] or (close.shift(5) < vwap_use.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap_use.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap_use.shift(3)).iloc[-1] or (close.shift(5) > vwap_use.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap_use.shift(1).iloc[-1])

    # RSI + MACD events
    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    # Pro: RSI divergence (RSI-5 vs price pivots)
    rsi_div = None
    if pro_mode:
        try:
            rsi_div = _detect_rsi_divergence(df, rsi_fast, lookback=int(min(160, max(60, lookback_bars))))
        except Exception:
            rsi_div = None
    extras["rsi_divergence"] = rsi_div

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    # Volume confirmation (liquidity weighted)
    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Swings
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    # Trend context (EMA)
    trend_long_ok = bool((close.iloc[-1] >= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] >= df["ema50"].iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] <= df["ema50"].iloc[-1]))
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    # Fib context (scoring + fib-anchored take profits)
    seg = df.tail(int(min(max(60, fib_lookback_bars), len(df))))
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    rng = hi - lo

    fib_name = fib_level = fib_dist = None
    fib_near_long = fib_near_short = False
    fib_bias = "range"
    retr = _fib_retracement_levels(hi, lo) if rng > 0 else []
    fib_name, fib_level, fib_dist = _closest_level(last_price, retr)

    if rng > 0:
        pos = (last_price - lo) / rng
        if pos >= 0.60:
            fib_bias = "up"
        elif pos <= 0.40:
            fib_bias = "down"
        else:
            fib_bias = "range"

    if fib_level is not None and fib_dist is not None:
        near = fib_dist <= max(buffer, 0.0) if atr_last else (fib_dist <= (0.002 * last_price))
        if near:
            if fib_bias == "up":
                fib_near_long = True
            elif fib_bias == "down":
                fib_near_short = True

    extras["fib_hi"] = hi if rng > 0 else None
    extras["fib_lo"] = lo if rng > 0 else None
    extras["fib_bias"] = fib_bias
    extras["fib_closest"] = {"name": fib_name, "level": fib_level, "dist": fib_dist}
    extras["fib_near_long"] = fib_near_long
    extras["fib_near_short"] = fib_near_short

    # Liquidity sweeps + ORB context
    # Use session-aware levels (prior day high/low, premarket high/low, ORB high/low) when possible.
    try:
        levels = _session_liquidity_levels(df, interval_mins=interval_mins, orb_minutes=int(orb_minutes))
    except Exception:
        levels = {}

    extras["liq_levels"] = levels

    # Fallback swing-based levels (always available)
    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())

    # Sweep definition:
    # - Primary: wick through a key level, then close back inside (ICT-style)
    # - Secondary fallback: take + reclaim against recent swing
    bull_sweep = False
    bear_sweep = False
    if pro_mode and levels:
        sweep = _detect_liquidity_sweep(df, levels)
        extras["liquidity_sweep"] = sweep
        if isinstance(sweep, dict) and sweep.get("type"):
            stype = str(sweep.get("type")).lower()
            bull_sweep = stype.startswith("bull")
            bear_sweep = stype.startswith("bear")
    else:
        bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low))
        bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high))

    extras["bull_liquidity_sweep"] = bool(bull_sweep)
    extras["bear_liquidity_sweep"] = bool(bear_sweep)

    # ORB bias (simple): break and hold above ORB high (bull) / below ORB low (bear)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    extras["orb_high"] = orb_high
    extras["orb_low"] = orb_low
    orb_bull = bool(orb_high is not None and last_price > float(orb_high))
    orb_bear = bool(orb_low is not None and last_price < float(orb_low))
    extras["orb_bull_break"] = orb_bull
    extras["orb_bear_break"] = orb_bear


    # FVG + OB + Breaker
    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, df["atr14"], side="bull", lookback=35)
    ob_bear = find_order_block(df, df["atr14"], side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear
    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    brk_bull = find_breaker_block(df, df["atr14"], side="bull", lookback=60)
    brk_bear = find_breaker_block(df, df["atr14"], side="bear", lookback=60)
    extras["bull_breaker"] = brk_bull
    extras["bear_breaker"] = brk_bear
    bull_breaker_retest = bool(brk_bull[0] is not None and in_zone(last_price, brk_bull[0], brk_bull[1], buffer=buffer))
    bear_breaker_retest = bool(brk_bear[0] is not None and in_zone(last_price, brk_bear[0], brk_bear[1], buffer=buffer))
    extras["bull_breaker_retest"] = bull_breaker_retest
    extras["bear_breaker_retest"] = bear_breaker_retest

    displacement = bool(atr_last and float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 1.5 * atr_last)
    extras["displacement"] = displacement

    # HTF bias overlay
    htf_b = None
    if isinstance(htf_bias, dict):
        htf_b = htf_bias.get("bias")
    extras["htf_bias_value"] = htf_b

    # --- Scoring (raw) ---
    contrib: Dict[str, Dict[str, int]] = {"LONG": {}, "SHORT": {}}

    def _add(side: str, key: str, pts: int, why: str | None = None):
        nonlocal long_points, short_points
        if side == "LONG":
            long_points += int(pts)
            contrib["LONG"][key] = contrib["LONG"].get(key, 0) + int(pts)
            if why:
                long_reasons.append(why)
        else:
            short_points += int(pts)
            contrib["SHORT"][key] = contrib["SHORT"].get(key, 0) + int(pts)
            if why:
                short_reasons.append(why)

    long_points = 0
    long_reasons: List[str] = []
    if was_below_vwap and reclaim_vwap:
        _add("LONG", "vwap_event", 35, f"VWAP reclaim ({vwap_logic})")
    if rsi_snap and rsi14 < 60:
        _add("LONG", "rsi_snap", 20, "RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        _add("LONG", "macd_turn", 20, "MACD hist turning up")
    if vol_ok:
        _add("LONG", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if df["low"].tail(12).iloc[-1] > df["low"].tail(12).min():
        _add("LONG", "micro_structure", 10, "Higher-low micro structure")

    short_points = 0
    short_reasons: List[str] = []
    if was_above_vwap and reject_vwap:
        _add("SHORT", "vwap_event", 35, f"VWAP rejection ({vwap_logic})")
    if rsi_downshift and rsi14 > 40:
        _add("SHORT", "rsi_downshift", 20, "RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        _add("SHORT", "macd_turn", 20, "MACD hist turning down")
    if vol_ok:
        _add("SHORT", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if df["high"].tail(12).iloc[-1] < df["high"].tail(12).max():
        _add("SHORT", "micro_structure", 10, "Lower-high micro structure")

    # Fib scoring
    if fib_near_long and fib_name is not None:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("LONG", "fib", add, f"Near {fib_name}")
    if fib_near_short and fib_name is not None:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("SHORT", "fib", add, f"Near {fib_name}")

    # Pro structure scoring
    if pro_mode:
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bull":
            _add("LONG", "rsi_divergence", 22, "RSI bullish divergence")
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bear":
            _add("SHORT", "rsi_divergence", 22, "RSI bearish divergence")
        if bull_sweep:
            _add("LONG", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (low)")
        if bear_sweep:
            _add("SHORT", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (high)")
        if orb_bull:
            _add("LONG", "orb", int(round(12 * liquidity_mult)), f"ORB break ({orb_minutes}m)")
        if orb_bear:
            _add("SHORT", "orb", int(round(12 * liquidity_mult)), f"ORB break ({orb_minutes}m)")
        if bull_ob_retest:
            _add("LONG", "order_block", 15, "Bullish order block retest")
        if bear_ob_retest:
            _add("SHORT", "order_block", 15, "Bearish order block retest")
        if bull_fvg is not None:
            _add("LONG", "fvg", 10, "Bullish FVG present")
        if bear_fvg is not None:
            _add("SHORT", "fvg", 10, "Bearish FVG present")
        if bull_breaker_retest:
            _add("LONG", "breaker", 20, "Bullish breaker retest")
        if bear_breaker_retest:
            _add("SHORT", "breaker", 20, "Bearish breaker retest")
        if displacement:
            _add("LONG", "displacement", 5, None)
            _add("SHORT", "displacement", 5, None)

        # ADX trend-strength bonus (directional): helps avoid low-energy chop.
        # - If ADX is strong and DI agrees with direction => small bonus.
        # - If ADX is very low => mild penalty (but don't over-filter reversal setups).
        try:
            adx_val = float(adx14) if adx14 is not None else None
            pdi_val = float(plus_di) if plus_di is not None else None
            mdi_val = float(minus_di) if minus_di is not None else None
        except Exception:
            adx_val = pdi_val = mdi_val = None

        if adx_val is not None and np.isfinite(adx_val):
            if adx_val >= 20 and pdi_val is not None and mdi_val is not None:
                if pdi_val > mdi_val:
                    _add("LONG", "adx_trend", 8, "ADX trend strength (DI+)")
                elif mdi_val > pdi_val:
                    _add("SHORT", "adx_trend", 8, "ADX trend strength (DI-)")
            elif adx_val <= 15:
                # Penalize both slightly during very low trend strength
                long_points = max(0, long_points - 5)
                short_points = max(0, short_points - 5)
                contrib["LONG"]["adx_chop_penalty"] = contrib["LONG"].get("adx_chop_penalty", 0) - 5
                contrib["SHORT"]["adx_chop_penalty"] = contrib["SHORT"].get("adx_chop_penalty", 0) - 5

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # HTF overlay scoring
    if htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL":
            long_points += 10; long_reasons.append("HTF bias bullish")
            short_points = max(0, short_points - 10)
        elif htf_b == "BEAR":
            short_points += 10; short_reasons.append("HTF bias bearish")
            long_points = max(0, long_points - 10)

    # Requirements / Gatekeeping (product-safe)
    # Historically we hard-gated on VWAP reclaim/rejection. That works during liquid RTH,
    # but it causes "high-score but no alert" situations in PREMARKET/AFTERHOURS where
    # VWAP events can be sparse/noisy.
    #
    # New rule:
    #   - Keep VWAP as a top contributor, but don't let it be the sole gatekeeper.
    #   - During PREMARKET/AFTERHOURS, relax VWAP/RSI/MACD hard requirements.
    #   - In Pro mode, allow structural triggers (divergence/sweep/ORB/OB/breaker) to satisfy gating.
    vwap_event = bool((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap))
    rsi_event = bool(rsi_snap or rsi_downshift)
    macd_event = bool(macd_turn_up or macd_turn_down)
    volume_event = bool(vol_ok)

    is_extended_session = session in ("PREMARKET", "AFTERHOURS")
    extras["gates"] = {
        "vwap_event": vwap_event,
        "rsi_event": rsi_event,
        "macd_event": macd_event,
        "volume_event": volume_event,
        "extended_session": bool(is_extended_session),
    }

    # Pro structural trigger (if enabled)
    pro_trigger = False
    if pro_mode:
        pro_trigger = bool(
            bull_sweep or bear_sweep
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or orb_bull or orb_bear
            or (isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
        )
        extras["pro_trigger"] = pro_trigger

    # Hard volume requirement stays hard (when enabled)
    if int(cfg["require_volume"]) == 1 and not volume_event:
        return SignalResult(
            symbol, "NEUTRAL", int(max(long_points, short_points)),
            "No volume confirmation",
            None, None, None, None,
            last_price, last_ts, session, extras,
        )

    # For RTH, honor preset hard requirements, but allow Pro triggers to substitute.
    # For extended sessions (PM/AH), treat them as soft requirements (penalize vs block).
    hard_vwap = (int(cfg["require_vwap_event"]) == 1) and (not is_extended_session)
    hard_rsi  = (int(cfg["require_rsi_event"]) == 1) and (not is_extended_session)
    hard_macd = (int(cfg["require_macd_turn"]) == 1) and (not is_extended_session)

    # If hard-gated and we have no alternative trigger, block.
    if hard_vwap and (not vwap_event) and (not pro_trigger):
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
    if hard_rsi and (not rsi_event) and (not pro_trigger):
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
    if hard_macd and (not macd_event) and (not pro_trigger):
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)

    # Extended sessions (PM/AH): we *record* missing classic triggers for transparency,
    # but we do not penalize the score. The score should reflect the setup quality;
    # gating is handled separately via `primary_trigger`.
    if is_extended_session:
        if int(cfg["require_vwap_event"]) == 1 and (not vwap_event) and (not pro_trigger):
            extras["soft_gate_missing_vwap"] = True
        if int(cfg["require_rsi_event"]) == 1 and (not rsi_event) and (not pro_trigger):
            extras["soft_gate_missing_rsi"] = True
        if int(cfg["require_macd_turn"]) == 1 and (not macd_event) and (not pro_trigger):
            extras["soft_gate_missing_macd"] = True

    # Final sanity: require at least one primary trigger to produce actionable levels.
    primary_trigger = bool(vwap_event or rsi_event or macd_event or pro_trigger)
    extras["primary_trigger"] = primary_trigger
    if not primary_trigger:
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No primary trigger (VWAP/RSI/MACD/Pro)", None, None, None, None, last_price, last_ts, session, extras)

    # HTF strict filter (optional)
    if htf_strict and htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL" and not (was_below_vwap and reclaim_vwap):
            # only allow longs if bullish HTF and long setup
            pass
        if htf_b == "BEAR" and not (was_above_vwap and reject_vwap):
            pass

    # ATR-normalized score calibration (per ticker)
    # If target_atr_pct is None => auto-tune per ticker using median ATR% over a recent window.
    # Otherwise => use the manual target ATR% as a global anchor.
    scale = 1.0
    ref_atr_pct = None
    if atr_pct:
        if target_atr_pct is None:
            atr_series = df["atr14"].tail(120)
            close_series = df["close"].tail(120).replace(0, np.nan)
            atr_pct_series = (atr_series / close_series).replace([np.inf, -np.inf], np.nan).dropna()
            if len(atr_pct_series) >= 20:
                ref_atr_pct = float(np.nanmedian(atr_pct_series.values))
        else:
            ref_atr_pct = float(target_atr_pct)

        if ref_atr_pct and ref_atr_pct > 0:
            scale = ref_atr_pct / atr_pct
            # Keep calibration gentle; we want comparability, not distortion.
            scale = float(np.clip(scale, 0.75, 1.25))

    extras["atr_score_scale"] = scale
    extras["atr_ref_pct"] = ref_atr_pct

    long_points_cal = int(round(long_points * scale))
    short_points_cal = int(round(short_points * scale))
    extras["long_points_raw"] = long_points
    extras["short_points_raw"] = short_points
    extras["long_points_cal"] = long_points_cal
    extras["short_points_cal"] = short_points_cal
    extras["contrib_points"] = contrib

    min_score = int(cfg["min_actionable_score"])

    # Entry/stop + targets
    tighten_factor = 1.0
    if pro_mode:
        # Tighten stops a bit when we have structural confluence.
        # NOTE: We intentionally do NOT mutate the setup_score here; scoring is handled above.
        confluence = bool(
            (isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
            or bull_sweep or bear_sweep
            or orb_bull or orb_bear
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or (bull_fvg is not None) or (bear_fvg is not None)
        )
        if confluence:
            tighten_factor = 0.85
        extras["stop_tighten_factor"] = float(tighten_factor)

    def _fib_take_profits_long(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        exts = _fib_extensions(hi, lo)
        # Partial at recent high if above entry, else at ext 1.272
        tp1 = hi if entry_px < hi else next((lvl for _, lvl in exts if lvl > entry_px), None)
        tp2 = next((lvl for _, lvl in exts if lvl and tp1 and lvl > tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _fib_take_profits_short(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        # Mirror extensions below lo
        ratios = [1.0, 1.272, 1.618]
        exts_dn = [ (f"Ext -{r:g}", lo - (r - 1.0) * rng) for r in ratios ]
        tp1 = lo if entry_px > lo else next((lvl for _, lvl in exts_dn if lvl < entry_px), None)
        tp2 = next((lvl for _, lvl in exts_dn if lvl and tp1 and lvl < tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _long_entry_stop(entry_px: float):
        stop_px = float(min(recent_swing_low, entry_px - max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px - (entry_px - stop_px) * tighten_factor)
        if bull_breaker_retest and brk_bull[0] is not None:
            stop_px = float(min(stop_px, brk_bull[0] - buffer))
        if fib_near_long and fib_level is not None:
            stop_px = float(min(stop_px, fib_level - buffer))
        return entry_px, stop_px

    def _short_entry_stop(entry_px: float):
        stop_px = float(max(recent_swing_high, entry_px + max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px + (stop_px - entry_px) * tighten_factor)
        if bear_breaker_retest and brk_bear[1] is not None:
            stop_px = float(max(stop_px, brk_bear[1] + buffer))
        if fib_near_short and fib_level is not None:
            stop_px = float(max(stop_px, fib_level + buffer))
        return entry_px, stop_px
    # Final decision + trade levels
    long_score = int(round(float(long_points_cal))) if 'long_points_cal' in locals() else int(round(float(long_points)))
    short_score = int(round(float(short_points_cal))) if 'short_points_cal' in locals() else int(round(float(short_points)))

    if long_score < min_score and short_score < min_score:
        reason = "Score below threshold"
        extras["decision"] = {"long": long_score, "short": short_score, "min": min_score}
        return SignalResult(symbol, "NEUTRAL", int(max(long_score, short_score)), reason, None, None, None, None, last_price, last_ts, session, extras)

    bias = "LONG" if long_score >= short_score else "SHORT"
    setup_score = int(max(long_score, short_score))

    # Assemble reason text from the winning side
    if bias == "LONG":
        reasons = long_reasons[:] if 'long_reasons' in locals() else []
    else:
        reasons = short_reasons[:] if 'short_reasons' in locals() else []

    reason = "; ".join(reasons) if reasons else "Actionable setup"

    # Entry model context
    ref_vwap = None
    try:
        ref_vwap = float(vwap_use.iloc[-1])
    except Exception:
        ref_vwap = None

    mid_price = None
    try:
        mid_price = float((df["high"].iloc[-1] + df["low"].iloc[-1]) / 2.0)
    except Exception:
        mid_price = None

    entry_px = _entry_from_model(
        bias,
        entry_model=entry_model,
        last_price=float(last_price),
        ref_vwap=ref_vwap,
        mid_price=mid_price,
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    if bias == "LONG":
        entry_px, stop_px = _long_entry_stop(entry_px)
        risk = max(1e-9, entry_px - stop_px)
        tp1 = entry_px + risk
        tp2 = entry_px + 2 * risk
        # If fib extension helper is available, prefer it for pro mode.
        if pro_mode and "_fib_take_profits_long" in locals():
            f1, f2 = _fib_take_profits_long(entry_px)
            tp1 = f1 if f1 is not None else tp1
            tp2 = f2 if f2 is not None else tp2
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
    else:
        entry_px, stop_px = _short_entry_stop(entry_px)
        risk = max(1e-9, stop_px - entry_px)
        tp1 = entry_px - risk
        tp2 = entry_px - 2 * risk
        if pro_mode and "_fib_take_profits_short" in locals():
            f1, f2 = _fib_take_profits_short(entry_px)
            tp1 = f1 if f1 is not None else tp1
            tp2 = f2 if f2 is not None else tp2
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None

    extras["decision"] = {"bias": bias, "long": long_score, "short": short_score, "min": min_score}
    return SignalResult(symbol, bias, setup_score, reason, float(entry_px), float(stop_px), float(tp1) if tp1 is not None else None, float(tp2) if tp2 is not None else None, last_price, last_ts, session, extras)

def _slip_amount(*, slippage_mode: str, fixed_slippage_cents: float, atr_last: float, atr_fraction_slippage: float) -> float:
    """Return slippage amount in price units (not percent)."""
    try:
        mode = (slippage_mode or "Off").strip()
    except Exception:
        mode = "Off"

    if mode == "Off":
        return 0.0

    if mode == "Fixed cents":
        try:
            return max(0.0, float(fixed_slippage_cents)) / 100.0
        except Exception:
            return 0.0

    if mode == "ATR fraction":
        try:
            return max(0.0, float(atr_last)) * max(0.0, float(atr_fraction_slippage))
        except Exception:
            return 0.0

    return 0.0
def _entry_from_model(
    direction: str,
    *,
    entry_model: str,
    last_price: float,
    ref_vwap: float | None,
    mid_price: float | None,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> float:
    """Compute an execution-realistic entry based on the selected entry model."""
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=atr_last,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    model = (entry_model or "Last price").strip()

    # 1) VWAP-based: place a limit slightly beyond VWAP in the adverse direction (more realistic fills).
    if model == "VWAP reclaim limit" and isinstance(ref_vwap, (float, int)):
        return (float(ref_vwap) + slip) if direction == "LONG" else (float(ref_vwap) - slip)

    # 2) Midpoint of the last completed bar
    if model == "Midpoint (last closed bar)" and isinstance(mid_price, (float, int)):
        return (float(mid_price) + slip) if direction == "LONG" else (float(mid_price) - slip)

    # 3) Default: last price with slippage in the adverse direction
    return (float(last_price) + slip) if direction == "LONG" else (float(last_price) - slip)
