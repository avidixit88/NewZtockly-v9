from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import pandas as pd
import time
import numpy as np

from av_client import AlphaVantageClient
from indicators import rsi as calc_rsi, macd_hist as calc_macd_hist
from signals import compute_scalp_signal, SignalResult


def fetch_bundle(
    client: AlphaVantageClient,
    symbol: str,
    interval: str = "1min",
    *,
    outputsize: str = "full",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, Optional[float]]:
    """Fetch OHLCV + the core indicator series used by the scalping engine.

    NOTE: We default to outputsize='full' so session-aware features (session VWAP, ORB,
    prior-day levels, sweeps, etc.) have enough history to behave correctly for both
    1min and 5min scans.
    """
    ohlcv = client.fetch_intraday(symbol, interval=interval, outputsize=outputsize)
    rsi5 = calc_rsi(ohlcv["close"], 5)
    rsi14 = calc_rsi(ohlcv["close"], 14)
    mh = calc_macd_hist(ohlcv["close"], 12, 26, 9)
    quote = client.fetch_quote(symbol)
    return ohlcv, rsi5, rsi14, mh, quote


def compute_htf_bias(client: AlphaVantageClient, symbol: str, interval: str = "15min", *, outputsize: str = "full") -> Dict[str, object]:
    """
    Simple higher-TF bias:
      - close vs session VWAP (computed locally)
      - EMA20 vs EMA50
      - RSI14 > 55 bullish, <45 bearish
    Returns dict: {bias: BULL/BEAR/NEUTRAL, score: 0..100, details: {...}}
    """
    from indicators import session_vwap, ema

    ohlcv = client.fetch_intraday(symbol, interval=interval, outputsize=outputsize)
    if len(ohlcv) < 60:
        return {"bias": "NEUTRAL", "score": 50, "details": {"reason": "Not enough HTF bars"}}

    df = ohlcv.tail(240).copy()
    df["vwap_sess"] = session_vwap(df)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    r14 = calc_rsi(df["close"], 14).iloc[-1]
    last = float(df["close"].iloc[-1])
    vwap = float(df["vwap_sess"].iloc[-1]) if np.isfinite(df["vwap_sess"].iloc[-1]) else None
    e20 = float(df["ema20"].iloc[-1])
    e50 = float(df["ema50"].iloc[-1])

    bull = 0
    bear = 0
    if vwap is not None:
        if last > vwap: bull += 1
        if last < vwap: bear += 1
    if e20 > e50: bull += 1
    if e20 < e50: bear += 1
    if r14 > 55: bull += 1
    if r14 < 45: bear += 1

    if bull >= 2 and bull > bear:
        return {"bias": "BULL", "score": 70 + 10 * (bull - 2), "details": {"last": last, "vwap": vwap, "ema20": e20, "ema50": e50, "rsi14": float(r14)}}
    if bear >= 2 and bear > bull:
        return {"bias": "BEAR", "score": 70 + 10 * (bear - 2), "details": {"last": last, "vwap": vwap, "ema20": e20, "ema50": e50, "rsi14": float(r14)}}
    return {"bias": "NEUTRAL", "score": 50, "details": {"last": last, "vwap": vwap, "ema20": e20, "ema50": e50, "rsi14": float(r14)}}


def _arm_pending(symbol: str, row: dict, bar_time: str):
    """Store a pending setup that requires next-bar confirmation."""
    st.session_state.pending_confirm[symbol] = {
        "symbol": symbol,
        "bias": row.get("Bias"),
        "score": float(row.get("Score") or 0),
        "entry": row.get("Entry"),
        "stop": row.get("Stop"),
        "tp1": row.get("TP1"),
        "tp2": row.get("TP2"),
        "why": row.get("Why", ""),
        "session": row.get("Session", ""),
        "asof": row.get("AsOf", ""),
        "bar_time": bar_time,
        "created_ts": time.time(),
    }

def _expire_old_pending(max_age_sec: int = 20 * 60):
    """Drop stale pending setups."""
    now = time.time()
    dead = [k for k, v in st.session_state.pending_confirm.items()
            if now - float(v.get("created_ts", now)) > max_age_sec]
    for k in dead:
        st.session_state.pending_confirm.pop(k, None)

def _try_confirm(symbol: str, last_price: float, bar_time: str):
    """Confirm a pending setup on a NEW bar and return an alert payload or None."""
    pend = st.session_state.pending_confirm.get(symbol)
    if not pend:
        return None

    # Only evaluate confirmation on a NEW bar
    if str(bar_time) <= str(pend.get("bar_time", "")):
        return None

    bias = pend.get("bias")
    entry = pend.get("entry")
    if entry is None:
        st.session_state.pending_confirm.pop(symbol, None)
        return None

    try:
        entry_f = float(entry)
    except Exception:
        st.session_state.pending_confirm.pop(symbol, None)
        return None

    ok = False
    if bias == "LONG" and last_price >= entry_f:
        ok = True
    elif bias == "SHORT" and last_price <= entry_f:
        ok = True

    if not ok:
        return None

    payload = {
        "Symbol": pend.get("symbol"),
        "Bias": bias,
        "Score": pend.get("score"),
        "Session": pend.get("session"),
        "Last": last_price,
        "Entry": entry_f,
        "Stop": pend.get("stop"),
        "TP1": pend.get("tp1"),
        "TP2": pend.get("tp2"),
        "Why": (pend.get("why") or "") + " | Confirmed next-bar",
        "AsOf": pend.get("asof"),
    }
    st.session_state.pending_confirm.pop(symbol, None)
    return payload

def scan_watchlist(
    client: AlphaVantageClient,
    symbols: List[str],
    *,
    interval: str = "1min",
    outputsize: str = "full",
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
    # VWAP / Fib / HTF
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    fib_lookback_bars: int = 120,
    enable_htf_bias: bool = False,
    htf_interval: str = "15min",
    htf_strict: bool = False,
    # Liquidity / ORB / execution model
    killzone_preset: str = "Custom (use toggles)",
    liquidity_weighting: float = 0.55,
    orb_minutes: int = 15,
    entry_model: str = "VWAP reclaim limit",
    slippage_mode: str = "Fixed cents",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,
    # Score normalization
    target_atr_pct: float | None = None,
) -> List[SignalResult]:

    htf_map: Dict[str, Dict[str, object]] = {}
    if enable_htf_bias:
        for sym in symbols:
            try:
                htf_map[sym] = compute_htf_bias(client, sym, interval=htf_interval, outputsize=outputsize)
            except Exception as e:
                htf_map[sym] = {"bias": "NEUTRAL", "score": 50, "details": {"error": str(e)}}

    results: List[SignalResult] = []
    for sym in symbols:
        try:
            ohlcv, rsi5, rsi14, mh, quote = fetch_bundle(client, sym, interval=interval, outputsize=outputsize)
            htf = htf_map.get(sym) if enable_htf_bias else None
            res = compute_scalp_signal(
                sym, ohlcv, rsi5, rsi14, mh,
                mode=mode,
                pro_mode=pro_mode,
                allow_opening=allow_opening,
                allow_midday=allow_midday,
                allow_power=allow_power,
                allow_premarket=allow_premarket,
                allow_afterhours=allow_afterhours,
                use_last_closed_only=use_last_closed_only,
                bar_closed_guard=bar_closed_guard,
                interval=interval,
                vwap_logic=vwap_logic,
                session_vwap_include_premarket=session_vwap_include_premarket,
                fib_lookback_bars=fib_lookback_bars,
                htf_bias=htf,
                htf_strict=htf_strict,
                killzone_preset=killzone_preset,
                liquidity_weighting=liquidity_weighting,
                orb_minutes=orb_minutes,
                entry_model=entry_model,
                slippage_mode=slippage_mode,
                fixed_slippage_cents=fixed_slippage_cents,
                atr_fraction_slippage=atr_fraction_slippage,
                target_atr_pct=target_atr_pct,
)
            # Use quote if present
            if quote is not None:
                res.last_price = quote  # type: ignore
            results.append(res)
        except Exception as e:
            results.append(SignalResult(sym, "NEUTRAL", 0, f"Fetch error: {e}", None, None, None, None, None, None, "OFF", {"error": str(e)}))
    # Rank actionable setups (LONG/SHORT) above blocked/NEUTRAL ones.
    # This prevents a high *potential* score from sitting at the top when
    # hard requirements (VWAP reclaim, RSI snap, MACD turn, volume, Pro triggers)
    # were not met and therefore no entry/TP can be produced.
    def _rank_key(r: SignalResult):
        actionable = 1 if r.bias in ("LONG", "SHORT") else 0
        try:
            score = int(r.setup_score or 0)
        except Exception:
            score = 0
        return (actionable, score)

    results.sort(key=_rank_key, reverse=True)
    return results
