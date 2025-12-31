import time
import os
import pandas as pd
import numpy as np
import streamlit as st
from email_utils import send_email_alert, format_alert_email
import plotly.graph_objects as go

from av_client import AlphaVantageClient
from engine import scan_watchlist, fetch_bundle
from indicators import vwap as calc_vwap, session_vwap as calc_session_vwap
from signals import compute_scalp_signal, PRESETS

def load_email_secrets():
    """Load email settings from Streamlit Secrets."""
    email_tbl = st.secrets.get("email", {})
    smtp_server = email_tbl.get("smtp_server") or st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(email_tbl.get("smtp_port") or st.secrets.get("SMTP_PORT", 587))
    smtp_user = email_tbl.get("smtp_user") or st.secrets.get("SMTP_USER", "")
    smtp_password = email_tbl.get("smtp_password") or st.secrets.get("SMTP_APP_PASSWORD", "")
    to_email = email_tbl.get("to_email") or st.secrets.get("ALERT_TO", smtp_user)
    return smtp_server, smtp_port, smtp_user, smtp_password, to_email

def send_email_safe(payload: dict, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str, to_email: str):
    """Send an email alert and return (ok, err_msg)."""
    if not (smtp_user and smtp_password and to_email):
        return False, "Missing SMTP secrets"
    try:
        subject = f"Ztockly Alert: {payload.get('Symbol','?')} {payload.get('Bias','')}"
        body = format_alert_email(payload)
        send_email_alert(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_email=to_email,
            subject=subject,
            body=body,
        )
        return True, ""
    except Exception as e:
        return False, str(e)

st.set_page_config(page_title="Ztockly Scalping Scanner", layout="wide")

# Persist results across reruns
st.session_state.setdefault('last_results', None)
st.session_state.setdefault('last_df_view', None)
st.session_state.setdefault('last_scan_ts', None)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"]
if "last_alert_ts" not in st.session_state:
    st.session_state.last_alert_ts = {}
if "pending_confirm" not in st.session_state:
    # per-symbol pending setup waiting for next-bar confirmation (only used when auto-refresh is ON)
    st.session_state.pending_confirm = {}
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Track per-symbol state so alerts only fire on *new* actionable transitions.
# This prevents re-sending the same alert on every auto-refresh rerun.
if "symbol_state" not in st.session_state:
    # {SYM: {"bias": str, "actionable": bool, "score": float}}
    st.session_state.symbol_state = {}

st.sidebar.title("Scalping Scanner")
watchlist_text = st.sidebar.text_area("Watchlist (comma or newline separated)", value="\n".join(st.session_state.watchlist), height=150)

interval = st.sidebar.selectbox("Intraday interval", ["1min", "5min"], index=0)
interval_mins = int(interval.replace("min","").strip())
mode = st.sidebar.selectbox("Signal mode", list(PRESETS.keys()), index=list(PRESETS.keys()).index("Cleaner signals"))

st.sidebar.markdown("#### Causality / bar guards")
use_last_closed_only = st.sidebar.toggle("Use last completed bar only (snapshot)", value=True, help="Uses the last fully completed candle for indicator reads.")
bar_closed_guard = st.sidebar.toggle("Bar-closed guard (avoid partial current bar)", value=True, help="Steps back if the latest candle is still forming.")


st.sidebar.markdown("### VWAP")
vwap_logic = st.sidebar.selectbox("VWAP logic for signals", ["session", "cumulative"], index=0)
session_vwap_include_premarket = st.sidebar.toggle("Session VWAP includes Premarket (starts 04:00)", value=False, help="OFF = RTH VWAP reset at 09:30 ET. ON = Extended VWAP starts 04:00 ET.")
show_dual_vwap = st.sidebar.toggle("Dual VWAP (show both lines)", value=True)

st.sidebar.markdown("### Engine complexity")
pro_mode = st.sidebar.toggle("Pro mode", value=True, help="Enables ICT-style diagnostics + extra scoring components.")
entry_model = st.sidebar.selectbox(
    "Entry model",
    ["Last price", "Midpoint (last closed bar)", "VWAP reclaim limit"],
    index=2,
    help="Controls how the app proposes an entry price when a setup is detected."
)

slip_mode = st.sidebar.selectbox(
    "Slippage buffer",
    ["Off", "Fixed cents", "ATR fraction"],
    index=1,
    help="Adds a small buffer to entry to be more realistic for fast/volatile names."
)
slip_fixed_cents = st.sidebar.slider("Fixed slippage (cents)", 0.0, 0.25, 0.02, 0.01)
slip_atr_frac = st.sidebar.slider("ATR fraction slippage", 0.0, 1.0, 0.15, 0.05)


st.sidebar.markdown("### Time-of-day filter (ET)")
allow_opening = st.sidebar.checkbox("Opening (09:30–11:00)", value=True)
allow_midday = st.sidebar.checkbox("Midday (11:00–15:00)", value=False)
allow_power = st.sidebar.checkbox("Power hour (15:00–16:00)", value=True)
allow_premarket = st.sidebar.checkbox("Premarket (04:00–09:30)", value=False)
allow_afterhours = st.sidebar.checkbox("Afterhours (16:00+)", value=False)
st.sidebar.markdown("#### Killzone presets")
killzone_preset = st.sidebar.selectbox(
    "Killzone preset",
    ["Custom (use toggles)", "Opening Drive", "Lunch Chop", "Power Hour", "Pre-market"],
    index=0,
    help="Quick presets that bias scoring + optionally constrain time windows."
)
liquidity_weighting = st.sidebar.slider(
    "Liquidity-weighted scoring (0–1)",
    0.0, 1.0, 0.55, 0.05,
    help="Boosts scoring during higher-liquidity windows (open/close) and de-emphasizes lunch chop."
)
orb_minutes = st.sidebar.slider(
    "ORB window (minutes)",
    5, 60, 15, 5,
    help="Opening Range Breakout window used to compute ORB high/low levels."
)


st.sidebar.markdown("### Higher‑TF bias overlay (optional)")
enable_htf = st.sidebar.toggle("Enable HTF bias", value=False)
htf_interval = st.sidebar.selectbox("HTF interval", ["15min", "30min"], index=0, disabled=not enable_htf)
htf_strict = st.sidebar.checkbox("Strict HTF alignment", value=False, disabled=not enable_htf)

st.sidebar.markdown("### ATR score normalization")
atr_norm_mode = st.sidebar.selectbox("ATR normalization", ["Auto (per ticker)", "Manual"], index=0, help="Auto uses each ticker's recent median ATR% as its baseline so high-vol names aren't punished.")
if atr_norm_mode == "Manual":
    target_atr_pct = st.sidebar.slider("Target ATR% (score normalization)", 0.001, 0.020, 0.004, 0.001, format="%.3f")
else:
    target_atr_pct = None

st.sidebar.markdown("### Fib logic")
show_fibs = st.sidebar.checkbox("Show Fibonacci retracement", value=True)
fib_lookback = st.sidebar.slider("Fib lookback bars", 60, 240, 120, 10) if show_fibs else 120

st.sidebar.markdown("### In‑App Alerts")
cooldown_minutes = st.sidebar.slider("Cooldown minutes (per ticker)", 1, 30, 7, 1)
alert_threshold = st.sidebar.slider("Alert score threshold", 60, 100, int(PRESETS[mode]["min_actionable_score"]), 1)

# Debug/product controls: makes it obvious when cooldown is suppressing email alerts.
col_cd1, col_cd2 = st.sidebar.columns(2)
with col_cd1:
    if st.button("Clear cooldowns", use_container_width=True):
        st.session_state.last_alert_ts = {}
        st.sidebar.success("Cooldowns cleared")
with col_cd2:
    if st.button("Clear signal state", use_container_width=True):
        st.session_state.symbol_state = {}
        st.session_state.pending_confirm = {}
        st.sidebar.success("Signal state cleared")
# Bias strictness tuning
st.sidebar.markdown("#### Bias strictness")
bias_strictness = st.sidebar.slider(
    "Bias strictness (looser ↔ stricter)",
    0.0, 1.0, 0.65, 0.05,
    help="Higher = fewer signals, stronger confirmation requirements."
)
split_long_short = st.sidebar.toggle(
    "Separate LONG vs SHORT thresholds",
    value=False,
    help="If enabled, you can require different score thresholds for LONG vs SHORT."
)
long_threshold = int(alert_threshold)
short_threshold = int(alert_threshold)
if split_long_short:
    long_threshold = st.sidebar.slider("LONG score threshold", 50, 99, int(alert_threshold), 1)
    short_threshold = st.sidebar.slider("SHORT score threshold", 50, 99, int(alert_threshold), 1)

capture_alerts = st.sidebar.checkbox("Capture alerts in-app", value=True)
max_alerts_kept = st.sidebar.slider("Max alerts kept", 10, 300, 60, 10)
smtp_server, smtp_port, smtp_user, smtp_password, to_email = load_email_secrets()
enable_email_alerts = st.sidebar.toggle(
    "Send email alerts",
    value=False,
    help="Sends alerts via Gmail SMTP (requires Secrets). You can keep in-app alerts ON too.",
)

# If the user turns email alerts ON mid-session, arm the system so already-actionable
# rows can trigger on the next scan (instead of requiring a fresh transition).
prev_email_enabled = bool(st.session_state.get("_email_enabled_prev", False))
if enable_email_alerts and not prev_email_enabled:
    # Reset per-symbol state so threshold/actionable crossings are re-evaluated.
    st.session_state.symbol_state = {}
    st.session_state.last_alert_ts = {}
st.session_state["_email_enabled_prev"] = bool(enable_email_alerts)

if enable_email_alerts:
    if not (smtp_user and smtp_password and to_email):
        st.sidebar.warning('Email is ON but Secrets are missing. Add [email] smtp_user/smtp_password/to_email in Streamlit Secrets.')
    else:
        st.sidebar.success(f'Email enabled → {to_email}')




st.sidebar.markdown("### API pacing / refresh")
# Premium entitlement: ensures intraday candles are real-time if your plan supports it.
env_ent = (os.getenv("ALPHAVANTAGE_ENTITLEMENT") or os.getenv("AV_ENTITLEMENT") or "").strip()
entitlement_ui = st.sidebar.selectbox(
    "Alpha Vantage entitlement",
    ["(auto)" if env_ent else "(none)", "realtime", "delayed"],
    index=0,
    help="Premium customers should use 'realtime' so intraday candles don't look like yesterday's feed. '(auto)' uses ALPHAVANTAGE_ENTITLEMENT env var if set."
)
min_between_calls = st.sidebar.slider("Seconds between API calls", 0.5, 8.0, 1.5, 0.5)
auto_refresh = st.sidebar.checkbox("Auto-refresh scanner", value=False)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 10, 180, 30, 5) if auto_refresh else None

st.sidebar.markdown("---")
st.sidebar.caption("Required env var: ALPHAVANTAGE_API_KEY")

symbols = [s.strip().upper() for s in watchlist_text.replace(",", "\n").splitlines() if s.strip()]
st.session_state.watchlist = symbols

st.title("Ztockly — Intraday Reversal Scalping Engine (v7)")
st.caption("Basic: VWAP + RSI‑5 event + MACD histogram turn + volume. Pro adds sweeps/OB/breaker/FVG/EMA. v7 adds fib‑anchored TPs, liquidity‑weighted scoring, ATR score normalization, and optional HTF bias.")

@st.cache_resource
def get_client(min_seconds_between_calls: float, entitlement_choice: str):
    client = AlphaVantageClient()
    client.cfg.min_seconds_between_calls = float(min_seconds_between_calls)
    # Allow UI override, otherwise rely on env var configured in AlphaVantageClient.
    try:
        if entitlement_choice == "realtime":
            client.cfg.entitlement = "realtime"
        elif entitlement_choice == "delayed":
            client.cfg.entitlement = "delayed"
        else:
            # (auto)/(none): keep whatever av_client initialized
            pass
    except Exception:
        pass
    return client

client = get_client(min_between_calls, entitlement_ui)

def _now_label() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def can_alert(symbol: str, now_ts: float, cooldown_min: int) -> bool:
    last = st.session_state.last_alert_ts.get(symbol)
    if last is None:
        return True
    return (now_ts - float(last)) >= cooldown_min * 60.0

def add_in_app_alert(row: dict) -> None:
    alert = {
        "ts_unix": time.time(),
        "time": _now_label(),
        "symbol": row["Symbol"],
        "bias": row["Bias"],
        "score": int(row["Score"]),
        "session": row.get("Session"),
        "last": row.get("Last"),
        "entry": row.get("Entry"),
        "stop": row.get("Stop"),
        "t1": row.get("TP1"),
        "t2": row.get("TP2"),
        "why": row.get("Why"),
        "as_of": row.get("AsOf"),
        "mode": mode,
        "interval": interval,
        "pro_mode": pro_mode,
        "extras": row.get("Extras", {}),
    }
    st.session_state.alerts.insert(0, alert)
    st.session_state.alerts = st.session_state.alerts[: int(max_alerts_kept)]

def render_alerts_panel():
    st.subheader("🚨 Live Alerts")
    left, right = st.columns([2, 1])

    with right:
        st.metric("Alerts stored", len(st.session_state.alerts))
        if st.button("Clear alerts", type="secondary"):
            st.session_state.alerts = []
            st.session_state.last_alert_ts = {}
            st.rerun()
        st.markdown("**Filters**")
        f_bias = st.multiselect("Bias", ["LONG", "SHORT"], default=["LONG", "SHORT"])
        min_score = st.slider("Min score", 0, 100, 80, 1)

    with left:
        alerts = [a for a in st.session_state.alerts if a["bias"] in f_bias and a["score"] >= min_score]
        if not alerts:
            st.info("No alerts yet. Turn on auto-refresh + capture alerts, then let it scan.")
            return

        for a in alerts[:30]:
            badge = "🟢" if a["bias"] == "LONG" else "🔴"
            pro_badge = "⚡ Pro" if a.get("pro_mode") else "🧱 Basic"
            title = f"{badge} **{a['symbol']}** — **{a['bias']}** — Score **{a['score']}** ({a.get('session','')}) • {pro_badge}"
            with st.container(border=True):
                st.markdown(title)
                cols = st.columns(6)
                cols[0].metric("Last", f"{a['last']:.4f}" if a["last"] is not None else "N/A")
                cols[1].metric("Entry", f"{a['entry']:.4f}" if a["entry"] is not None else "—")
                cols[2].metric("Stop", f"{a['stop']:.4f}" if a["stop"] is not None else "—")
                cols[3].metric("1R", f"{a['t1']:.4f}" if a["t1"] is not None else "—")
                cols[4].metric("2R", f"{a['t2']:.4f}" if a["t2"] is not None else "—")
                fib_tp1 = (a.get("extras") or {}).get("fib_tp1")
                cols[5].metric("Fib TP1", f"{fib_tp1:.4f}" if isinstance(fib_tp1, (float,int)) else "—")
                st.caption(f"{a['time']} • interval={a['interval']} • mode={a['mode']} • VWAP={a.get('extras',{}).get('vwap_logic')} • liquidity={a.get('extras',{}).get('liquidity_phase')} • as_of={a.get('as_of')}")
                st.write(a.get("why") or "")

                ex = a.get("extras") or {}
                chips = []
                if ex.get("bull_liquidity_sweep"): chips.append("Liquidity sweep (low)")
                if ex.get("bear_liquidity_sweep"): chips.append("Liquidity sweep (high)")
                if ex.get("bull_ob_retest"): chips.append("Bull OB retest")
                if ex.get("bear_ob_retest"): chips.append("Bear OB retest")
                if ex.get("bull_breaker_retest"): chips.append("Bull breaker retest")
                if ex.get("bear_breaker_retest"): chips.append("Bear breaker retest")
                if ex.get("fib_near_long") or ex.get("fib_near_short"): chips.append("Near Fib")
                if ex.get("htf_bias_value") in ("BULL","BEAR"): chips.append(f"HTF {ex.get('htf_bias_value')}")
                if chips:
                    st.markdown("**Chips:** " + " • ".join([f"`{c}`" for c in chips]))
                with st.expander("Raw payload"):
                    st.json(a)

tab_scan, tab_alerts = st.tabs(["📡 Scanner", "🚨 Alerts"])
with tab_alerts:
    render_alerts_panel()

with tab_scan:
    col_a, col_b, col_c, col_d = st.columns([1, 1, 2, 1])
    with col_a:
        scan_now = st.button("Scan Watchlist", type="primary")
    with col_b:
        if st.button("Capture test alert", use_container_width=True):
            test = {
                "Symbol": "TEST",
                "Bias": "LONG",
                "Score": 95,
                "Session": "TEST",
                "Last": 100.00,
                "Entry": 100.00,
                "Stop": 99.50,
                "TP1": 100.50,
                "TP2": 101.00,
                "Why": "Test alert (wiring check).",
                "AsOf": pd.Timestamp.utcnow().isoformat(),
            }

            if capture_alerts:
                add_in_app_alert(test)
                st.success("Test alert captured in-app.")
            else:
                st.info("In-app capture is OFF; test alert not stored.")

            if enable_email_alerts:
                ok, err = send_email_safe(test, smtp_server, smtp_port, smtp_user, smtp_password, to_email)
                if ok:
                    st.success(f"Test email sent to {to_email}.")
                else:
                    st.error(f"Test email failed: {err}")
    with col_c:
        st.write("Tip: Keep watchlist small (5–15) to stay within API limits.")
    with col_d:
        st.write(f"Now: {_now_label()}")

    def run_scan():
        if not symbols:
            st.warning("Add at least one ticker to your watchlist.")
            return []
        with st.spinner("Scanning watchlist..."):
            return scan_watchlist(
                client, symbols,
                interval=interval,
                mode=mode,
                pro_mode=pro_mode,
                allow_opening=allow_opening,
                allow_midday=allow_midday,
                allow_power=allow_power,
                allow_premarket=allow_premarket,
                allow_afterhours=allow_afterhours,
                vwap_logic=vwap_logic,
                session_vwap_include_premarket=session_vwap_include_premarket,
                fib_lookback_bars=fib_lookback,
                enable_htf_bias=enable_htf,
                htf_interval=htf_interval,
                htf_strict=htf_strict,
                target_atr_pct=target_atr_pct,
                use_last_closed_only=use_last_closed_only,
                bar_closed_guard=bar_closed_guard,
                killzone_preset=killzone_preset,
                liquidity_weighting=liquidity_weighting,
                orb_minutes=orb_minutes,
                entry_model=entry_model,
                slippage_mode=slip_mode,
                fixed_slippage_cents=slip_fixed_cents,
                atr_fraction_slippage=slip_atr_frac,
            )


    results = []
    if auto_refresh:
        results = run_scan()
        st.session_state['last_results'] = results
        st.session_state['last_scan_ts'] = time.time()
        st.info(f"Auto-refresh is ON — rerunning every ~{refresh_seconds}s.")
    else:
        if scan_now:
            results = run_scan()

    if results:
        # Build ranked table
        df = pd.DataFrame([{
            "Symbol": r.symbol,
            "Bias": r.bias,
            "Actionable": (r.bias in ["LONG", "SHORT"]),
            # Product rule: never hide the score.
            # Actionability is expressed by Bias/Actionable + Entry/Stop/TP.
            "Score": int(r.setup_score),
            "Potential": int(r.setup_score),
            "Session": r.session,
            "Last": r.last_price,
            "Entry": r.entry,
            "Stop": r.stop,
            "TP1": r.target_1r,
            "TP2": r.target_2r,
            "ATR%": (r.extras or {}).get("atr_pct"),
            "ATR baseline%": (r.extras or {}).get("atr_ref_pct"),
            "Score scale": (r.extras or {}).get("atr_score_scale"),
            "Why": r.reason,
            # Show the candle timestamp (ET) and help diagnose stale feeds.
            "AsOf": str(r.timestamp) if r.timestamp is not None else None,
            "Extras": r.extras,
        } for r in results])

        # Data freshness diagnostics (helps catch stale intraday feeds).
        def _age_minutes(ts):
            try:
                t = pd.to_datetime(ts)
                # treat naive as ET (Alpha Vantage timestamps are US/Eastern strings)
                if t.tzinfo is None:
                    t = t.tz_localize("America/New_York")
                else:
                    t = t.tz_convert("America/New_York")
                now_et = pd.Timestamp.now(tz="America/New_York")
                return float((now_et - t).total_seconds() / 60.0)
            except Exception:
                return None

        df["Data age (min)"] = df["AsOf"].map(_age_minutes)

        try:
            oldest = df["Data age (min)"].dropna().max()
        except Exception:
            oldest = None
        if isinstance(oldest, (float, int)) and oldest >= 30:
            st.warning(
                f"Heads up: intraday feed looks stale (oldest AsOf is ~{oldest:.0f} min ago). "
                "This can happen with free/delayed APIs, rate limits, or extended-hours gaps. "
                "Scores may still rank, but actionability/alerts can be misleading until data refreshes."
            )

        # Styling: color scale column + per-row tooltip explaining normalization
        df_view = df.drop(columns=["Extras"]).copy()

        def _scale_tooltip(row):
            atrp = row.get("ATR%")
            basep = row.get("ATR baseline%")
            sc = row.get("Score scale")
            if isinstance(atrp, (float, int)) and isinstance(basep, (float, int)) and isinstance(sc, (float, int)):
                return (
                    f"Score normalized because ATR% differs from baseline. "
                    f"Current ATR%={atrp:.3f}, Baseline={basep:.3f}. "
                    f"Scale={sc:.2f} (clipped to 0.75–1.25)."
                )
            return "No ATR normalization data."

        # Add human-readable sanity check columns (Streamlit Cloud safe)
        df_view["Scale note"] = df_view.apply(_scale_tooltip, axis=1)

        def _flag(sc):
            try:
                x = float(sc)
            except Exception:
                return ""
            if x < 0.90:
                return "🔻 scaled down"
            if x > 1.10:
                return "🔺 scaled up"
            return "•"

        df_view["Scale flag"] = df_view["Score scale"].map(_flag)

        st.subheader("Ranked Setups")
        st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "Potential": st.column_config.NumberColumn("Potential", format="%d", help="Unblocked scoring potential before hard requirements are satisfied."),
                "Actionable": st.column_config.CheckboxColumn("Actionable", help="True only when the engine can produce Entry/Stop/TP (bias LONG/SHORT)."),
                "Data age (min)": st.column_config.NumberColumn("Data age (min)", format="%.0f", help="How old the most recent candle is (ET). Large values usually mean delayed/stale feed."),
                "ATR%": st.column_config.NumberColumn("ATR%", format="%.3f"),
                "ATR baseline%": st.column_config.NumberColumn("ATR baseline%", format="%.3f"),
                "Score scale": st.column_config.NumberColumn(
                    "Score scale",
                    format="%.2f",
                    help="ATR normalization scale. <0.90 means more volatile than baseline (score scaled down). >1.10 means less volatile than baseline (score scaled up).",
                ),
                "Scale flag": st.column_config.TextColumn("Scale", help="Quick visual: scaled up/down based on ATR normalization."),
                "Scale note": st.column_config.TextColumn("Scale note", help="Why this ticker was scaled (sanity-check ATR normalization)."),
            },
        )

        top = next((x for x in results if x.bias in ["LONG", "SHORT"]), results[0])
        pro_badge = "⚡ Pro" if pro_mode else "🧱 Basic"
        st.success(f"Top setup: **{top.symbol}** — **{top.bias}** (Score {top.setup_score}, {top.session}) • {pro_badge}")

        if top.bias == "NEUTRAL":
            st.info("Top-ranked row is **non-actionable** right now (hard requirement not met), so no Entry/TP and no alert will fire. Check the *Why* column for the blocker.")

        now = time.time()
        suppressed_by_cooldown = []

        for r in results:
            actionable = (r.bias in ["LONG", "SHORT"])
            score_now = float(r.setup_score or 0)
            prev = st.session_state.symbol_state.get(r.symbol)

            # Persist state for next rerun (we track even NEUTRAL rows so we can detect threshold crossings).
            st.session_state.symbol_state[r.symbol] = {
                "bias": r.bias,
                "actionable": actionable,
                "score": score_now,
            }

            # Alert eligibility: actionable + above threshold.
            if not (actionable and score_now >= alert_threshold):
                continue

            # Fire alerts when the setup:
            #  - becomes actionable (NEUTRAL -> LONG/SHORT)
            #  - flips direction (LONG <-> SHORT)
            #  - crosses the score threshold (below -> above)
            prev_actionable = bool(prev.get("actionable")) if prev else False
            prev_bias = prev.get("bias") if prev else None
            prev_score = float(prev.get("score") or 0) if prev else 0.0

            crossed_threshold = (prev is None) or (prev_score < alert_threshold)
            became_actionable = (prev is None) or (not prev_actionable)
            flipped_direction = (prev is not None) and (prev_bias in ["LONG", "SHORT"]) and (prev_bias != r.bias)

            if not (crossed_threshold or became_actionable or flipped_direction):
                continue

            if not can_alert(r.symbol, now, cooldown_minutes):
                suppressed_by_cooldown.append(r.symbol)
                continue

            row = df.loc[df['Symbol'] == r.symbol].iloc[0].to_dict()

            # In-app capture
            if capture_alerts:
                add_in_app_alert(row)

            # Email delivery
            if enable_email_alerts:
                ok, err = send_email_safe(row, smtp_server, smtp_port, smtp_user, smtp_password, to_email)
                if not ok:
                    st.warning(f"Email alert failed for {r.symbol}: {err}")

            st.session_state.last_alert_ts[r.symbol] = now

        if suppressed_by_cooldown:
            st.info(
                "Alert cooldown suppressed: "
                + ", ".join(sorted(set(suppressed_by_cooldown)))
                + ". Use **Clear cooldowns** in the sidebar if you want to force re-alerts."
            )


        st.subheader("Chart & Signal Detail")
        pick = st.selectbox("Select ticker", [r.symbol for r in results], index=0)

        with st.spinner(f"Loading chart data for {pick}..."):
            ohlcv, rsi5, rsi14, macd_hist, quote = fetch_bundle(client, pick, interval=interval)

        sig = compute_scalp_signal(
            pick, ohlcv, rsi5, rsi14, macd_hist,
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
            fib_lookback_bars=fib_lookback,
            target_atr_pct=target_atr_pct,
            killzone_preset=killzone_preset,
            liquidity_weighting=liquidity_weighting,
            orb_minutes=orb_minutes,
            entry_model=entry_model,
            slippage_mode=slip_mode,
            fixed_slippage_cents=slip_fixed_cents,
            atr_fraction_slippage=slip_atr_frac,
        )
        plot_df = ohlcv.sort_index().copy().tail(260)
        plot_df["vwap_cum"] = calc_vwap(plot_df)
        plot_df["vwap_sess"] = calc_session_vwap(plot_df, include_premarket=session_vwap_include_premarket)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"], name="Price"))

        if show_dual_vwap:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap_sess"], mode="lines", name="VWAP (Session)"))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap_cum"], mode="lines", name="VWAP (Cumulative)"))
        else:
            key = "vwap_sess" if vwap_logic == "session" else "vwap_cum"
            nm = "VWAP (Session)" if vwap_logic == "session" else "VWAP (Cumulative)"
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[key], mode="lines", name=nm))

        # Fib lines (visual)
        if show_fibs:
            seg = plot_df.tail(int(min(fib_lookback, len(plot_df))))
            hi = float(seg["high"].max())
            lo = float(seg["low"].min())
            if hi > lo:
                for name, level in [("Fib 0.382", hi - 0.382*(hi-lo)), ("Fib 0.5", hi - 0.5*(hi-lo)), ("Fib 0.618", hi - 0.618*(hi-lo)), ("Fib 0.786", hi - 0.786*(hi-lo))]:
                    fig.add_hline(y=level, line_dash="dot", annotation_text=name, annotation_position="top left")

        # Entry/Stop/Targets
        if sig.entry and sig.stop:
            fig.add_hline(y=sig.entry, line_dash="dot", annotation_text="Entry", annotation_position="top left")
            fig.add_hline(y=sig.stop, line_dash="dash", annotation_text="Stop", annotation_position="bottom left")
        if sig.target_1r:
            fig.add_hline(y=sig.target_1r, line_dash="dot", annotation_text="1R", annotation_position="top right")
        if sig.target_2r:
            fig.add_hline(y=sig.target_2r, line_dash="dot", annotation_text="2R", annotation_position="top right")
        fib_tp1 = (sig.extras or {}).get("fib_tp1")
        fib_tp2 = (sig.extras or {}).get("fib_tp2")
        if isinstance(fib_tp1, (float, int)):
            fig.add_hline(y=float(fib_tp1), line_dash="dash", annotation_text="Fib TP1", annotation_position="top right")
        if isinstance(fib_tp2, (float, int)):
            fig.add_hline(y=float(fib_tp2), line_dash="dash", annotation_text="Fib TP2", annotation_position="top right")

        fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1: st.metric("Bias", sig.bias)
        with c2: st.metric("Score", sig.setup_score)
        with c3: st.metric("Session", sig.session)
        with c4: st.metric("Liquidity", (sig.extras or {}).get("liquidity_phase", ""))
        with c5:
            lp = quote if quote is not None else sig.last_price
            st.metric("Last", f"{lp:.4f}" if lp is not None else "N/A")
        with c6:
            atrp = (sig.extras or {}).get("atr_pct")
            basep = (sig.extras or {}).get("atr_ref_pct")
            st.metric("ATR% / Base", f"{atrp:.3f} / {basep:.3f}" if isinstance(atrp, (float,int)) and isinstance(basep, (float,int)) else "N/A")
        with c7:
            sc = (sig.extras or {}).get("atr_score_scale")
            st.metric("Score scale", f"{sc:.2f}" if isinstance(sc, (float,int)) else "N/A")

        st.write("**Reasoning:**", sig.reason)

        st.markdown("### Trade Plan")
        if sig.bias in ["LONG", "SHORT"] and sig.entry and sig.stop:
            st.write(f"- **Entry:** {sig.entry:.4f}")
            st.write(f"- **Stop:** {sig.stop:.4f}")
            st.write(f"- **Targets (R):** 1R={sig.target_1r:.4f} • 2R={sig.target_2r:.4f}")
            if isinstance(fib_tp1, (float,int)) or isinstance(fib_tp2, (float,int)):
                st.write(f"- **Fib partials:** TP1={fib_tp1 if fib_tp1 is not None else '—'} • TP2={fib_tp2 if fib_tp2 is not None else '—'}")
            st.write("- **Fail-safe exit:** if price loses VWAP and MACD histogram turns against you, flatten remainder.")
            st.warning("Analytics tool only — always position-size and respect stops.")
        else:
            st.info("No clean confluence signal right now (or time-of-day filter blocking).")

        with st.expander("Diagnostics"):
            st.json(sig.extras)

    else:
        st.info("Add your watchlist in the sidebar, then click **Scan Watchlist** or enable auto-refresh.")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
