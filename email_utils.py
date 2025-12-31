import smtplib
from email.message import EmailMessage
from typing import Optional, Dict, Any

def send_email_alert(
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    """Send a simple plaintext email via SMTP (Gmail app-password compatible)."""
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def format_alert_email(payload: Dict[str, Any]) -> str:
    """Create a human-readable email body from an alert payload dict."""
    lines = []
    lines.append(f"Time: {payload.get('time')}")
    lines.append(f"Symbol: {payload.get('symbol')}")
    lines.append(f"Bias: {payload.get('bias')}   Score: {payload.get('score')}   Session: {payload.get('session')}")
    lines.append("")
    lines.append(f"Last: {payload.get('last')}")
    lines.append(f"Entry: {payload.get('entry')}")
    lines.append(f"Stop: {payload.get('stop')}")
    lines.append(f"TP1: {payload.get('tp1')}")
    lines.append(f"TP2: {payload.get('tp2')}")
    lines.append("")
    why = payload.get("why") or ""
    lines.append("Why:")
    lines.append(str(why))
    lines.append("")
    extras = payload.get("extras") or {}
    if extras:
        lines.append("Diagnostics:")
        for k in ["vwap_logic","session_vwap_include_premarket","atr_pct","baseline_atr_pct","atr_ref_pct","atr_score_scale","htf_bias","liquidity_phase"]:
            if k in extras:
                lines.append(f"- {k}: {extras.get(k)}")
    return "\n".join(lines)
