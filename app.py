"""Painel Streamlit para visualizar os sinais produzidos por rsi_divergence_bot.py."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

st.set_page_config(page_title="Crypto RSI Divergences", layout="wide")
st.title("⚡ Crypto RSI Divergences")
st.caption("O scanner é executado pelo serviço Background Worker do Render. Este painel lê os resultados do último scan.")


def read_latest():
    try:
        if DATABASE_URL:
            import psycopg
            with psycopg.connect(DATABASE_URL, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('public.rsi_divergence_store')")
                    exists = cur.fetchone()[0]
                    if not exists:
                        return None
                    cur.execute("SELECT value FROM rsi_divergence_store WHERE key = %s", ("latest_scan",))
                    row = cur.fetchone()
                    return row[0] if row else None
        if not LATEST_FILE.exists():
            return None
        with LATEST_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Não foi possível ler os resultados do scanner: {exc}")
        return None


if st.button("🔄 Atualizar painel"):
    st.rerun()

payload = read_latest()
if payload is None:
    st.warning("Ainda não existem resultados. Confirma se o Background Worker está ativo e executa `python rsi_divergence_bot.py --once` para testar.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Divergências na última varredura", int(payload.get("signals_count", 0)))
col2.metric("Pares analisados", int(payload.get("symbols_scanned", 0)))
col3.metric("Pedidos OHLCV OK", int(payload.get("ohlcv_requests_ok", 0)))
col4.metric("Erros de API/dados", int(payload.get("errors", 0)))

st.caption(f"Último scan concluído: {payload.get('scan_completed_at', 'desconhecido')}")
st.caption("Timeframes: " + ", ".join(payload.get("timeframes", [])))

signals = payload.get("signals", [])
if not signals:
    st.info("Não foram confirmadas divergências recentes neste scan. Isto não significa necessariamente um erro: os pivots têm de estar confirmados.")
    st.stop()

# Agrupar sem eliminar sinais de outros timeframes para a mesma moeda.
for signal in signals:
    bullish = signal.get("tipo") == "bullish"
    colour = "#1b8f4d" if bullish else "#b83232"
    arrow = "🟢" if bullish else "🔻"
    st.markdown(
        f"""
        <div style="background:#1e1e24;border-radius:10px;padding:14px 16px;margin:0 0 12px 0;border-left:5px solid {colour};">
          <div style="font-size:18px;font-weight:700;">{arrow} {signal.get('symbol', '?')} — {signal.get('timeframe', '?')} — RSI {signal.get('tipo', '').upper()}</div>
          <div style="color:#c7c7d1;margin-top:4px;">Pivot: {signal.get('signal_time', '?')} | Idade: {signal.get('idade_velas', '?')} vela(s)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
    a.metric("Entrada indicativa", f"{float(signal.get('entry', 0)):.8g}")
    b.metric("Stop indicativo", f"{float(signal.get('sl', 0)):.8g}")
    c.metric("Target indicativo (2R)", f"{float(signal.get('tp', 0)):.8g}")
    d.metric("RSI: anterior → atual", f"{signal.get('rsi_anterior', '?')} → {signal.get('rsi_atual', '?')}")
    st.caption(
        f"Preço no pivot: {float(signal.get('preco_pivot_anterior', 0)):.8g} → "
        f"{float(signal.get('preco_pivot_atual', 0)):.8g} | Pivot anterior: {signal.get('previous_pivot_time', '?')}"
    )

st.divider()
st.caption("Os sinais são padrões técnicos sujeitos a falsos positivos; entrada, stop e alvo são indicativos e não constituem aconselhamento financeiro.")
