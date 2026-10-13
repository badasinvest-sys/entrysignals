"""Painel do scanner gratuito. Partilha latest_signals.json com scanner local."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))

st.set_page_config(page_title="RSI + Rompimentos | Scanner", page_icon="📈", layout="wide")
st.title("📈 RSI Divergências & Rompimentos de Resistência")
st.caption("Apenas MEXC/Binance Spot USDT. Um único serviço Render, sem base de dados. Atualização do painel a cada 30 segundos.")


def _latest() -> dict | None:
    try:
        if not LATEST_FILE.exists():
            return None
        return json.loads(LATEST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(f"Falha na leitura da última análise: {exc}")
        return None


def _number(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@st.fragment(run_every="30s")
def mostrar():
    data = _latest()
    if data is None:
        st.info("A aguardar a primeira varredura automática. Consulta os logs do Render se a espera for prolongada.")
        return

    a, b, c, d, e = st.columns(5)
    a.metric("Divergências RSI", data.get("rsi_count", 0))
    b.metric("Possíveis setups", data.get("breakout_possible_count", 0))
    c.metric("Entradas em reteste", data.get("breakout_entry_count", 0))
    d.metric("Pares analisados", data.get("symbols_scanned", 0))
    e.metric("Erros", data.get("errors", 0))
    st.caption(
        f"Exchange: {data.get('exchange', '?')} | Último scan: {data.get('scan_completed_at', '?')} "
        f"| Duração: {data.get('scan_seconds', '?')} segundos | OHLCV OK: {data.get('ohlcv_requests_ok', 0)}"
    )
    st.caption("Timeframes realmente analisados: " + ", ".join(data.get("timeframes", [])))

    with st.expander("Regras em vigor / Configuração"):
        st.markdown(
            "**RSI:** ≥3 pivots em todos os timeframes; 2 pivots apenas em 4h e 1d. "
            "**Rompimentos:** apenas em 4h e 1d, nível historicamente testado "
            "com evidência de volume alto, rompimento com volume, sustentação e reteste. "
            "O perfil de volume é estimado com OHLCV e não equivale ao volume real por preço."
        )
        st.json(data.get("configuration", {}))

    options = ("Todos", "Divergências RSI", "Possível setup", "Entrada")
    kind = st.radio("Filtrar sinais", options, horizontal=True)
    signals = data.get("signals", [])
    if kind == "Divergências RSI":
        signals = [s for s in signals if s.get("categoria") == "rsi"]
    elif kind in ("Possível setup", "Entrada"):
        signals = [s for s in signals if s.get("fase") == kind]
    if not signals:
        st.info("Nenhum sinal recente corresponde ao filtro. Os padrões são avaliados com as velas fechadas, exceto o reteste intrabar opcional.")
        return

    for s in signals:
        symbol = s.get("symbol", "?")
        tf = s.get("timeframe", "?")
        when = s.get("signal_time", "?")
        if s.get("categoria") == "breakout":
            phase = s.get("fase", "?")
            emoji = "🚀" if phase == "Possível setup" else "🎯"
            with st.container(border=True):
                st.markdown(f"#### {emoji} {phase} — {symbol} · {tf}")
                st.caption(
                    f"{when} | Breakout: {s.get('breakout_time', '?')} | "
                    f"{'Reteste de vela ABERTA (pode invalidar)' if s.get('intrabar') else 'Velas confirmadas'}"
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Resistência", f"{_number(s.get('nivel')):.8g}")
                c2.metric("Zona mínima", f"{_number(s.get('zona_min')):.8g}")
                c3.metric("Zona máxima", f"{_number(s.get('zona_max')):.8g}")
                c4.metric("Preço observado", f"{_number(s.get('preco_atual')):.8g}")
                st.markdown(
                    f"**Testes prévios:** {s.get('toques_resistencia', '?')} · "
                    f"**Volume da zona:** {_number(s.get('volume_zona_ratio')):.2f}× · "
                    f"**Volume do rompimento:** {_number(s.get('volume_rompimento_ratio')):.2f}× · "
                    f"**Sustentação:** {s.get('velas_sustentacao', '?')} velas fechadas"
                )
                if phase == "Possível setup":
                    st.info("Nível rompido e sustentado. Aguardar preço regressar à zona; ainda não é alerta de entrada.")
                elif s.get("intrabar"):
                    st.warning("Reteste detetado com vela ainda aberta. Se a vela quebrar a zona, o padrão poderá ser invalidado.")
                else:
                    st.success("Preço regressou à zona e recuperou o nível no fecho da vela de reteste.")
        else:
            icon = "🟢" if s.get("tipo") == "bullish" else "🔻"
            with st.container(border=True):
                st.markdown(f"#### {icon} RSI {s.get('tipo', '?').upper()} — {symbol} · {tf} · {s.get('pivots_count', 0)} pivots")
                st.caption(f"Pivô confirmado: {when} | Idade: {s.get('idade_velas', '?')} vela(s)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preço inicial", f"{_number(s.get('preco_inicial')):.8g}")
                c2.metric("Preço final", f"{_number(s.get('preco_final')):.8g}")
                c3.metric("RSI inicial", f"{_number(s.get('rsi_inicial')):.2f}")
                c4.metric("RSI final", f"{_number(s.get('rsi_final')):.2f}")
                if s.get("pivots"):
                    st.dataframe(pd.DataFrame([
                        {"N.º": i, "Data pivot preço": p.get("price_time"),
                         "Preço": p.get("price"), "RSI": p.get("rsi"),
                         "Data pivot RSI": p.get("rsi_time")}
                        for i, p in enumerate(s["pivots"], start=1)
                    ]), hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Ferramenta experimental de análise de mercado. Alertas são padrões técnicos, não ordens de compra. "
        "O ficheiro local de sinais/alertas não é persistente após reinício ou redeploy do Render gratuito."
    )


mostrar()
