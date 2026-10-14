"""Painel local Streamlit — sinais, trades simuladas e respetivos gráficos."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))
TRADES_FILE = Path(os.getenv("TRADES_FILE", str(BASE_DIR / "trades_history.json")))
CHARTS_DIR = Path(os.getenv("CHARTS_DIR", str(BASE_DIR / "trade_charts")))

st.set_page_config(page_title="Scanner RSI + Breakouts", page_icon="📈", layout="wide")
st.title("📈 Scanner RSI + Breakouts")
st.caption("Divergências de preço/RSI seletivas · sweep/rejeição · confirmação opcional por Open Interest · breakout/reteste.")


def read_json(path: Path):
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Erro ao ler {path.name}: {exc}")
        return None


def render_trade_row(trade: dict) -> None:
    symbol = str(trade.get("symbol", "?"))
    tf = str(trade.get("timeframe", "?"))
    status = str(trade.get("status", "Em curso"))
    icon = {"Ganha": "✅", "Perdida": "❌", "Em curso": "🕒"}.get(status, "📌")
    with st.container(border=True):
        c_img, c_details = st.columns([1, 2.35], gap="medium")
        with c_img:
            # O ficheiro é um nome gerado por SHA-256, nunca um caminho arbitrário recebido da API.
            file_name = str(trade.get("imagem") or "")
            if file_name.startswith("trade_") and file_name.endswith(".png") and "/" not in file_name and "\\" not in file_name:
                path = CHARTS_DIR / file_name
                if path.is_file():
                    st.image(str(path), use_container_width=True)
                else:
                    st.caption("Gráfico indisponível: histórico local apagado no reinício ou à espera de atualização.")
            else:
                st.caption("Gráfico ainda não gerado.")
        with c_details:
            st.markdown(f"**{icon} {symbol} · {tf} · {status}**")
            st.caption(f"Abertura: {trade.get('opened_at', '?')} · Fecho: {trade.get('closed_at') or '—'}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Entrada", f"{float(trade.get('entrada', 0)):.8g}")
            c2.metric("Stop-loss", f"{float(trade.get('sl', 0)):.8g}")
            c3.metric("Take-profit", f"{float(trade.get('tp', 0)):.8g}")
            r = trade.get("resultado_r")
            st.caption(f"RR planeado: 2:1 · Resultado: {float(r):+.1f}R" if r is not None else "RR planeado: 2:1 · Resultado: Em curso")
            if trade.get("notas"):
                st.caption(str(trade["notas"]))


@st.fragment(run_every="30s")
def render() -> None:
    payload = read_json(LATEST_FILE)
    trades_data = read_json(TRADES_FILE) or {"trades": []}
    trades = trades_data.get("trades", []) if isinstance(trades_data, dict) else []
    if payload is None:
        st.info("O scanner ainda não concluiu a primeira varredura. Confirma os logs do Render.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sinais recentes", int(payload.get("signals_count", 0)))
    c2.metric("Pares verificados", int(payload.get("symbols_scanned", 0)))
    c3.metric("Pedidos OHLCV OK", int(payload.get("ohlcv_requests_ok", 0)))
    c4.metric("Erros de API", int(payload.get("errors", 0)))
    st.caption(f"{payload.get('exchange', '?')} · Scan: {payload.get('scan_completed_at', '?')} · Duração: {payload.get('scan_seconds', '?')}s · Timeframes: {', '.join(payload.get('timeframes', []))}")

    tab1, tab2, tab3 = st.tabs(["Sinais de mercado", "Trades e win rate", "Critérios utilizados"])
    with tab1:
        signals = payload.get("signals", [])
        if not signals:
            st.info("Sem padrões válidos no último scan. O filtro privilegia qualidade em vez de quantidade.")
        for sig in signals:
            with st.container(border=True):
                if sig.get("categoria") == "rsi_divergence":
                    icon = "🔻" if sig.get("tipo") == "bearish" else "🟢"
                    st.subheader(f"{icon} {sig.get('symbol')} · {sig.get('timeframe')} · RSI {str(sig.get('tipo', '')).upper()}")
                    st.caption(f"{sig.get('pivots_count')} pivots estruturais · {sig.get('rejection_count', 0)} rejeições · {sig.get('sweep_count', 0)} sweeps · {sig.get('signal_time')}")
                    oi = sig.get("oi") or {}
                    status = oi.get("status", "Indisponível")
                    oi_value = f"{oi.get('variacao_pct'):+.1f}%" if oi.get("variacao_pct") is not None else "Sem dados"
                    x1, x2, x3 = st.columns(3)
                    x1.metric("Padrão", sig.get("descricao_padrao", ""))
                    x2.metric("Variação RSI", f"{sig.get('variacao_rsi', 0):+.1f}")
                    x3.metric(f"OI: {status}", oi_value)
                    if oi.get("fonte"):
                        st.caption(f"OI de perpétuos: {oi['fonte']}. Não é OI spot.")
                    pivots = sig.get("pivots") or []
                    if pivots:
                        st.dataframe(pd.DataFrame([{
                            "Pivot": i,
                            "Momento": p.get("price_time"),
                            "Preço": p.get("price"),
                            "RSI": p.get("rsi"),
                            "Rejeição": "Sim" if p.get("rejection") else "Não",
                            "Liquidity sweep": "Sim" if p.get("sweep") else "Não",
                        } for i, p in enumerate(pivots, start=1)]), hide_index=True, use_container_width=True)
                elif sig.get("categoria") == "breakout":
                    icon = "🚀" if sig.get("fase") == "Possível setup" else "🎯"
                    st.subheader(f"{icon} {sig.get('symbol')} · {sig.get('timeframe')} · {sig.get('fase')}")
                    st.caption(f"Zona {sig.get('zona_min', 0):.8g}–{sig.get('zona_max', 0):.8g} · {sig.get('signal_time')}")
                    a, b, c = st.columns(3)
                    a.metric("Resistência", f"{sig.get('nivel', 0):.8g}")
                    b.metric("Toques anteriores", sig.get("toques_resistencia", "?"))
                    c.metric("Volume relativo", f"{sig.get('volume_rompimento_ratio', 0)}x")
                    if sig.get("trade"):
                        t = sig["trade"]
                        st.markdown(f"**Entrada {t['entry']:.8g} · SL {t['sl']:.8g} · TP {t['tp']:.8g} · RR 2:1**")
    with tab2:
        st.caption("Resultados de trades **simuladas**, avaliadas por candles fechadas. Não são ordens executadas; não incluem comissões, slippage ou liquidação. Quando SL e TP coincidem na mesma vela, prevalece SL.")
        if not trades:
            st.info("Ainda não existem alertas de entrada registados.")
        else:
            closed = [t for t in trades if t.get("status") in ("Ganha", "Perdida")]
            wins = sum(t["status"] == "Ganha" for t in closed)
            active = sum(t.get("status") == "Em curso" for t in trades)
            closed_count = len(closed)
            total_r = sum(float(t.get("resultado_r") or 0) for t in closed)
            cols = st.columns(5)
            cols[0].metric("Fechadas", closed_count)
            cols[1].metric("Win rate", f"{100.0 * wins / closed_count:.1f}%" if closed_count else "—")
            cols[2].metric("Ganhas", wins)
            cols[3].metric("Em curso", active)
            cols[4].metric("Resultado acumulado", f"{total_r:+.1f}R")
            status_filter = st.selectbox("Estado das trades", ["Todas", "Em curso", "Ganha", "Perdida"], index=0)
            filtered = [t for t in trades if status_filter == "Todas" or t.get("status") == status_filter]
            filtered.sort(key=lambda t: int(t.get("opened_at_ms", 0)), reverse=True)
            per_page = 12
            page_count = max(1, (len(filtered) + per_page - 1) // per_page)
            page = st.selectbox("Página do histórico", options=list(range(1, page_count + 1)), index=0)
            begin = (page - 1) * per_page
            showing = filtered[begin: begin + per_page]
            st.caption(f"A mostrar trades {begin + 1} a {begin + len(showing)} de {len(filtered)} (imagens incluídas).")
            for t in showing:
                render_trade_row(t)
    with tab3:
        st.json(payload.get("configuration", {}))
        st.markdown("Pivots consecutivos, variação mínima de preço por percentagem e ATR, queda/subida oposta do RSI em **cada perna**, rejeição no pivot mais recente e filtro de 2h/6h/12h desativado.")
        st.caption("Histórico e PNG locais não são persistentes após um reinício/deploy do Render gratuito. Win rate apenas sobre o período efetivamente registado.")


render()
