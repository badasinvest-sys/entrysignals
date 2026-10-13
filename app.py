from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))
TRADES_FILE = Path(os.getenv("TRADES_FILE", str(BASE_DIR / "trades_history.json")))

st.set_page_config(page_title="Scanner RSI + Breakouts", page_icon="📊", layout="wide")
st.title("📊 Scanner RSI + Breakouts")
st.caption("Divergências RSI por highs/lows absolutos e setups de rompimento/reteste em 4h e 1d.")


def read_json(path: Path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Não foi possível ler {path.name}: {exc}")
        return None


@st.fragment(run_every="30s")
def render() -> None:
    payload = read_json(LATEST_FILE)
    trades_payload = read_json(TRADES_FILE) or {"trades": []}
    trades = trades_payload.get("trades", []) if isinstance(trades_payload, dict) else []

    if payload is None:
        st.info("O scanner ainda está a arrancar. Aguarda o primeiro scan.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sinais recentes", int(payload.get("signals_count", 0)))
    c2.metric("Pares verificados", int(payload.get("symbols_scanned", 0)))
    c3.metric("OHLCV OK", int(payload.get("ohlcv_requests_ok", 0)))
    c4.metric("Erros API", int(payload.get("errors", 0)))

    st.caption(
        f"Exchange: {payload.get('exchange', '?')} · Último scan: {payload.get('scan_completed_at', '?')} · "
        f"Duração: {payload.get('scan_seconds', '?')} s"
    )
    st.caption("Timeframes ativos: " + ", ".join(payload.get("timeframes", [])))

    tab1, tab2, tab3 = st.tabs(["Sinais atuais", "Trades & win rate", "Configuração"])

    with tab1:
        signals = payload.get("signals", [])
        if not signals:
            st.info("Sem sinais recentes neste scan.")
        for signal in signals:
            categoria = signal.get("categoria")
            if categoria == "rsi_divergence":
                icon = "🔻" if signal.get("tipo") == "bearish" else "🟢"
                title = f"{icon} {signal.get('symbol')} · {signal.get('timeframe')} · RSI {signal.get('tipo', '').upper()}"
                with st.container(border=True):
                    st.subheader(title)
                    st.caption(f"{signal.get('pivots_count')} pivots · {signal.get('descricao_padrao')} · Confirmada: {signal.get('signal_time')}")
                    prices = [float(p['price']) for p in signal.get('pivots', [])]
                    rsis = [float(p['rsi']) for p in signal.get('pivots', [])]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Preço", " → ".join(f"{x:.8g}" for x in prices))
                    m2.metric("RSI", " → ".join(f"{x:.1f}" for x in rsis))
                    m3.metric("Var. preço", f"{signal.get('variacao_preco_pct', 0):+.2f}%")
                    m4.metric("Var. RSI", f"{signal.get('variacao_rsi', 0):+.2f}")
                    rows = []
                    for i, p in enumerate(signal.get("pivots", []), start=1):
                        rows.append({
                            "Pivot": i,
                            "Data preço": p.get("price_time"),
                            "Preço": p.get("price"),
                            "RSI": p.get("rsi"),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            elif categoria == "breakout":
                icon = "🚀" if signal.get("fase") == "Possível setup" else "🎯"
                title = f"{icon} {signal.get('symbol')} · {signal.get('timeframe')} · {signal.get('fase')}"
                with st.container(border=True):
                    st.subheader(title)
                    st.caption(f"Zona {signal.get('zona_min'):.8g} - {signal.get('zona_max'):.8g} · Confirmado: {signal.get('signal_time')}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Nível", f"{signal.get('nivel', 0):.8g}")
                    m2.metric("Toques", int(signal.get('toques_resistencia', 0)))
                    m3.metric("Vol. zona", f"{signal.get('volume_zona_ratio', 0)}x")
                    m4.metric("Vol. rompimento", f"{signal.get('volume_rompimento_ratio', 0)}x")
                    if signal.get("trade"):
                        trade = signal["trade"]
                        st.markdown(
                            f"**Sugestão de trade:** entrada `{trade['entry']:.8g}` · SL `{trade['sl']:.8g}` · TP `{trade['tp']:.8g}` · RR `2:1`"
                        )

    with tab2:
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            st.info("Ainda não existem trades registadas.")
        else:
            closed = trades_df[trades_df["status"].isin(["Ganha", "Perdida"])].copy()
            wins = int((closed["status"] == "Ganha").sum())
            losses = int((closed["status"] == "Perdida").sum())
            total_closed = len(closed)
            win_rate = (wins / total_closed * 100.0) if total_closed else 0.0
            em_curso = int((trades_df["status"] == "Em curso").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades fechadas", total_closed)
            c2.metric("Win rate", f"{win_rate:.1f}%")
            c3.metric("Ganhos", wins)
            c4.metric("Em curso", em_curso)

            recent = trades_df.sort_values(by="opened_at_ms", ascending=False).copy()
            show_cols = [
                "symbol", "timeframe", "entrada", "sl", "tp", "status",
                "opened_at", "closed_at", "resultado_r"
            ]
            for col in show_cols:
                if col not in recent.columns:
                    recent[col] = None
            st.subheader("Trades recentes")
            st.dataframe(recent[show_cols], hide_index=True, use_container_width=True)

    with tab3:
        st.json(payload.get("configuration", {}))
        st.caption("Notas: 2h, 6h e 12h foram excluídos. Divergências de 2 pivots só são aceites em 4h e 1d.")


render()
