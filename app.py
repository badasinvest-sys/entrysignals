"""Painel Streamlit para os sinais RSI escritos pelo scanner local."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))

st.set_page_config(page_title="Crypto RSI Divergences", page_icon="📈", layout="wide")
st.title("📈 Crypto RSI Divergences")
st.caption(
    "Deteção de divergências regulares RSI bullish/bearish. Nesta fase, o alerta identifica o padrão; "
    "não tenta decidir se já existe uma entrada válida."
)


def read_latest() -> dict | None:
    try:
        if not LATEST_FILE.exists():
            return None
        with LATEST_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        st.error(f"Não foi possível ler os resultados do scanner: {exc}")
        return None


@st.fragment(run_every="30s")
def mostrar_resultados() -> None:
    payload = read_latest()
    if payload is None:
        st.info(
            "O scanner está a iniciar a primeira varredura. Aguarda um pouco e este painel atualiza automaticamente. "
            "Confirma os logs do Render se isto não mudar."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Divergências recentes", int(payload.get("signals_count", 0)))
    c2.metric("Pares verificados", int(payload.get("symbols_scanned", 0)))
    c3.metric("Pedidos OHLCV OK", int(payload.get("ohlcv_requests_ok", 0)))
    c4.metric("Erros de mercado/API", int(payload.get("errors", 0)))

    st.caption(
        f"Exchange: {payload.get('exchange', '?')} · Último scan: {payload.get('scan_completed_at', '?')} · "
        f"Duração: {payload.get('scan_seconds', '?')} s"
    )
    st.caption("Timeframes: " + ", ".join(payload.get("timeframes", [])))

    config = payload.get("configuration", {})
    with st.expander("Parâmetros de deteção atuais"):
        st.json(config)

    signals = payload.get("signals", [])
    if not signals:
        st.info(
            "Não foram confirmadas divergências recentes neste scan. Isso pode significar que não há um padrão "
            "que cumpra os limiares, e não necessariamente que o scanner falhou. Consulta os logs para confirmar "
            "que os pedidos OHLCV estão a funcionar."
        )
        return

    for signal in signals:
        bullish = signal.get("tipo") == "bullish"
        icon = "🟢" if bullish else "🔻"
        title = (
            f"{icon} {signal.get('symbol', '?')} · {signal.get('timeframe', '?')} · "
            f"{signal.get('tipo', '').upper()} · {signal.get('pivots_count', 2)} pivots"
        )
        with st.container(border=True):
            st.subheader(title)
            st.caption(
                f"{signal.get('forca_padrao', 'Clássica')} — {signal.get('descricao_padrao', '')} · "
                f"Pivot confirmado: {signal.get('signal_time', '?')} · "
                f"Idade: {signal.get('idade_velas', '?')} vela(s)"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Preço inicial → final", f"{signal.get('preco_inicial', 0):.8g} → {signal.get('preco_final', 0):.8g}")
            m2.metric("RSI inicial → final", f"{signal.get('rsi_inicial', 0):.2f} → {signal.get('rsi_final', 0):.2f}")
            m3.metric("Variação do preço", f"{signal.get('variacao_preco_pct', 0):+.3f}%")
            m4.metric("Variação do RSI", f"{signal.get('variacao_rsi', 0):+.2f}")

            pivots = signal.get("pivots", [])
            if pivots:
                table_rows = []
                for number, pivot in enumerate(pivots, start=1):
                    table_rows.append({
                        "Pivot": number,
                        "Data do extremo de preço": pivot.get("price_time", "?"),
                        "Preço": float(pivot.get("price", 0.0)),
                        "RSI": float(pivot.get("rsi", 0.0)),
                        "Data do extremo RSI": pivot.get("rsi_time", "?"),
                    })
                st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Definição usada: bearish = máximo mais alto no preço + máximo mais baixo no RSI; "
        "bullish = mínimo mais baixo no preço + mínimo mais alto no RSI. Padrões com 3 ou mais pivots "
        "só são classificados como multi-pivot se as pernas intermédias mantiverem a mesma direção."
    )


mostrar_resultados()
