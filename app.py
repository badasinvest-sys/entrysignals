import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.patches import Rectangle
import io
import gc
import json
import os

st.set_page_config(page_title="Crypto Bot Top 250", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #121214; color: #ffffff; }
    .card {
        background-color: #1e1e24;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #ff5252;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Crypto Signal Bot (Top 250 & Correção Telegram)")

SENT_FILE = "sent_signals.json"

def carregar_sinais_enviados():
    if os.path.exists(SENT_FILE):
        try:
            with open(SENT_FILE, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def guardar_sinal_enviado(signal_id):
    enviados = carregar_sinais_enviados()
    enviados.add(signal_id)
    try:
        with open(SENT_FILE, "w") as f:
            json.dump(list(enviados), f)
    except Exception as e:
        print(f"Erro ao guardar sinal: {e}")

st.sidebar.header("⚙️ Configurações do Telegram")
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password", placeholder="Ex: 123456:ABC...")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", placeholder="Ex: 987654321")
enviar_telegram = st.sidebar.checkbox("Ativar Alertas Automáticos", value=True)

if st.sidebar.button("🔄 Reenviar Alertas para Teste (Últimas 2)"):
    st.session_state.forcar_teste = True
    st.sidebar.success("Modo de teste ativado!")
else:
    st.session_state.forcar_teste = False

st.sidebar.markdown("---")
timeframe_escolhido = st.sidebar.selectbox("Timeframe", ['1h', '2h', '4h'], index=2)

def gerar_grafico_mini(df, entry_price, sl_price, tp_price, status):
    subset_df = df.copy()
    subset_df['timestamp'] = pd.to_datetime(subset_df['timestamp'], unit='ms')
    subset_df.set_index('timestamp', inplace=True)
    
    mc = mpf.make_marketcolors(up='#4caf50', down='#ff5252', edge='inherit', wick='inherit', volume='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, facecolor='#1e1e24', edgecolor='none', gridstyle='none')
    
    fig, axes = mpf.plot(subset_df, type='candle', style=s, figsize=(3.5, 1.5), axisoff=True, returnfig=True)
    ax = axes[0]
    
    total_velas = len(subset_df)
    largura_caixa = min(10, total_velas)
    x_inicio = total_velas - largura_caixa
    
    altura_risco = entry_price - sl_price
    if altura_risco > 0:
        rect_risk = Rectangle((x_inicio, sl_price), largura_caixa, altura_risco, facecolor='#ff5252', alpha=0.35, edgecolor='none', zorder=3)
        ax.add_patch(rect_risk)
        
    altura_lucro = tp_price - entry_price
    if altura_lucro > 0:
        rect_reward = Rectangle((x_inicio, entry_price), largura_caixa, altura_lucro, facecolor='#4caf50', alpha=0.35, edgecolor='none', zorder=3)
        ax.add_patch(rect_reward)
    
    ax.axhline(entry_price, color='#ffeb3b', linestyle='--', linewidth=1, alpha=0.9, zorder=4)
    ax.axhline(sl_price, color='#ff5252', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
    ax.axhline(tp_price, color='#4caf50', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#1e1e24', edgecolor='none', bbox_inches='tight', dpi=72)
    buf.seek(0)
    
    # Retornar os bytes puros garante que a imagem não corrompe se for reenviada
    image_bytes = buf.getvalue()
    
    plt.close(fig)
    plt.close('all')
    gc.collect()
    
    return image_bytes

def enviar_alerta_telegram_com_foto(token, chat_id, caption, foto_bytes):
    if not token or not chat_id:
        return False, "Token ou Chat ID em falta."
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('chart.png', foto_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=data, files=files, timeout=15)
        if response.status_code == 200:
            return True, "Sucesso"
        else:
            return False, f"Erro Telegram ({response.status_code}): {response.text}"
    except Exception as e:
        return False, f"Exceção de rede: {e}"

def calcular_rsi(closes, period=14):
    deltas = np.diff(closes)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(closes)
    rsi[:period] = 100. - 100./(1. + rs)
    
    for i in range(period, len(closes)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up*(period - 1) + upval)/period
        down = (down*(period - 1) + downval)/period
        rs = up/down if down != 0 else 0
        rsi[i] = 100. - 100./(1. + rs)
    return rsi

@st.cache_data(ttl=300, show_spinner=False)
def analisar_mercado(tf):
    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.load_markets()
    tickers = exchange.fetch_tickers()
    
    pairs_volume = []
    for symbol, ticker in tickers.items():
        if '/USDT' in symbol and not 'UP/' in symbol and not 'DOWN/' in symbol:
            quote_vol = ticker.get('quoteVolume', 0)
            if quote_vol and quote_vol > 5000000:
                pairs_volume.append((symbol, quote_vol))
                
    pairs_volume.sort(key=lambda x: x[1], reverse=True)
    symbols = [item[0] for item in pairs_volume[:250]]

    total_sinais = 0
    vitorias = 0
    perdas = 0
    em_andamento = 0
    sinais_registados = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=50)
            if len(ohlcv) < 30:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['rsi'] = calcular_rsi(df['close'].values, 14)
            
            inicio_busca = max(19, len(df) - 15)
            
            for i in range(inicio_busca, len(df) - 1):
                prev_rsi = df['rsi'].iloc[i-1]
                curr_rsi = df['rsi'].iloc[i]
                
                if prev_rsi < 35 and curr_rsi >= 35:
                    entry_price = df['close'].iloc[i]
                    sl_price = df['low'].iloc[i-5:i].min()
                    if sl_price >= entry_price:
                        sl_price = entry_price * 0.98
                        
                    risk = entry_price - sl_price
                    tp_price = entry_price + (risk * 2.0)
                    
                    signal_time = pd.to_datetime(df['timestamp'].iloc[i], unit='ms')
                    
                    status = "em curso"
                    for j in range(i + 1, len(df)):
                        if df['high'].iloc[j] >= tp_price:
                            status = "2R ✓"
                            vitorias += 1
                            break
                        elif df['low'].iloc[j] <= sl_price:
                            status = "Loss ❌"
                            perdas += 1
                            break
                    
                    if status == "em curso":
                        em_andamento += 1

                    total_sinais += 1
                    horas_atras = int((datetime.now() - signal_time.to_pydatetime()).total_seconds() // 3600)
                    data_formatada = signal_time.strftime('%d/%m %H:%M')
                    nome_moeda = symbol.split('/')[0]
                    
                    signal_id = f"{nome_moeda}_{tf}_{signal_time}"
                    
                    foto_bytes = None
                    if horas_atras <= 24:
                        foto_bytes = gerar_grafico_mini(df.iloc[:i+1], entry_price, sl_price, tp_price, status)

                    sinais_registados.append({
                        "id": signal_id,
                        "moeda": nome_moeda,
                        "timeframe": tf,
                        "preco": entry_price,
                        "status": status,
                        "horas": horas_atras,
                        "data": data_formatada,
                        "foto": foto_bytes
                    })
        except:
            continue

    sinais_registados.sort(key=lambda x: x['data'], reverse=True)
    return total_sinais, vitorias, perdas, em_andamento, sinais_registados

if enviar_telegram and (not telegram_token or not telegram_chat_id):
    st.warning("⚠️ Insere o **Telegram Bot Token** e o **Chat ID** na barra lateral.")

with st.spinner(f"A vasculhar o Top 250 ({timeframe_escolhido})..."):
    try:
        total_sinais, vitorias, perdas, em_andamento, sinais_registados = analisar_mercado(timeframe_escolhido)

        sinais_enviados_set = carregar_sinais_enviados()

        if enviar_telegram and telegram_token and telegram_chat_id:
            if st.session_state.get('forcar_teste', False):
                trades_teste = sinais_registados[:2]
                if not trades_teste:
                    st.warning("Nenhum sinal recente encontrado para teste.")
                for s in trades_teste:
                    if s['foto']:
                        caption = (
                            f"🧪 *[TESTE]* *{s['moeda']}*  `{s['timeframe']}`          📅 `{s['data']}`\n"
                            f"RSI reset... @ `${s['preco']:.5f}`   *{s['status']}*   há {s['horas']} h"
                        )
                        sucesso, msg = enviar_alerta_telegram_com_foto(telegram_token, telegram_chat_id, caption, s['foto'])
                        if sucesso:
                            st.success(f"✅ Alerta de teste enviado para {s['moeda']} no Telegram!")
                        else:
                            st.error(f"❌ Falha no Telegram para {s['moeda']}: {msg}")
            else:
                for s in sinais_registados:
                    if s['id'] not in sinais_enviados_set and s['foto'] and s['horas'] <= 6:
                        caption = (
                            f"🔴 *{s['moeda']}*  `{s['timeframe']}`          📅 `{s['data']}`\n"
                            f"RSI reset... @ `${s['preco']:.5f}`   *{s['status']}*   há {s['horas']} h"
                        )
                        sucesso, msg = enviar_alerta_telegram_com_foto(telegram_token, telegram_chat_id, caption, s['foto'])
                        if sucesso:
                            guardar_sinal_enviado(s['id'])
                        else:
                            st.error(f"❌ Erro ao enviar automático ({s['moeda']}): {msg}")

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Sinais (Top 250)", total_sinais)
        wr = (vitorias / (vitorias + perdas) * 100) if (vitorias + perdas) > 0 else 0
        col2.metric("Win Rate (% WR)", f"{wr:.1f}%")
        col3.metric("Vitórias (2R ✓)", vitorias)
        col4.metric("Em Curso", em_andamento)
        st.markdown("---")

        if len(sinais_registados) == 0:
            st.info("Nenhum sinal detetado nas últimas velas com os parâmetros definidos.")
        else:
            for s in sinais_registados:
                cor_status = "#4caf50" if s["status"] == "2R ✓" else ("#ff5252" if s["status"] == "Loss ❌" else "#ff9800")
                st.markdown(f"""
                    <div class="card">
                        <b style="font-size: 18px; color: #ff5252;">● {s['moeda']}</b> 
                        <span style="background: #2d2d3d; padding: 2px 6px; border-radius: 4px; font-size: 12px;">{s['timeframe']}</span>
                        <span style="float: right; background: #3d1a1a; color: #ff8080; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{s['data']}</span>
                        <br><br>
                        <span>RSI reset + ... @ <b>\${s['preco']:.5f}</b></span>
                        <span style="float: right;">
                            Estado: <b style="color: {cor_status};">{s['status']}</b> | há {s['horas']} h
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"Erro ao carregar o mercado: {e}")
