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

# Configuração da página
st.set_page_config(page_title="Crypto Bot com RR Ajustado", layout="wide")

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

st.title("⚡ Crypto Signal Bot (Zonas de Entrada Ajustadas)")

# Memória para controlar os alertas enviados na sessão
if 'sinais_enviados' not in st.session_state:
    st.session_state.sinais_enviados = set()

# Barra lateral para configurações
st.sidebar.header("⚙️ Configurações do Telegram")
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password", placeholder="Ex: 123456:ABC...")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", placeholder="Ex: 987654321")
enviar_telegram = st.sidebar.checkbox("Ativar Alertas Automáticos", value=True)

if st.sidebar.button("🔄 Reenviar Alertas para Teste"):
    st.session_state.sinais_enviados.clear()
    st.sidebar.success("Memória limpa! Atualiza a página para reenviar.")

st.sidebar.markdown("---")
timeframe_escolhido = st.sidebar.selectbox("Timeframe", ['1h', '2h', '4h'], index=2)

def gerar_grafico_mini(df, entry_price, sl_price, tp_price, status):
    """Gera um mini-gráfico de velas com as caixas de RR limitadas à zona final (estilo TradingView)"""
    subset_df = df.copy()
    subset_df['timestamp'] = pd.to_datetime(subset_df['timestamp'], unit='ms')
    subset_df.set_index('timestamp', inplace=True)
    
    mc = mpf.make_marketcolors(up='#4caf50', down='#ff5252',
                               edge='inherit', wick='inherit',
                               volume='inherit')
    
    s = mpf.make_mpf_style(marketcolors=mc, facecolor='#1e1e24', edgecolor='none', gridstyle='none')
    
    fig, axes = mpf.plot(subset_df, type='candle', style=s, figsize=(6, 2.5),
                         axisoff=True, returnfig=True)
    ax = axes[0]
    
    # Delimitar a caixa apenas para as últimas velas (bloco à direita, ex: últimas 12 velas)
    total_velas = len(subset_df)
    largura_caixa = min(12, total_velas)
    x_inicio = total_velas - largura_caixa
    
    # Caixa de Risco (Stop Loss até Entrada) - Vermelho transparente
    altura_risco = entry_price - sl_price
    if altura_risco > 0:
        rect_risk = Rectangle((x_inicio, sl_price), largura_caixa, altura_risco, 
                              facecolor='#ff5252', alpha=0.35, edgecolor='none', zorder=3)
        ax.add_patch(rect_risk)
        
    # Caixa de Recompensa (Entrada até Take Profit) - Verde transparente
    altura_lucro = tp_price - entry_price
    if altura_lucro > 0:
        rect_reward = Rectangle((x_inicio, entry_price), largura_caixa, altura_lucro, 
                                facecolor='#4caf50', alpha=0.35, edgecolor='none', zorder=3)
        ax.add_patch(rect_reward)
    
    # Linhas indicadoras de preço discretas
    ax.axhline(entry_price, color='#ffeb3b', linestyle='--', linewidth=1, alpha=0.9, zorder=4)
    ax.axhline(sl_price, color='#ff5252', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
    ax.axhline(tp_price, color='#4caf50', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#1e1e24', edgecolor='none', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def enviar_alerta_telegram_com_foto(token, chat_id, caption, foto_bytes):
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('chart.png', foto_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=data, files=files, timeout=10)
    except Exception as e:
        print(f"Erro ao enviar para o Telegram: {e}")

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

# Scan automático do mercado
exchange = ccxt.binance({'enableRateLimit': True})
RSI_PERIOD = 14
RR_TARGET = 2.0

with st.spinner(f"A executar scan automático ({timeframe_escolhido}) | Filtro MC > 100M\$..."):
    try:
        exchange.load_markets()
        tickers = exchange.fetch_tickers()
        
        symbols = []
        for symbol, ticker in tickers.items():
            if '/USDT' in symbol and not 'UP/' in symbol and not 'DOWN/' in symbol:
                quote_vol = ticker.get('quoteVolume', 0)
                if quote_vol and quote_vol > 5000000:
                    symbols.append(symbol)

        total_sinais = 0
        vitorias = 0
        perdas = 0
        em_andamento = 0
        sinais_registados = []

        for symbol in symbols[:20]:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe_escolhido, limit=60)
                if len(ohlcv) < 30:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['rsi'] = calcular_rsi(df['close'].values, RSI_PERIOD)
                
                inicio_busca = max(RSI_PERIOD + 5, len(df) - 15)
                
                for i in range(inicio_busca, len(df) - 1):
                    prev_rsi = df['rsi'].iloc[i-1]
                    curr_rsi = df['rsi'].iloc[i]
                    
                    if prev_rsi < 35 and curr_rsi >= 35:
                        entry_price = df['close'].iloc[i]
                        sl_price = df['low'].iloc[i-5:i].min()
                        if sl_price >= entry_price:
                            sl_price = entry_price * 0.98
                            
                        risk = entry_price - sl_price
                        tp_price = entry_price + (risk * RR_TARGET)
                        
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
                        
                        sinais_registados.append({
                            "moeda": nome_moeda,
                            "timeframe": timeframe_escolhido,
                            "preco": entry_price,
                            "status": status,
                            "horas": horas_atras,
                            "data": data_formatada
                        })

                        signal_id = f"{nome_moeda}_{timeframe_escolhydo if 'timeframe_escolhydo' in locals() else timeframe_escolhido}_{signal_time}"

                        # Enviar para o Telegram com as caixas de RR localizadas
                        if enviar_telegram and horas_atras <= 24 and signal_id not in st.session_state.sinais_enviados:
                            foto_bytes = gerar_grafico_mini(df.iloc[:i+1], entry_price, sl_price, tp_price, status)
                            
                            caption = (
                                f"🔴 *{nome_moeda}*  `{timeframe_escolhido}`          📅 `{data_formatada}`\n"
                                f"RSI reset... @ `${entry_price:.5f}`   *{status}*   há {horas_atras} h"
                            )
                            enviar_alerta_telegram_com_foto(telegram_token, telegram_chat_id, caption, foto_bytes)
                            st.session_state.sinais_enviados.add(signal_id)
            except:
                continue

        # Métricas de Win Rate no Topo
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Sinais", total_sinais)
        wr = (vitorias / (vitorias + perdas) * 100) if (vitorias + perdas) > 0 else 0
        col2.metric("Win Rate (% WR)", f"{wr:.1f}%")
        col3.metric("Vitórias (2R ✓)", vitorias)
        col4.metric("Em Curso", em_andamento)
        st.markdown("---")

        # Renderizar cartões na web
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