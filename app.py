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

# --- CREDENCIAIS DO TELEGRAM FIXAS E BLINDADAS ---
TELEGRAM_TOKEN = "8882729761:AAEyKB_DBGEgecNdlOgeYVVm_zNWh778vkE"
TELEGRAM_CHAT_ID = "6311547341"

st.set_page_config(page_title="Crypto SMC Bot (Top 250 & Multi-Timeframe)", layout="wide")

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

st.title("⚡ Crypto SMC & Divergence Bot (Automático 24/7)")

SENT_FILE = "sent_signals_smc.json"

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

st.sidebar.header("⚙️ Configurações do Bot")
enviar_telegram = st.sidebar.checkbox("Ativar Alertas Automáticos no Telegram", value=True)

if st.sidebar.button("🔄 Reenviar Alertas para Teste (Últimas 2)"):
    st.session_state.forcar_teste = True
    st.sidebar.success("Modo de teste ativado!")
else:
    st.session_state.forcar_teste = False

st.sidebar.markdown("---")
st.sidebar.info("💡 **Modo Totalmente Autónomo:** As credenciais do Telegram estão configuradas. O bot analisa o Top 250 nos timeframes **2h, 4h e 1d** de 5 em 5 minutos.")

def gerar_grafico_mini(df, entry_price, sl_price, tp_price):
    try:
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
        
        altura_risco = sl_price - entry_price
        if altura_risco > 0:
            rect_risk = Rectangle((x_inicio, entry_price), largura_caixa, altura_risco, facecolor='#ff5252', alpha=0.35, edgecolor='none', zorder=3)
            ax.add_patch(rect_risk)
            
        altura_lucro = entry_price - tp_price
        if altura_lucro > 0:
            rect_reward = Rectangle((x_inicio, tp_price), largura_caixa, altura_lucro, facecolor='#4caf50', alpha=0.35, edgecolor='none', zorder=3)
            ax.add_patch(rect_reward)
        
        ax.axhline(entry_price, color='#ffeb3b', linestyle='--', linewidth=1, alpha=0.9, zorder=4)
        ax.axhline(sl_price, color='#ff5252', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
        ax.axhline(tp_price, color='#4caf50', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#1e1e24', edgecolor='none', bbox_inches='tight', dpi=72)
        buf.seek(0)
        
        image_bytes = buf.getvalue()
        plt.close(fig)
        plt.close('all')
        gc.collect()
        
        return image_bytes
    except Exception:
        return None

def enviar_alerta_telegram_com_foto(caption, foto_bytes):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or foto_bytes is None:
        return False, "Credenciais ou imagem em falta."
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', foto_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
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

def detetar_divergencia_e_estrutura_short(df):
    try:
        if len(df) < 40:
            return None

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        rsi = df['rsi'].values

        picos_idx = []
        for i in range(2, len(df) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i-2] < highs[i] and highs[i+2] < highs[i]:
                picos_idx.append(i)

        if len(picos_idx) >= 3:
            p1, p2, p3 = picos_idx[-3], picos_idx[-2], picos_idx[-1]
            
            preco_sobe = highs[p3] > highs[p2] > highs[p1]
            rsi_desce = rsi[p3] < rsi[p2] < rsi[p1] and rsi[p1] > 60

            if preco_sobe and rsi_desce:
                rsi_reset = any(rsi[p3:] < 55)
                ultimo_fundo = lows[p3:].min()
                preco_atual = closes[-1]
                
                if rsi_reset and (highs[-1] >= highs[p3] * 0.98 or lows[-1] <= ultimo_fundo * 1.02):
                    entry_price = preco_atual
                    sl_price = max(highs[p3:]) * 1.015 
                    risco = sl_price - entry_price
                    if risco <= 0:
                        return None
                    tp_price = entry_price - (risco * 2.0) 
                    
                    return {
                        "entry": entry_price,
                        "sl": sl_price,
                        "tp": tp_price,
                        "status": "Short em Curso 🔻"
                    }
    except Exception:
        pass
    return None

@st.cache_data(ttl=300, show_spinner=False)
def analisar_mercado_autonomo():
    sinais_registados = []
    try:
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
        
        timeframes = ['2h', '4h', '1d']

        for symbol in symbols:
            for tf in timeframes:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=80)
                    if len(ohlcv) < 50:
                        continue
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['rsi'] = calcular_rsi(df['close'].values, 14)
                    
                    setup = detetar_divergencia_e_estrutura_short(df)
                    if setup:
                        signal_time = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')
                        horas_atras = int((datetime.now() - signal_time.to_pydatetime()).total_seconds() // 3600)
                        data_formatada = signal_time.strftime('%d/%m %H:%M')
                        nome_moeda = symbol.split('/')[0]
                        signal_id = f"{nome_moeda}_{tf}_{signal_time.strftime('%Y%m%d%H')}"
                        
                        foto_bytes = gerar_grafico_mini(df, setup['entry'], setup['sl'], setup['tp'])
                        
                        sinais_registados.append({
                            "id": signal_id,
                            "moeda": nome_moeda,
                            "timeframe": tf,
                            "preco": setup['entry'],
                            "status": setup['status'],
                            "horas": max(0, horas_atras),
                            "data": data_formatada,
                            "foto": foto_bytes
                        })
                except Exception:
                    continue
    except Exception:
        pass

    sinais_registados.sort(key=lambda x: x['data'], reverse=True)
    return sinais_registados

with st.spinner("A analisar Top 250 em Multi-Timeframe (2h, 4h, 1d) por Divergências e SMC..."):
    try:
        sinais_registados = analisar_mercado_autonomo()
        sinais_enviados_set = carregar_sinais_enviados()

        if enviar_telegram and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            if st.session_state.get('forcar_teste', False):
                trades_teste = sinais_registados[:2]
                if not trades_teste:
                    st.warning("Nenhum sinal SMC recente encontrado para teste.")
                for s in trades_teste:
                    if s['foto']:
                        caption = (
                            f"🧪 *[TESTE SMC]* 🔻 *{s['moeda']}* `{s['timeframe']}`   📅 `{s['data']}`\n"
                            f"Divergência Tripla + Sweep @ `${s['preco']:.5f}`\n"
                            f"Estado: *{s['status']}*"
                        )
                        sucesso, msg = enviar_alerta_telegram_com_foto(caption, s['foto'])
                        if sucesso:
                            st.success(f"✅ Alerta de teste SMC enviado para {s['moeda']} ({s['timeframe']})!")
                        else:
                            st.error(f"❌ Falha no Telegram para {s['moeda']}: {msg}")
            else:
                for s in sinais_registados:
                    if s['id'] not in sinais_enviados_set and s['foto'] and s['horas'] <= 6:
                        caption = (
                            f"🔻 *SMC SHORT* *{s['moeda']}* `{s['timeframe']}`   📅 `{s['data']}`\n"
                            f"Divergência Tripla RSI + Liquidity Sweep @ `${s['preco']:.5f}`"
                        )
                        sucesso, msg = enviar_alerta_telegram_com_foto(caption, s['foto'])
                        if sucesso:
                            guardar_sinal_enviado(s['id'])

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Oportunidades Ativas (Top 250)", len(sinais_registados))
        col2.metric("Timeframes Varridos", "2h, 4h, 1d")
        col3.metric("Estado da Rede", "🟢 Ativo 24/7")
        st.markdown("---")

        if len(sinais_registados) == 0:
            st.info("A aguardar formações de divergência tripla e sweeps de liquidez nos timeframes superiores...")
        else:
            for s in sinais_registados:
                st.markdown(f"""
                    <div class="card">
                        <b style="font-size: 18px; color: #ff5252;">🔻 SMC SHORT: {s['moeda']}</b> 
                        <span style="background: #2d2d3d; padding: 2px 6px; border-radius: 4px; font-size: 12px;">{s['timeframe']}</span>
                        <span style="float: right; background: #3d1a1a; color: #ff8080; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{s['data']}</span>
                        <br><br>
                        <span>Divergência Tripla RSI + Sweep @ <b>\${s['preco']:.5f}</b></span>
                        <span style="float: right;">
                            Estado: <b style="color: #ff9800;">{s['status']}</b>
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Erro crítico no motor de execução: {e}")
