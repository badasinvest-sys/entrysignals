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
from scipy.signal import find_peaks
from google import genai
from google.genai import types

# --- CREDENCIAIS DO TELEGRAM (Fixas) ---
TELEGRAM_TOKEN = "8882729761:AAEyKB_DBGEgecNdlOgeYVVm_zNWh778vkE"
TELEGRAM_CHAT_ID = "6311547341"

# --- CHAVE DA IA (Lida com segurança das Variáveis de Ambiente do Render) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(page_title="Crypto RSI Divergence Bot Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #121214; color: #ffffff; }
    .card-bearish {
        background-color: #1e1e24;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #ff5252;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .card-bullish {
        background-color: #1e1e24;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .ai-box {
        background-color: #262630;
        border-radius: 6px;
        padding: 10px;
        margin-top: 10px;
        font-size: 13px;
        border-left: 3px solid #00bcd4;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Crypto RSI Divergence & AI Quant Bot (Sem Limites de Tempo)")

SENT_FILE = "sent_signals_divergence_v7.json"

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

st.sidebar.header("⚙️ Configurações do Bot & IA")
enviar_telegram = st.sidebar.checkbox("Ativar Alertas Automáticos no Telegram", value=True)

if st.sidebar.button("🔄 Reenviar Alertas para Teste"):
    st.session_state.forcar_teste = True
    st.sidebar.success("Modo de teste ativado!")
else:
    st.session_state.forcar_teste = False

st.sidebar.markdown("---")
st.sidebar.info("💡 **Modo Profissional:** Deteção por pivots (`scipy.find_peaks`) sem restrições de tempo em todas as moedas USDT.")

def gerar_grafico_mini(df, entry_price, sl_price, tp_price, tipo):
    try:
        subset_df = df.copy()
        subset_df['timestamp'] = pd.to_datetime(subset_df['timestamp'], unit='ms')
        subset_df.set_index('timestamp', inplace=True)
        
        mc = mpf.make_marketcolors(up='#4caf50', down='#ff5252', edge='inherit', wick='inherit', volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, facecolor='#1e1e24', edgecolor='none', gridstyle='none')
        
        fig, axes = mpf.plot(subset_df, type='candle', style=s, figsize=(4, 2), axisoff=True, returnfig=True)
        ax = axes[0]
        
        total_velas = len(subset_df)
        largura_caixa = min(10, total_velas)
        x_inicio = total_velas - largura_caixa
        
        cor_caixa = '#ff5252' if tipo == 'bearish' else '#4caf50'
        
        altura_risco = abs(sl_price - entry_price)
        if altura_risco > 0:
            y_risco = entry_price if tipo == 'bearish' else sl_price
            rect_risk = Rectangle((x_inicio, y_risco), largura_caixa, altura_risco, facecolor=cor_caixa, alpha=0.35, edgecolor='none', zorder=3)
            ax.add_patch(rect_risk)
            
        altura_lucro = abs(entry_price - tp_price)
        if altura_lucro > 0:
            y_lucro = tp_price if tipo == 'bearish' else entry_price
            rect_reward = Rectangle((x_inicio, y_lucro), largura_caixa, altura_lucro, facecolor='#4caf50' if tipo=='bearish' else '#2196f3', alpha=0.35, edgecolor='none', zorder=3)
            ax.add_patch(rect_reward)
        
        ax.axhline(entry_price, color='#ffeb3b', linestyle='--', linewidth=1, alpha=0.9, zorder=4)
        ax.axhline(sl_price, color='#ff5252', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
        ax.axhline(tp_price, color='#4caf50', linestyle=':', linewidth=1, alpha=0.9, zorder=4)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#1e1e24', edgecolor='none', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        image_bytes = buf.getvalue()
        plt.close(fig)
        plt.close('all')
        gc.collect()
        
        return image_bytes
    except Exception:
        return None

def auditar_grafico_com_gemini(foto_bytes, symbol, tf, preco, tipo):
    if not GEMINI_API_KEY or foto_bytes is None:
        return "Análise de IA pendente."
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        direcao = "Short (Bearish)" if tipo == 'bearish' else "Long (Bullish)"
        prompt = (
            f"És um analista trader institucional especialista em Divergências de RSI. "
            f"Analisa este gráfico de {symbol} ({tf}) - Divergência {direcao} @ \${preco:.5f}. "
            f"Responde estritamente em duas linhas curtas em Português: "
            f"1) Validação técnica da divergência. "
            f"2) Veredito final com emoji (ex: Aprovado ✅ ou Cuidado ⚠️)."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=foto_bytes, mime_type="image/png")],
        )
        return response.text.strip()
    except Exception as e:
        return f"Erro na IA: {e}"

def enviar_alerta_telegram_com_foto(caption, foto_bytes):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or foto_bytes is None:
        return False, "Credenciais em falta."
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', foto_bytes, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=data, files=files, timeout=15)
        return (True, "Sucesso") if response.status_code == 200 else (False, response.text)
    except Exception as e:
        return False, f"Exceção: {e}"

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

def detetar_divergencia_rsi(df):
    try:
        if len(df) < 30:
            return None

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        rsi = df['rsi'].values

        # Deteção profissional de picos e fundos com scipy.signal.find_peaks (distância mínima de 4 velas)
        picos, _ = find_peaks(highs, distance=4)
        fundos, _ = find_peaks(-lows, distance=4)

        # 1. Avaliar Divergências Bearish (Picos)
        for n in range(len(picos) - 1, -1, -1):
            p2 = picos[n]
            for m in range(n - 1, -1, -1):
                p1 = picos[m]
                dist = p2 - p1
                # Sem restrição de tempo, apenas limites razoáveis de distância entre picos (3 a 200 velas)
                if 3 <= dist <= 200:
                    # Bearish: Preço sobe (Higher High) e RSI desce (Lower High)
                    if highs[p2] > highs[p1] and rsi[p2] < rsi[p1]:
                        signal_time = pd.to_datetime(df['timestamp'].iloc[p2], unit='ms')
                        entry = closes[p2]
                        sl = max(highs[p1], highs[p2]) * 1.015
                        risco = sl - entry
                        if risco > 0:
                            return {
                                "tipo": "bearish",
                                "entry": entry,
                                "sl": sl,
                                "tp": entry - (risco * 2.0),
                                "status": "Divergência Bearish RSI 🔻",
                                "signal_time": signal_time
                            }

        # 2. Avaliar Divergências Bullish (Fundos)
        for n in range(len(fundos) - 1, -1, -1):
            f2 = fundos[n]
            for m in range(n - 1, -1, -1):
                f1 = fundos[m]
                dist = f2 - f1
                if 3 <= dist <= 200:
                    # Bullish: Preço desce (Lower Low) e RSI sobe (Higher Low)
                    if lows[f2] < lows[f1] and rsi[f2] > rsi[f1]:
                        signal_time = pd.to_datetime(df['timestamp'].iloc[f2], unit='ms')
                        entry = closes[f2]
                        sl = min(lows[f1], lows[f2]) * 0.985
                        risco = entry - sl
                        if risco > 0:
                            return {
                                "tipo": "bullish",
                                "entry": entry,
                                "sl": sl,
                                "tp": entry + (risco * 2.0),
                                "status": "Divergência Bullish RSI 🟢",
                                "signal_time": signal_time
                            }
    except Exception:
        pass
    return None

@st.cache_data(ttl=300, show_spinner=False)
def analisar_mercado_autonomo():
    sinais_brutos = []
    tf_priority = {'15m': 1, '30m': 2, '1h': 3, '2h': 4, '4h': 5, '6h': 6, '12h': 7, '1d': 8}
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        exchange.load_markets()
        symbols = [s for s in exchange.markets if '/USDT' in s and not 'UP/' in s and not 'DOWN/' in s and exchange.markets[s]['active']]
        timeframes = ['15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']

        for symbol in symbols:
            for tf in timeframes:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=1000)
                    if len(ohlcv) < 50:
                        continue
                    df = pd.DataFrame(ohlcv[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['rsi'] = calcular_rsi(df['close'].values, 14)
                    setup = detetar_divergencia_rsi(df)
                    if setup:
                        signal_time = setup['signal_time']
                        data_fmt = signal_time.strftime('%d/%m %H:%M')
                        moeda = symbol.split('/')[0]
                        sid = f"{moeda}_{tf}_{setup['tipo']}_{signal_time.strftime('%Y%m%d%H')}"
                        foto = gerar_grafico_mini(df, setup['entry'], setup['sl'], setup['tp'], setup['tipo'])
                        parecer = auditar_grafico_com_gemini(foto, moeda, tf, setup['entry'], setup['tipo'])
                        sinais_brutos.append({
                            "id": sid, "moeda": moeda, "timeframe": tf, "tf_peso": tf_priority.get(tf, 1),
                            "preco": setup['entry'], "status": setup['status'], "tipo": setup['tipo'],
                            "data": data_fmt, "foto": foto, "parecer_ia": parecer
                        })
                except Exception:
                    continue
    except Exception:
        pass

    sinais_por_moeda = {}
    # Sem restrições de tempo: recolhemos o sinal mais recente válido por moeda
    for s in sinais_brutos:
        moeda = s['moeda']
        if moeda not in sinais_por_moeda or s['tf_peso'] > sinais_por_moeda[moeda]['tf_peso']:
            sinais_por_moeda[moeda] = s

    sinais = list(sinais_por_moeda.values())
    sinais.sort(key=lambda x: x['data'], reverse=True)
    return sinais

with st.spinner("A varrer o mercado com motor de pivots profissional (Sem limites de tempo)..."):
    try:
        sinais_registados = analisar_mercado_autonomo()
        sinais_enviados_set = carregar_sinais_enviados()

        if enviar_telegram and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            if st.session_state.get('forcar_teste', False):
                for s in sinais_registados[:2]:
                    if s['foto']:
                        caption = f"🧪 *[TESTE PRO]* 📌 *{s['moeda']}* `{s['timeframe']}`\n🤖 *IA:* {s['parecer_ia']}"
                        enviar_alerta_telegram_com_foto(caption, s['foto'])
            else:
                for s in sinais_registados:
                    if s['id'] not in sinais_enviados_set and s['foto']:
                        caption = f"📌 *DIVERGÊNCIA RSI* *{s['moeda']}* `{s['timeframe']}`\n🤖 *IA:* {s['parecer_ia']}"
                        sucesso, _ = enviar_alerta_telegram_com_foto(caption, s['foto'])
                        if sucesso:
                            guardar_sinal_enviado(s['id'])

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Divergências Detetadas", len(sinais_registados))
        col2.metric("Timeframes Varridos", "15m a 1D")
        col3.metric("Filtro Temporal", "Nenhum (Total)")
        st.markdown("---")

        if len(sinais_registados) == 0:
            st.info("Nenhuma divergência encontrada.")
        else:
            for s in sinais_registados:
                css_card = "card-bearish" if s['tipo'] == 'bearish' else "card-bullish"
                st.markdown(f"""
                    <div class="{css_card}">
                        <b style="font-size: 18px;">📌 DIVERGÊNCIA RSI: {s['moeda']}</b> 
                        <span style="background: #2d2d3d; padding: 2px 6px; border-radius: 4px; font-size: 12px;">{s['timeframe']}</span>
                        <span style="float: right; background: #252530; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{s['data']}</span>
                        <br><br>
                        <span>Preço @ <b>\${s['preco']:.5f}</b></span>
                        <span style="float: right;">Estado: <b style="color: #ff9800;">{s['status']}</b></span>
                        <div class="ai-box">🤖 <b>Parecer do Agente IA:</b> {s['parecer_ia']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Erro crítico: {e}")
