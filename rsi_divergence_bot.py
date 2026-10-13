from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import ccxt
except ImportError:
    ccxt = None  # type: ignore[assignment]

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
try:
    import mplfinance as mpf
except ImportError:
    mpf = None  # type: ignore[assignment]
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

from breakout_setup import atr_value, detetar_setups_breakout

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))
SENT_FILE = Path(os.getenv("SENT_FILE", str(BASE_DIR / "sent_alerts.json")))
TRADES_FILE = Path(os.getenv("TRADES_FILE", str(BASE_DIR / "trades_history.json")))

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "mexc").strip().lower()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SCAN_INTERVAL_SECONDS = max(60, int(os.getenv("SCAN_INTERVAL_SECONDS", "300")))
SCAN_TOP_N = max(1, int(os.getenv("SCAN_TOP_N", "35")))
OHLCV_LIMIT = max(140, min(600, int(os.getenv("OHLCV_LIMIT", "320"))))
RSI_PERIOD = max(2, int(os.getenv("RSI_PERIOD", "14")))
MAX_SIGNAL_AGE_BARS = max(1, int(os.getenv("MAX_SIGNAL_AGE_BARS", "4")))
MIN_RSI_DIFFERENCE = max(0.0, float(os.getenv("MIN_RSI_DIFFERENCE", "2.0")))
MIN_PRICE_DIFFERENCE_PCT = max(0.0, float(os.getenv("MIN_PRICE_DIFFERENCE_PCT", "0.0015")))
PRICE_PIVOT_DISTANCE = max(2, int(os.getenv("PRICE_PIVOT_DISTANCE", "5")))
PRICE_PIVOT_PROMINENCE_ATR = max(0.1, float(os.getenv("PRICE_PIVOT_PROMINENCE_ATR", "0.55")))
PIVOT_MATCH_RADIUS = max(1, int(os.getenv("PIVOT_MATCH_RADIUS", "3")))
MIN_PIVOT_SEPARATION = max(2, int(os.getenv("MIN_PIVOT_SEPARATION", "5")))
MAX_PIVOT_SEPARATION = max(MIN_PIVOT_SEPARATION + 1, int(os.getenv("MAX_PIVOT_SEPARATION", "90")))
TIMEFRAMES_DEFAULT = ["15m", "30m", "1h", "4h", "1d"]
ALLOWED_TIMEFRAMES = set(TIMEFRAMES_DEFAULT)
TIMEFRAMES = [tf for tf in [x.strip() for x in os.getenv("TIMEFRAMES", ",".join(TIMEFRAMES_DEFAULT)).split(",") if x.strip()] if tf in ALLOWED_TIMEFRAMES]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rsi-breakout-bot")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Falha ao ler %s", path.name)
    return default


def read_sent_ids() -> set[str]:
    data = read_json(SENT_FILE, [])
    if isinstance(data, list):
        return {str(x) for x in data}
    return set()


def save_sent_ids(sent_ids: set[str]) -> None:
    write_json_atomic(SENT_FILE, sorted(sent_ids)[-20000:])


def read_trades() -> list[dict[str, Any]]:
    data = read_json(TRADES_FILE, {"trades": []})
    if isinstance(data, dict) and isinstance(data.get("trades"), list):
        return data["trades"]
    return []


def save_trades(trades: list[dict[str, Any]]) -> None:
    payload = {"updated_at": utc_now_iso(), "trades": trades}
    write_json_atomic(TRADES_FILE, payload)


def calcular_rsi(closes: np.ndarray | list[float], period: int = 14) -> np.ndarray:
    prices = np.asarray(closes, dtype=float)
    result = np.full(prices.size, np.nan, dtype=float)
    if prices.size <= period or period < 2:
        return result
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def to_rsi(gain: float, loss: float) -> float:
        if loss == 0.0 and gain == 0.0:
            return 50.0
        if loss == 0.0:
            return 100.0
        if gain == 0.0:
            return 0.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    result[period] = to_rsi(avg_gain, avg_loss)
    for close_idx in range(period + 1, prices.size):
        delta_idx = close_idx - 1
        avg_gain = (avg_gain * (period - 1) + gains[delta_idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[delta_idx]) / period
        result[close_idx] = to_rsi(avg_gain, avg_loss)
    return result


def _point_for_ui(idx: int, price: float, rsi_idx: int, rsi: float, times: np.ndarray) -> dict[str, Any]:
    return {
        "price_idx": int(idx),
        "rsi_idx": int(rsi_idx),
        "price_time": iso_from_ms(int(times[idx])),
        "rsi_time": iso_from_ms(int(times[rsi_idx])),
        "price": float(price),
        "rsi": round(float(rsi), 2),
    }


def _price_pivots(df: pd.DataFrame, direction: str) -> list[dict[str, Any]]:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    rsi = df["rsi"].to_numpy(dtype=float)
    times = df["timestamp"].to_numpy(dtype=np.int64)
    atr = max(atr_value(df), float(np.median(df["close"].to_numpy(dtype=float))) * 0.001)
    prominence = max(atr * PRICE_PIVOT_PROMINENCE_ATR, float(np.median(df["close"].to_numpy(dtype=float))) * 0.0025)

    if direction == "bearish":
        idxs, _ = find_peaks(highs, distance=PRICE_PIVOT_DISTANCE, prominence=prominence)
        price_values = highs
        choose = np.nanargmax
    else:
        idxs, _ = find_peaks(-lows, distance=PRICE_PIVOT_DISTANCE, prominence=prominence)
        price_values = lows
        choose = np.nanargmin

    points: list[dict[str, Any]] = []
    for idx in idxs.tolist():
        start = max(0, idx - PIVOT_MATCH_RADIUS)
        stop = min(len(df), idx + PIVOT_MATCH_RADIUS + 1)
        rsi_window = rsi[start:stop]
        if not np.isfinite(rsi_window).any():
            continue
        local_rel = int(choose(rsi_window))
        rsi_idx = start + local_rel
        points.append({
            "idx": int(idx),
            "price": float(price_values[idx]),
            "rsi_idx": int(rsi_idx),
            "rsi": float(rsi[rsi_idx]),
            "time_ms": int(times[idx]),
        })
    return points


def _segment_is_absolute(highs_or_lows: np.ndarray, i1: int, i2: int, direction: str) -> bool:
    segment = highs_or_lows[i1:i2 + 1]
    if direction == "bearish":
        return float(highs_or_lows[i2]) >= float(np.max(segment)) - 1e-12
    return float(highs_or_lows[i2]) <= float(np.min(segment)) + 1e-12


def _leg_valid(first: dict[str, Any], second: dict[str, Any], highs: np.ndarray, lows: np.ndarray, direction: str) -> bool:
    bars = second["idx"] - first["idx"]
    if bars < MIN_PIVOT_SEPARATION or bars > MAX_PIVOT_SEPARATION:
        return False
    if direction == "bearish":
        price_ok = second["price"] >= first["price"] * (1.0 + MIN_PRICE_DIFFERENCE_PCT)
        rsi_ok = second["rsi"] <= first["rsi"] - MIN_RSI_DIFFERENCE
        absolute_ok = _segment_is_absolute(highs, first["idx"], second["idx"], "bearish")
        return bool(price_ok and rsi_ok and absolute_ok)
    price_ok = second["price"] <= first["price"] * (1.0 - MIN_PRICE_DIFFERENCE_PCT)
    rsi_ok = second["rsi"] >= first["rsi"] + MIN_RSI_DIFFERENCE
    absolute_ok = _segment_is_absolute(lows, first["idx"], second["idx"], "bullish")
    return bool(price_ok and rsi_ok and absolute_ok)


def _seq_valid(sequence: list[dict[str, Any]], df: pd.DataFrame, direction: str) -> bool:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    for a, b in zip(sequence, sequence[1:]):
        if not _leg_valid(a, b, highs, lows, direction):
            return False
    return True


def _allowed_pivot_counts(timeframe: str) -> list[int]:
    if timeframe in {"4h", "1d"}:
        return [4, 3, 2]
    return [4, 3]


def detetar_divergencias_rsi(df: pd.DataFrame, timeframe: str) -> list[dict[str, Any]]:
    needed = {"timestamp", "high", "low", "close"}
    if len(df) < max(70, RSI_PERIOD + 20) or not needed.issubset(df.columns):
        return []
    data = df.reset_index(drop=True).copy()
    data["rsi"] = calcular_rsi(data["close"].to_numpy(dtype=float), RSI_PERIOD)
    times = data["timestamp"].to_numpy(dtype=np.int64)
    latest_closed_idx = len(data) - 1
    results: list[dict[str, Any]] = []

    for direction in ("bearish", "bullish"):
        pivots = _price_pivots(data, direction)
        if len(pivots) < 2:
            continue
        chosen_signal: dict[str, Any] | None = None
        for count in _allowed_pivot_counts(timeframe):
            if len(pivots) < count:
                continue
            # Procurar a sequência válida mais recente.
            for end in range(len(pivots) - 1, count - 2, -1):
                end_pivot = pivots[end]
                if latest_closed_idx - end_pivot["idx"] > MAX_SIGNAL_AGE_BARS:
                    continue
                if count == 2:
                    seq = [pivots[end - 1], pivots[end]]
                    if not _seq_valid(seq, data, direction):
                        continue
                    sequence = seq
                elif count == 3:
                    found = None
                    for mid in range(end - 1, 0, -1):
                        seq = [pivots[mid - 1], pivots[mid], pivots[end]]
                        if _seq_valid(seq, data, direction):
                            found = seq
                            break
                    if found is None:
                        continue
                    sequence = found
                else:
                    found = None
                    for c in range(end - 1, 1, -1):
                        for b in range(c - 1, 0, -1):
                            seq = [pivots[b - 1], pivots[b], pivots[c], pivots[end]]
                            if _seq_valid(seq, data, direction):
                                found = seq
                                break
                        if found is not None:
                            break
                    if found is None:
                        continue
                    sequence = found

                start = sequence[0]
                final = sequence[-1]
                price_change_pct = (final["price"] / start["price"] - 1.0) * 100.0
                rsi_change = final["rsi"] - start["rsi"]
                description = "HH no preço + LH no RSI" if direction == "bearish" else "LL no preço + HL no RSI"
                chosen_signal = {
                    "categoria": "rsi_divergence",
                    "tipo": direction,
                    "timeframe": timeframe,
                    "signal_time_ms": int(times[final["idx"]]),
                    "signal_time": iso_from_ms(int(times[final["idx"]])),
                    "idade_velas": int(latest_closed_idx - final["idx"]),
                    "pivots_count": len(sequence),
                    "forca_padrao": "3+ pivots" if len(sequence) >= 3 else "2 pivots",
                    "descricao_padrao": description,
                    "preco_inicial": float(start["price"]),
                    "preco_final": float(final["price"]),
                    "rsi_inicial": round(float(start["rsi"]), 2),
                    "rsi_final": round(float(final["rsi"]), 2),
                    "variacao_preco_pct": round(float(price_change_pct), 3),
                    "variacao_rsi": round(float(rsi_change), 2),
                    "pivots": [_point_for_ui(p["idx"], p["price"], p["rsi_idx"], p["rsi"], times) for p in sequence],
                    "id": f"rsi_{timeframe}_{direction}_{int(times[final['idx']])}_{len(sequence)}",
                    "status": "Divergência RSI",
                }
                break
            if chosen_signal is not None:
                break
        if chosen_signal is not None:
            results.append(chosen_signal)
    return results


def get_liquid_usdt_symbols(exchange: ccxt.Exchange, limit: int) -> list[str]:
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    excluded_suffixes = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")
    excluded_bases = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDE"}
    ranked: list[tuple[float, str]] = []
    extra_symbols = [s.strip().upper() for s in os.getenv("EXTRA_SYMBOLS", "").split(",") if s.strip()]
    extra_symbols_fmt = {s if "/" in s else f"{s[:-4]}/USDT" if s.endswith("USDT") else s for s in extra_symbols}

    for symbol, market in markets.items():
        if not symbol.endswith("/USDT"):
            continue
        if market.get("active") is False or market.get("spot") is False or market.get("contract") is True:
            continue
        base = str(market.get("base") or symbol.split("/")[0]).upper()
        if base in excluded_bases or base.endswith(excluded_suffixes):
            continue
        ticker = tickers.get(symbol) or {}
        try:
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            quote_volume = 0.0
        if quote_volume > 0:
            ranked.append((quote_volume, symbol))

    ranked.sort(reverse=True)
    symbols = [symbol for _, symbol in ranked[:limit]]
    for s in extra_symbols_fmt:
        if s and s not in symbols and s in markets:
            symbols.append(s)
    return symbols


def _telegram_endpoint(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def enviar_alerta_telegram(texto: str, foto_bytes: bytes | None = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram não configurado.")
        return False
    try:
        if foto_bytes is None:
            response = requests.post(
                _telegram_endpoint("sendMessage"),
                data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "disable_web_page_preview": "true"},
                timeout=20,
            )
        else:
            response = requests.post(
                _telegram_endpoint("sendPhoto"),
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": texto},
                files={"photo": ("alert.png", foto_bytes, "image/png")},
                timeout=30,
            )
        response.raise_for_status()
        if not response.json().get("ok"):
            log.error("Telegram recusou o alerta: %s", response.text[:500])
            return False
        return True
    except Exception:
        log.exception("Falha ao enviar alerta Telegram")
        return False


def formatar_alerta_rsi(signal: dict[str, Any]) -> str:
    icon = "🔻" if signal["tipo"] == "bearish" else "🟢"
    pivots = signal.get("pivots", [])
    prices = " → ".join(f"{float(p['price']):.8g}" for p in pivots)
    rsis = " → ".join(f"{float(p['rsi']):.1f}" for p in pivots)
    return "\n".join([
        f"{icon} RSI {signal['tipo'].upper()} | {signal['symbol']} {signal['timeframe']}",
        f"{signal['pivots_count']} pivots absolutos",
        f"Preço: {prices}",
        f"RSI: {rsis}",
        f"Confirmada: {signal['signal_time']}",
    ])


def formatar_alerta_breakout(signal: dict[str, Any]) -> str:
    if signal["fase"] == "Possível setup":
        return "\n".join([
            f"🚀 Possível setup | {signal['symbol']} {signal['timeframe']}",
            f"Zona: {signal['zona_min']:.8g} - {signal['zona_max']:.8g}",
            f"Rompimento sustentado acima do nível {signal['nivel']:.8g}",
            f"Toques: {signal['toques_resistencia']} | Vol. zona: {signal['volume_zona_ratio']}x | Vol. rompimento: {signal['volume_rompimento_ratio']}x",
            "Aguardando retrace para entrada.",
        ])
    trade = signal.get("trade") or {}
    return "\n".join([
        f"🎯 Entrada | {signal['symbol']} {signal['timeframe']}",
        f"Zona: {signal['zona_min']:.8g} - {signal['zona_max']:.8g}",
        f"Entrada: {trade.get('entry', 0):.8g} | SL: {trade.get('sl', 0):.8g} | TP: {trade.get('tp', 0):.8g}",
        "RR: 2.0 | Estado: Em curso",
    ])


def _plot_candles_base(df: pd.DataFrame, tail: int = 90):
    subset = df.tail(tail).copy()
    subset["timestamp"] = pd.to_datetime(subset["timestamp"], unit="ms", utc=True)
    if mpf is not None:
        indexed = subset.set_index("timestamp")
        mc = mpf.make_marketcolors(up="#2ecc71", down="#e74c3c", edge="inherit", wick="inherit", volume="inherit")
        style = mpf.make_mpf_style(marketcolors=mc, facecolor="#111827", figcolor="#111827", gridstyle="-", gridcolor="#374151")
        fig, axes = mpf.plot(indexed, type="candle", style=style, returnfig=True, volume=False, figsize=(9, 5))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111827")
        ax.set_facecolor("#111827")
        dates = mdates.date2num(subset["timestamp"].dt.to_pydatetime())
        width = 0.02 * (dates[-1] - dates[0]) / max(1, len(dates))
        for x, row in zip(dates, subset.itertuples(index=False)):
            color = "#2ecc71" if row.close >= row.open else "#e74c3c"
            ax.plot([x, x], [row.low, row.high], color=color, linewidth=1)
            lower = min(row.open, row.close)
            height = abs(row.close - row.open) or 1e-9
            ax.add_patch(Rectangle((x - width / 2, lower), width, height, color=color, alpha=0.9))
        ax.xaxis_date()
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#6b7280")
    return fig, ax


def gerar_grafico_breakout(df: pd.DataFrame, signal: dict[str, Any]) -> bytes | None:
    try:
        fig, ax = _plot_candles_base(df)
        ax.axhspan(signal["zona_min"], signal["zona_max"], color="#8b5cf6", alpha=0.18)
        ax.axhline(signal["nivel"], color="#a855f7", linestyle="--", linewidth=1.4)
        if signal["fase"] == "Entrada" and signal.get("trade"):
            trade = signal["trade"]
            ax.axhline(trade["entry"], color="#f59e0b", linestyle="--", linewidth=1.2)
            ax.axhline(trade["sl"], color="#ef4444", linestyle=":", linewidth=1.2)
            ax.axhline(trade["tp"], color="#22c55e", linestyle=":", linewidth=1.2)
            x0, x1 = ax.get_xlim()
            width = x1 - x0
            ax.add_patch(Rectangle((x1 - width * 0.16, trade["sl"]), width * 0.12, max(0, trade["entry"] - trade["sl"]), alpha=0.18, color="#ef4444"))
            ax.add_patch(Rectangle((x1 - width * 0.16, trade["entry"]), width * 0.12, max(0, trade["tp"] - trade["entry"]), alpha=0.18, color="#22c55e"))
        title = f"{signal['symbol']} {signal['timeframe']} - {signal['fase']}"
        ax.set_title(title, color="white")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#111827")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        log.exception("Falha a gerar gráfico do breakout")
        return None


def upsert_trade(trades: list[dict[str, Any]], signal: dict[str, Any]) -> None:
    trade = signal.get("trade") or {}
    if not trade:
        return
    trade_id = signal["id"]
    for row in trades:
        if row.get("trade_id") == trade_id:
            return
    trades.append({
        "trade_id": trade_id,
        "setup_id": signal.get("setup_id"),
        "symbol": signal["symbol"],
        "timeframe": signal["timeframe"],
        "categoria": signal["categoria"],
        "entrada": float(trade["entry"]),
        "sl": float(trade["sl"]),
        "tp": float(trade["tp"]),
        "rr": 2.0,
        "opened_at_ms": int(signal["signal_time_ms"]),
        "opened_at": signal["signal_time"],
        "status": "Em curso",
        "resultado_r": None,
        "closed_at_ms": None,
        "closed_at": None,
        "notas": "Sinal de entrada por reteste da resistência rompida.",
    })


def atualizar_estado_trades(trades: list[dict[str, Any]], market_cache: dict[tuple[str, str], pd.DataFrame]) -> None:
    for trade in trades:
        if trade.get("status") != "Em curso":
            continue
        key = (trade["symbol"], trade["timeframe"])
        df = market_cache.get(key)
        if df is None or df.empty:
            continue
        future = df[df["timestamp"] > int(trade["opened_at_ms"])]
        if future.empty:
            continue
        sl = float(trade["sl"])
        tp = float(trade["tp"])
        for _, row in future.iterrows():
            low = float(row["low"])
            high = float(row["high"])
            timestamp_ms = int(row["timestamp"])
            if low <= sl and high >= tp:
                trade["status"] = "Perdida"
                trade["resultado_r"] = -1.0
                trade["closed_at_ms"] = timestamp_ms
                trade["closed_at"] = iso_from_ms(timestamp_ms)
                trade["notas"] = "Na mesma vela tocou SL e TP; por prudência foi contabilizada como perda."
                break
            if low <= sl:
                trade["status"] = "Perdida"
                trade["resultado_r"] = -1.0
                trade["closed_at_ms"] = timestamp_ms
                trade["closed_at"] = iso_from_ms(timestamp_ms)
                break
            if high >= tp:
                trade["status"] = "Ganha"
                trade["resultado_r"] = 2.0
                trade["closed_at_ms"] = timestamp_ms
                trade["closed_at"] = iso_from_ms(timestamp_ms)
                break


def executar_scan() -> dict[str, Any]:
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    log.info("Scan iniciado | exchange=%s | timeframes=%s | top=%d pares | candles=%d", EXCHANGE_ID, TIMEFRAMES, SCAN_TOP_N, OHLCV_LIMIT)

    if ccxt is None:
        raise RuntimeError("A biblioteca ccxt não está instalada. Executa: pip install -r requirements.txt")
    factory = getattr(ccxt, EXCHANGE_ID, None)
    if factory is None:
        raise ValueError(f"Exchange CCXT desconhecida: {EXCHANGE_ID}")
    exchange = factory({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}})

    signals: list[dict[str, Any]] = []
    errors = 0
    ohlcv_ok = 0
    pair_timeframes_processed = 0
    symbols: list[str] = []
    supported_timeframes: list[str] = []
    market_cache: dict[tuple[str, str], pd.DataFrame] = {}
    chart_sources: dict[str, pd.DataFrame] = {}

    try:
        if getattr(exchange, "timeframes", None):
            supported_timeframes = [tf for tf in TIMEFRAMES if tf in exchange.timeframes]
        else:
            supported_timeframes = TIMEFRAMES[:]
        if not supported_timeframes:
            raise ValueError(f"Nenhum dos timeframes configurados é suportado por {EXCHANGE_ID}.")

        symbols = get_liquid_usdt_symbols(exchange, SCAN_TOP_N)
        log.info("Pares líquidos selecionados: %d", len(symbols))

        for symbol_index, symbol in enumerate(symbols, start=1):
            for timeframe in supported_timeframes:
                try:
                    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
                    if not raw or len(raw) < max(90, RSI_PERIOD + 20):
                        continue
                    full_df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    if len(full_df) < max(90, RSI_PERIOD + 20):
                        continue
                    forming = full_df.iloc[-1].to_dict()
                    df = full_df.iloc[:-1].reset_index(drop=True)
                    if len(df) < max(90, RSI_PERIOD + 20):
                        continue
                    market_cache[(symbol, timeframe)] = df.copy()
                    ohlcv_ok += 1
                    pair_timeframes_processed += 1

                    rsi_signals = detetar_divergencias_rsi(df, timeframe)
                    breakout_signals = detetar_setups_breakout(df, timeframe, forming_candle=forming)
                    for signal in rsi_signals + breakout_signals:
                        signal["symbol"] = symbol
                        signal["exchange"] = EXCHANGE_ID.upper()
                        signal["id"] = f"{EXCHANGE_ID}_{symbol.replace('/', '')}_{signal['id']}"
                        signals.append(signal)
                        chart_sources[signal["id"]] = df
                except Exception as exc:
                    errors += 1
                    log.warning("Falha em %s %s: %s", symbol, timeframe, exc)
            if symbol_index % 10 == 0:
                log.info("Progresso: %d/%d pares | pedidos OHLCV OK=%d | sinais=%d", symbol_index, len(symbols), ohlcv_ok, len(signals))
    finally:
        try:
            exchange.close()
        except Exception:
            pass

    unique = {signal["id"]: signal for signal in signals}
    signals = sorted(unique.values(), key=lambda item: item["signal_time_ms"], reverse=True)

    trades = read_trades()
    for signal in signals:
        if signal.get("categoria") == "breakout" and signal.get("fase") == "Entrada":
            upsert_trade(trades, signal)
    atualizar_estado_trades(trades, market_cache)
    save_trades(trades)

    payload = {
        "scan_started_at": started_at,
        "scan_completed_at": utc_now_iso(),
        "scan_seconds": round(time.monotonic() - started_monotonic, 1),
        "exchange": EXCHANGE_ID.upper(),
        "symbols_scanned": len(symbols),
        "timeframes": supported_timeframes,
        "ohlcv_requests_ok": ohlcv_ok,
        "pair_timeframes_processed": pair_timeframes_processed,
        "errors": errors,
        "signals_count": len(signals),
        "signals": signals,
        "configuration": {
            "timeframes": supported_timeframes,
            "rsi_period": RSI_PERIOD,
            "min_rsi_difference": MIN_RSI_DIFFERENCE,
            "min_price_difference_pct": MIN_PRICE_DIFFERENCE_PCT,
            "price_pivot_distance": PRICE_PIVOT_DISTANCE,
            "max_signal_age_bars": MAX_SIGNAL_AGE_BARS,
            "nota": "2h, 6h e 12h foram excluídos por completo.",
        },
    }
    write_json_atomic(LATEST_FILE, payload)

    sent_ids = read_sent_ids()
    new_alerts = 0
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        for signal in signals:
            if signal["id"] in sent_ids:
                continue
            if signal.get("categoria") == "rsi_divergence":
                ok = enviar_alerta_telegram(formatar_alerta_rsi(signal), None)
            else:
                chart = gerar_grafico_breakout(chart_sources.get(signal["id"], pd.DataFrame()), signal)
                ok = enviar_alerta_telegram(formatar_alerta_breakout(signal), chart)
            if ok:
                sent_ids.add(signal["id"])
                save_sent_ids(sent_ids)
                new_alerts += 1
    else:
        log.warning("Alertas Telegram desativados por falta de credenciais.")

    log.info(
        "Scan concluído | pares=%d | OHLCV OK=%d | pares/timeframes=%d | erros=%d | sinais=%d | novos alertas=%d | duração=%.1fs",
        len(symbols), ohlcv_ok, pair_timeframes_processed, errors, len(signals), new_alerts, payload["scan_seconds"],
    )
    return payload


def main() -> None:
    if "--once" in sys.argv:
        executar_scan()
        return
    log.info("Scanner iniciado; próxima varredura após cada ciclo de %d segundos.", SCAN_INTERVAL_SECONDS)
    while True:
        cycle_start = time.monotonic()
        try:
            executar_scan()
        except KeyboardInterrupt:
            log.info("Scanner interrompido.")
            return
        except Exception:
            log.exception("Falha global no scan; será tentado novamente no próximo ciclo.")
        wait_seconds = max(1.0, SCAN_INTERVAL_SECONDS - (time.monotonic() - cycle_start))
        log.info("Próximo scan em cerca de %.0f segundos.", wait_seconds)
        try:
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            log.info("Scanner interrompido.")
            return


if __name__ == "__main__":
    main()
