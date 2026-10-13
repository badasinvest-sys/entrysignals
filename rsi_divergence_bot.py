"""Scanner de divergências RSI para Binance Spot, preparado para Render Background Worker.

Executar uma vez: python rsi_divergence_bot.py --once
Executar em contínuo: python rsi_divergence_bot.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = Path(os.getenv("SENT_FILE", str(BASE_DIR / "sent_signals_divergence.json")))
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SCAN_INTERVAL_SECONDS = max(60, int(os.getenv("SCAN_INTERVAL_SECONDS", "300")))
SCAN_TOP_N = max(1, int(os.getenv("SCAN_TOP_N", "75")))
OHLCV_LIMIT = max(120, min(500, int(os.getenv("OHLCV_LIMIT", "250"))))
RSI_PERIOD = max(2, int(os.getenv("RSI_PERIOD", "14")))
MAX_SIGNAL_AGE_BARS = max(2, int(os.getenv("MAX_SIGNAL_AGE_BARS", "5")))
MIN_RSI_DIFFERENCE = float(os.getenv("MIN_RSI_DIFFERENCE", "2.0"))
MIN_PRICE_DIFFERENCE_PCT = float(os.getenv("MIN_PRICE_DIFFERENCE_PCT", "0.0005"))
PIVOT_DISTANCE = max(2, int(os.getenv("PIVOT_DISTANCE", "4")))
RSI_PIVOT_RADIUS = max(1, int(os.getenv("RSI_PIVOT_RADIUS", "2")))
TIMEFRAMES = [x.strip() for x in os.getenv(
    "TIMEFRAMES", "15m,30m,1h,2h,4h,6h,12h,1d"
).split(",") if x.strip()]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rsi-divergence")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: Any) -> None:
    """Writes JSON atomically so the dashboard never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _db_connection():
    """Abre Postgres partilhado entre o Worker e o painel, se DATABASE_URL existir."""
    import psycopg
    conn = psycopg.connect(DATABASE_URL, connect_timeout=10)
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS rsi_divergence_store ("
            "key TEXT PRIMARY KEY, value JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    conn.commit()
    return conn


def read_sent_ids() -> set[str]:
    if DATABASE_URL:
        with _db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM rsi_divergence_store WHERE key = %s", ("sent_signal_ids",))
                row = cur.fetchone()
        if row and isinstance(row[0], list):
            return {str(x) for x in row[0]}
        return set()
    try:
        if STATE_FILE.exists():
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return {str(x) for x in raw}
    except Exception:
        log.exception("Não foi possível ler o estado dos alertas (%s).", STATE_FILE)
    return set()


def save_sent_ids(ids: set[str]) -> None:
    # Limita o estado para não crescer indefinidamente.
    limited = sorted(ids)[-15000:]
    write_json_atomic(STATE_FILE, limited)
    if DATABASE_URL:
        from psycopg.types.json import Jsonb
        with _db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rsi_divergence_store (key, value, updated_at) VALUES (%s, %s, now()) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                    ("sent_signal_ids", Jsonb(limited)),
                )
            conn.commit()


def save_latest_payload(payload: dict[str, Any]) -> None:
    write_json_atomic(LATEST_FILE, payload)
    if DATABASE_URL:
        from psycopg.types.json import Jsonb
        with _db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rsi_divergence_store (key, value, updated_at) VALUES (%s, %s, now()) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                    ("latest_scan", Jsonb(payload)),
                )
            conn.commit()


def calcular_rsi(closes: np.ndarray | list[float], period: int = 14) -> np.ndarray:
    """RSI de Wilder, alinhado corretamente com as velas; os primeiros valores são NaN."""
    closes = np.asarray(closes, dtype=float)
    out = np.full(closes.size, np.nan, dtype=float)
    if closes.size <= period or period < 2:
        return out

    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def value(gain: float, loss: float) -> float:
        if loss == 0.0 and gain == 0.0:
            return 50.0
        if loss == 0.0:
            return 100.0
        if gain == 0.0:
            return 0.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = value(avg_gain, avg_loss)
    for close_index in range(period + 1, closes.size):
        delta_index = close_index - 1
        avg_gain = ((avg_gain * (period - 1)) + gains[delta_index]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[delta_index]) / period
        out[close_index] = value(avg_gain, avg_loss)
    return out


def _rsi_extreme_near(rsi: np.ndarray, index: int, direction: str, radius: int) -> tuple[float, int] | None:
    """Obtém o máximo/mínimo do RSI numa pequena janela em redor do pivot do preço."""
    start = max(0, index - radius)
    stop = min(len(rsi), index + radius + 1)
    segment = rsi[start:stop]
    valid = np.flatnonzero(np.isfinite(segment))
    if valid.size == 0:
        return None
    values = segment[valid]
    local_pos = int(np.argmax(values) if direction == "high" else np.argmin(values))
    chosen_index = start + int(valid[local_pos])
    return float(rsi[chosen_index]), chosen_index


def _make_signal(
    df: pd.DataFrame,
    timeframe: str,
    kind: str,
    first_idx: int,
    second_idx: int,
    rsi_first: tuple[float, int],
    rsi_second: tuple[float, int],
) -> dict[str, Any] | None:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    entry = float(closes[-1])  # preço da última vela fechada, não o preço antigo do pivot
    if kind == "bearish":
        sl = float(max(highs[first_idx], highs[second_idx]) * 1.015)
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - (2.0 * risk)
        status = "Divergência RSI bearish"
    else:
        sl = float(min(lows[first_idx], lows[second_idx]) * 0.985)
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + (2.0 * risk)
        status = "Divergência RSI bullish"

    pivot_ms = int(timestamps[second_idx])
    first_ms = int(timestamps[first_idx])
    return {
        "id": f"{timeframe}_{kind}_{pivot_ms}",  # o símbolo é adicionado pelo scanner
        "tipo": kind,
        "timeframe": timeframe,
        "status": status,
        "signal_time_ms": pivot_ms,
        "signal_time": datetime.fromtimestamp(pivot_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "previous_pivot_time": datetime.fromtimestamp(first_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "preco_pivot_anterior": float(highs[first_idx] if kind == "bearish" else lows[first_idx]),
        "preco_pivot_atual": float(highs[second_idx] if kind == "bearish" else lows[second_idx]),
        "rsi_anterior": round(rsi_first[0], 2),
        "rsi_atual": round(rsi_second[0], 2),
        "rsi_pivot_anterior_ms": int(df["timestamp"].iloc[rsi_first[1]]),
        "rsi_pivot_atual_ms": int(df["timestamp"].iloc[rsi_second[1]]),
        "idade_velas": int(len(df) - 1 - second_idx),
    }


def detetar_divergencias_rsi(
    df: pd.DataFrame,
    timeframe: str,
    max_age_bars: int = MAX_SIGNAL_AGE_BARS,
) -> list[dict[str, Any]]:
    """Deteta divergências regulares bullish e bearish confirmadas em pivots de preço.

    O extremo do RSI é procurado numa janela pequena junto de cada pivot do preço,
    evitando exigir que os dois extremos ocorram exatamente na mesma vela.
    """
    required = {"timestamp", "high", "low", "close"}
    if len(df) < max(40, RSI_PERIOD + 10) or not required.issubset(df.columns):
        return []

    data = df.reset_index(drop=True).copy()
    rsi = calcular_rsi(data["close"].to_numpy(dtype=float), RSI_PERIOD)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    last_idx = len(data) - 1
    signals: list[dict[str, Any]] = []

    price_highs, _ = find_peaks(highs, distance=PIVOT_DISTANCE)
    price_lows, _ = find_peaks(-lows, distance=PIVOT_DISTANCE)
    min_sep = max(5, 2 * RSI_PIVOT_RADIUS + 1)
    max_sep = 80

    # Bearish regular: preço faz higher high, RSI faz lower high.
    for p2 in reversed(price_highs.tolist()):
        age = last_idx - p2
        if age > max_age_bars:
            break  # os restantes pivots são ainda mais antigos
        if p2 < RSI_PERIOD:
            continue
        rsi2 = _rsi_extreme_near(rsi, p2, "high", RSI_PIVOT_RADIUS)
        if rsi2 is None:
            continue
        for p1 in reversed(price_highs.tolist()):
            if p1 >= p2:
                continue
            distance = p2 - p1
            if distance < min_sep:
                continue
            if distance > max_sep:
                break
            rsi1 = _rsi_extreme_near(rsi, p1, "high", RSI_PIVOT_RADIUS)
            if rsi1 is None:
                continue
            higher_high = highs[p2] >= highs[p1] * (1.0 + MIN_PRICE_DIFFERENCE_PCT)
            lower_rsi_high = rsi2[0] <= rsi1[0] - MIN_RSI_DIFFERENCE
            if higher_high and lower_rsi_high:
                signal = _make_signal(data, timeframe, "bearish", p1, p2, rsi1, rsi2)
                if signal:
                    signals.append(signal)
                    break
        if signals and signals[-1]["tipo"] == "bearish":
            break

    # Bullish regular: preço faz lower low, RSI faz higher low.
    bullish_found = False
    for p2 in reversed(price_lows.tolist()):
        age = last_idx - p2
        if age > max_age_bars:
            break
        if p2 < RSI_PERIOD:
            continue
        rsi2 = _rsi_extreme_near(rsi, p2, "low", RSI_PIVOT_RADIUS)
        if rsi2 is None:
            continue
        for p1 in reversed(price_lows.tolist()):
            if p1 >= p2:
                continue
            distance = p2 - p1
            if distance < min_sep:
                continue
            if distance > max_sep:
                break
            rsi1 = _rsi_extreme_near(rsi, p1, "low", RSI_PIVOT_RADIUS)
            if rsi1 is None:
                continue
            lower_low = lows[p2] <= lows[p1] * (1.0 - MIN_PRICE_DIFFERENCE_PCT)
            higher_rsi_low = rsi2[0] >= rsi1[0] + MIN_RSI_DIFFERENCE
            if lower_low and higher_rsi_low:
                signal = _make_signal(data, timeframe, "bullish", p1, p2, rsi1, rsi2)
                if signal:
                    signals.append(signal)
                    bullish_found = True
                    break
        if bullish_found:
            break

    return signals


def get_liquid_usdt_symbols(exchange: ccxt.Exchange, limit: int) -> list[str]:
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    candidates: list[tuple[float, str]] = []
    for symbol, market in markets.items():
        if not symbol.endswith("/USDT"):
            continue
        if market.get("active") is False or market.get("spot") is False:
            continue
        ticker = tickers.get(symbol) or {}
        try:
            volume = float(ticker.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume > 0:
            candidates.append((volume, symbol))
    candidates.sort(reverse=True)
    return [symbol for _, symbol in candidates[:limit]]


def enviar_alerta_telegram(texto: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram não configurado; alerta não enviado. Define TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "disable_web_page_preview": "true"},
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            log.error("Telegram recusou o alerta: %s", result)
            return False
        return True
    except Exception:
        log.exception("Falha ao enviar alerta para o Telegram.")
        return False


def formatar_alerta(signal: dict[str, Any]) -> str:
    arrow = "🔻" if signal["tipo"] == "bearish" else "🟢"
    return (
        f"{arrow} DIVERGÊNCIA RSI {signal['tipo'].upper()}\n"
        f"Par: {signal['symbol']} | Timeframe: {signal['timeframe']}\n"
        f"Pivot confirmado: {signal['signal_time']}\n"
        f"Preço atual/entrada indicativa: {signal['entry']:.8g}\n"
        f"RSI anterior → atual: {signal['rsi_anterior']:.2f} → {signal['rsi_atual']:.2f}\n"
        f"Stop indicativo: {signal['sl']:.8g}\n"
        f"Target 2R indicativo: {signal['tp']:.8g}\n"
        "Nota: sinal técnico, não é garantia de reversão nem recomendação financeira."
    )


def executar_scan() -> dict[str, Any]:
    started = utc_now_iso()
    log.info("Início do scan | timeframes=%s | top=%s pares | velas=%s", TIMEFRAMES, SCAN_TOP_N, OHLCV_LIMIT)
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    })
    signals: list[dict[str, Any]] = []
    errors = 0
    requests_ok = 0
    symbols: list[str] = []
    try:
        symbols = get_liquid_usdt_symbols(exchange, SCAN_TOP_N)
        log.info("Pares líquidos selecionados: %s", len(symbols))
        for symbol in symbols:
            for timeframe in TIMEFRAMES:
                try:
                    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
                    if len(raw) < max(40, RSI_PERIOD + 10):
                        continue
                    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    # Exclui a vela em formação: evita divergências que desaparecem antes do fecho.
                    df = df.iloc[:-1].reset_index(drop=True)
                    if len(df) < max(40, RSI_PERIOD + 10):
                        continue
                    requests_ok += 1
                    detected = detetar_divergencias_rsi(df, timeframe)
                    for signal in detected:
                        signal["symbol"] = symbol
                        signal["id"] = f"{symbol.replace('/', '')}_{timeframe}_{signal['tipo']}_{signal['signal_time_ms']}"
                        signals.append(signal)
                except Exception as exc:
                    errors += 1
                    log.warning("Falha em %s %s: %s", symbol, timeframe, exc)
    finally:
        try:
            exchange.close()
        except Exception:
            pass

    # Deduplica só pelo pivot confirmado; preserva sinais em timeframes diferentes.
    unique: dict[str, dict[str, Any]] = {}
    for signal in signals:
        unique[signal["id"]] = signal
    signals = sorted(unique.values(), key=lambda s: s["signal_time_ms"], reverse=True)

    payload = {
        "scan_started_at": started,
        "scan_completed_at": utc_now_iso(),
        "symbols_scanned": len(symbols),
        "timeframes": TIMEFRAMES,
        "ohlcv_requests_ok": requests_ok,
        "errors": errors,
        "signals_count": len(signals),
        "signals": signals,
    }
    save_latest_payload(payload)

    sent_ids = read_sent_ids()
    new_alerts = 0
    for signal in signals:
        if signal["id"] in sent_ids:
            continue
        if enviar_alerta_telegram(formatar_alerta(signal)):
            sent_ids.add(signal["id"])
            new_alerts += 1
            # Persistir logo após cada envio para reduzir duplicações em reinícios.
            save_sent_ids(sent_ids)
    log.info(
        "Scan concluído | pares=%d | pedidos OHLCV OK=%d | erros=%d | divergências=%d | novos alertas=%d",
        len(symbols), requests_ok, errors, len(signals), new_alerts,
    )
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("O scanner deteta e grava sinais, mas os alertas estão desativados por falta de credenciais Telegram.")
    return payload


def main() -> None:
    run_once = "--once" in sys.argv
    if run_once:
        executar_scan()
        return

    log.info("Worker iniciado. Um scan a cada %d segundos.", SCAN_INTERVAL_SECONDS)
    next_run = time.monotonic()
    while True:
        started = time.monotonic()
        try:
            executar_scan()
        except KeyboardInterrupt:
            log.info("Worker interrompido pelo utilizador.")
            break
        except Exception:
            log.exception("Erro não recuperável durante o scan; o worker continuará no próximo ciclo.")
        next_run += SCAN_INTERVAL_SECONDS
        # Se o scan demorar demasiado, inicia o próximo ciclo sem acumular atrasos.
        wait = max(1.0, next_run - time.monotonic())
        if time.monotonic() - started > SCAN_INTERVAL_SECONDS:
            log.warning("O scan demorou mais do que o intervalo de %d segundos.", SCAN_INTERVAL_SECONDS)
        log.info("Próximo scan em aproximadamente %.0f segundos.", wait)
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            log.info("Worker interrompido pelo utilizador.")
            break


if __name__ == "__main__":
    main()
