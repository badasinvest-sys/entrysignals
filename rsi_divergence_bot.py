"""Scanner RSI para MEXC/Binance, sem base de dados.

Executado pelo start.py no mesmo serviço Render que o painel Streamlit.
Também pode ser testado sozinho: python rsi_divergence_bot.py --once
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

try:
    import ccxt
except ImportError:  # Permite executar testes da lógica sem instalar o cliente de exchange.
    ccxt = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
from breakout_setup import detetar_setups_breakout, HOLD_BARS, ENTRY_ON_LIVE_CANDLE

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))
SENT_FILE = Path(os.getenv("SENT_FILE", str(BASE_DIR / "sent_signals_divergence.json")))

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "mexc").strip().lower()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SCAN_INTERVAL_SECONDS = max(60, int(os.getenv("SCAN_INTERVAL_SECONDS", "300")))
SCAN_TOP_N = max(1, int(os.getenv("SCAN_TOP_N", "35")))
OHLCV_LIMIT = max(220, min(1000, int(os.getenv("OHLCV_LIMIT", "360"))))
RSI_PERIOD = max(2, int(os.getenv("RSI_PERIOD", "14")))
MAX_SIGNAL_AGE_BARS = max(1, int(os.getenv("MAX_SIGNAL_AGE_BARS", "4")))
MIN_RSI_DIFFERENCE = max(0.0, float(os.getenv("MIN_RSI_DIFFERENCE", "2.0")))
MIN_PRICE_DIFFERENCE_PCT = max(0.0, float(os.getenv("MIN_PRICE_DIFFERENCE_PCT", "0.0005")))
PIVOT_DISTANCE = max(2, int(os.getenv("PIVOT_DISTANCE", "4")))
RSI_PIVOT_PROMINENCE = max(0.1, float(os.getenv("RSI_PIVOT_PROMINENCE", "2.0")))
PIVOT_MATCH_RADIUS = max(1, int(os.getenv("PIVOT_MATCH_RADIUS", "3")))
MIN_PIVOT_SEPARATION = max(2, int(os.getenv("MIN_PIVOT_SEPARATION", "5")))
MAX_PIVOT_SEPARATION = max(MIN_PIVOT_SEPARATION + 1, int(os.getenv("MAX_PIVOT_SEPARATION", "80")))
MAX_PIVOTS_IN_PATTERN = max(3, min(6, int(os.getenv("MAX_PIVOTS_IN_PATTERN", "4"))))
TIMEFRAMES = [x.strip() for x in os.getenv("TIMEFRAMES", "15m,30m,1h,2h,4h,6h,12h,1d").split(",") if x.strip()]
TWO_PIVOT_TIMEFRAMES = {"4h", "1d"}
EXTRA_SYMBOLS = [s.strip().upper() for s in os.getenv("EXTRA_SYMBOLS", "").split(",") if s.strip()]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rsi-divergence")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def write_json_atomic(path: Path, data: Any) -> None:
    """Evita que o painel leia um JSON parcialmente escrito."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_sent_ids() -> set[str]:
    try:
        if SENT_FILE.exists():
            data = json.loads(SENT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {str(item) for item in data}
    except Exception:
        log.exception("Não foi possível ler o ficheiro local de sinais já enviados.")
    return set()


def save_sent_ids(sent_ids: set[str]) -> None:
    # A lista é limitada para o ficheiro local não crescer sem controlo.
    write_json_atomic(SENT_FILE, sorted(sent_ids)[-15000:])


def calcular_rsi(closes: np.ndarray | list[float], period: int = 14) -> np.ndarray:
    """RSI de Wilder com alinhamento correto; valores iniciais indefinidos ficam NaN."""
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


def _build_swings(
    data: pd.DataFrame,
    rsi: np.ndarray,
    direction: str,
) -> list[dict[str, Any]]:
    """Liga cada pivot do RSI ao extremo de preço mais relevante numa janela próxima.

    Isto evita exigir que o topo/fundo do preço ocorra exatamente na mesma vela
    do topo/fundo do RSI, que era uma das limitações do código original.
    """
    first_valid = np.flatnonzero(np.isfinite(rsi))
    if first_valid.size == 0:
        return []
    valid_start = int(first_valid[0])
    clean_rsi = rsi[valid_start:]
    if direction == "bearish":
        pivot_idx, properties = find_peaks(
            clean_rsi, distance=PIVOT_DISTANCE, prominence=RSI_PIVOT_PROMINENCE
        )
        price_values = data["high"].to_numpy(dtype=float)
        choose = np.argmax
        rsi_is_top = True
    else:
        pivot_idx, properties = find_peaks(
            -clean_rsi, distance=PIVOT_DISTANCE, prominence=RSI_PIVOT_PROMINENCE
        )
        price_values = data["low"].to_numpy(dtype=float)
        choose = np.argmin
        rsi_is_top = False

    prominences = properties.get("prominences", np.ones(len(pivot_idx), dtype=float))
    timestamps = data["timestamp"].to_numpy(dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    for local_idx, prominence in zip(pivot_idx.tolist(), prominences.tolist()):
        rsi_idx = valid_start + int(local_idx)
        start = max(valid_start, rsi_idx - PIVOT_MATCH_RADIUS)
        stop = min(len(data), rsi_idx + PIVOT_MATCH_RADIUS + 1)
        if stop <= start:
            continue
        price_idx = start + int(choose(price_values[start:stop]))
        candidates.append({
            "rsi_idx": rsi_idx,
            "rsi": float(rsi[rsi_idx]),
            "price_idx": price_idx,
            "price": float(price_values[price_idx]),
            "prominence": float(prominence),
            "confirm_idx": max(rsi_idx, price_idx),
            "price_time_ms": int(timestamps[price_idx]),
            "rsi_time_ms": int(timestamps[rsi_idx]),
            "rsi_is_top": rsi_is_top,
        })

    # Se dois pivots do RSI apontam para o mesmo extremo de preço, conserva-se
    # aquele que está temporalmente mais perto; em empate, o de maior proeminência.
    by_price_index: dict[int, dict[str, Any]] = {}
    for item in candidates:
        old = by_price_index.get(item["price_idx"])
        score = (abs(item["rsi_idx"] - item["price_idx"]), -item["prominence"])
        old_score = None if old is None else (
            abs(old["rsi_idx"] - old["price_idx"]), -old["prominence"]
        )
        if old is None or score < old_score:
            by_price_index[item["price_idx"]] = item

    ordered = sorted(by_price_index.values(), key=lambda x: (x["confirm_idx"], x["price_idx"]))
    # Evita swings duplicados ou fora de ordem temporal por causa da janela de associação.
    result: list[dict[str, Any]] = []
    for item in ordered:
        if result and item["price_idx"] <= result[-1]["price_idx"]:
            continue
        if result and item["confirm_idx"] <= result[-1]["confirm_idx"]:
            continue
        result.append(item)
    return result


def _direct_divergence(first: dict[str, Any], last: dict[str, Any], kind: str) -> bool:
    if kind == "bearish":
        price_makes_higher_high = last["price"] >= first["price"] * (1.0 + MIN_PRICE_DIFFERENCE_PCT)
        rsi_makes_lower_high = last["rsi"] <= first["rsi"] - MIN_RSI_DIFFERENCE
        return bool(price_makes_higher_high and rsi_makes_lower_high)
    price_makes_lower_low = last["price"] <= first["price"] * (1.0 - MIN_PRICE_DIFFERENCE_PCT)
    rsi_makes_higher_low = last["rsi"] >= first["rsi"] + MIN_RSI_DIFFERENCE
    return bool(price_makes_lower_low and rsi_makes_higher_low)


def _divergent_leg(first: dict[str, Any], last: dict[str, Any], kind: str) -> bool:
    """Teste menos rígido usado apenas para contar pivots consecutivos do mesmo setup."""
    soft_price_threshold = max(MIN_PRICE_DIFFERENCE_PCT * 0.20, 0.0001)
    soft_rsi_threshold = max(MIN_RSI_DIFFERENCE * 0.25, 0.5)
    if kind == "bearish":
        return (
            last["price"] >= first["price"] * (1.0 + soft_price_threshold)
            and last["rsi"] <= first["rsi"] - soft_rsi_threshold
        )
    return (
        last["price"] <= first["price"] * (1.0 - soft_price_threshold)
        and last["rsi"] >= first["rsi"] + soft_rsi_threshold
    )


def _point_for_ui(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "price_time": iso_from_ms(point["price_time_ms"]),
        "rsi_time": iso_from_ms(point["rsi_time_ms"]),
        "price": float(point["price"]),
        "rsi": round(float(point["rsi"]), 2),
    }


def detetar_divergencias_rsi(
    df: pd.DataFrame,
    timeframe: str,
    max_age_bars: int = MAX_SIGNAL_AGE_BARS,
) -> list[dict[str, Any]]:
    """Deteta divergências: 3+ pivots em todos os TF; 2 só em 4h/1d.

    Bearish: preço HH (higher high) + RSI LH (lower high).
    Bullish: preço LL (lower low) + RSI HL (higher low).

    Para contar 3+ pivots, os swings intermédios têm de confirmar cada perna
    na mesma direção. Um padrão de 2 pivots nunca é admitido abaixo de 4h.
    """
    needed = {"timestamp", "high", "low", "close"}
    if len(df) < max(40, RSI_PERIOD + 10) or not needed.issubset(df.columns):
        return []
    data = df.reset_index(drop=True).copy()
    # A coluna RSI é útil para testes controlados; no scanner real é sempre calculada abaixo.
    rsi = (
        data["rsi"].to_numpy(dtype=float)
        if "rsi" in data.columns
        else calcular_rsi(data["close"].to_numpy(dtype=float), RSI_PERIOD)
    )
    if len(rsi) != len(data):
        return []

    latest_closed_idx = len(data) - 1
    detected: list[dict[str, Any]] = []
    for kind in ("bearish", "bullish"):
        swings = _build_swings(data, rsi, kind)
        if len(swings) < 2:
            continue

        # Só alertamos sobre a divergência mais recente e ainda útil neste timeframe.
        found_for_kind: dict[str, Any] | None = None
        for end_idx in range(len(swings) - 1, 0, -1):
            endpoint = swings[end_idx]
            age = latest_closed_idx - endpoint["confirm_idx"]
            if age > max_age_bars:
                break
            candidate_patterns: list[tuple[tuple[float, ...], list[dict[str, Any]]]] = []

            for start_idx in range(end_idx - 1, -1, -1):
                first = swings[start_idx]
                distance = endpoint["price_idx"] - first["price_idx"]
                if distance > MAX_PIVOT_SEPARATION:
                    break
                if distance < MIN_PIVOT_SEPARATION:
                    continue
                if not _direct_divergence(first, endpoint, kind):
                    continue

                path = swings[start_idx:end_idx + 1]
                # Contar múltiplos pivots só quando cada perna sucessiva confirma
                # a mesma direção; caso contrário, classificar como divergência de 2 pivots.
                consistent = all(
                    _divergent_leg(path[k], path[k + 1], kind)
                    for k in range(len(path) - 1)
                )
                chosen = path[-MAX_PIVOTS_IN_PATTERN:] if consistent else [first, endpoint]
                if len(chosen) > MAX_PIVOTS_IN_PATTERN:
                    chosen = chosen[-MAX_PIVOTS_IN_PATTERN:]
                # Exigir pelo menos 3 pivots fora de 4h e 1d.
                if len(chosen) < 3 and timeframe not in TWO_PIVOT_TIMEFRAMES:
                    continue
                # Não chamar multi-pivot se o recorte não conservar a diferença
                # mínima exigida entre o primeiro e o último pivot.
                if len(chosen) >= 3 and not _direct_divergence(chosen[0], chosen[-1], kind):
                    chosen = [first, endpoint]
                if len(chosen) < 3 and timeframe not in TWO_PIVOT_TIMEFRAMES:
                    continue
                rsi_delta = abs(float(chosen[-1]["rsi"]) - float(chosen[0]["rsi"]))
                price_delta_pct = abs(float(chosen[-1]["price"]) / float(chosen[0]["price"]) - 1.0)
                score = (float(len(chosen)), rsi_delta, price_delta_pct, -float(distance))
                candidate_patterns.append((score, chosen))

            if candidate_patterns:
                candidate_patterns.sort(key=lambda x: x[0], reverse=True)
                chosen = candidate_patterns[0][1]
                # Após limitar a sequência ao número configurado de pivots,
                # garante que o par inicial/final continua a cumprir os limiares.
                if not _direct_divergence(chosen[0], chosen[-1], kind):
                    chosen = [candidate_patterns[0][1][0], candidate_patterns[0][1][-1]]
                if len(chosen) < 3 and timeframe not in TWO_PIVOT_TIMEFRAMES:
                    continue
                timestamps = data["timestamp"].to_numpy(dtype=np.int64)
                final = chosen[-1]
                signal_idx = int(final["confirm_idx"])
                signal_ms = int(timestamps[signal_idx])
                start_point = chosen[0]
                price_change_pct = (final["price"] / start_point["price"] - 1.0) * 100.0
                rsi_change = final["rsi"] - start_point["rsi"]
                if kind == "bearish":
                    description = "Preço HH + RSI LH"
                else:
                    description = "Preço LL + RSI HL"
                detected.append({
                    "id": f"{timeframe}_{kind}_{signal_ms}",
                    "tipo": kind,
                    "timeframe": timeframe,
                    "status": "Divergência RSI " + kind,
                    "signal_time_ms": signal_ms,
                    "signal_time": iso_from_ms(signal_ms),
                    "idade_velas": int(latest_closed_idx - signal_idx),
                    "pivots_count": len(chosen),
                    "forca_padrao": "Multi-pivot" if len(chosen) >= 3 else "Clássica (2 pivots)",
                    "descricao_padrao": description,
                    "preco_inicial": float(start_point["price"]),
                    "preco_final": float(final["price"]),
                    "rsi_inicial": round(float(start_point["rsi"]), 2),
                    "rsi_final": round(float(final["rsi"]), 2),
                    "variacao_preco_pct": round(float(price_change_pct), 3),
                    "variacao_rsi": round(float(rsi_change), 2),
                    "pivots": [_point_for_ui(point) for point in chosen],
                })
                found_for_kind = detected[-1]
                break
        # Encontrar no máximo um sinal recente de cada direção por par/timeframe.
        if found_for_kind is None:
            continue
    return detected


def get_liquid_usdt_symbols(exchange: ccxt.Exchange, limit: int) -> list[str]:
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    excluded_suffixes = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")
    excluded_bases = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDE"}
    ranked: list[tuple[float, str]] = []

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
    for extra in EXTRA_SYMBOLS:
        # Pares colocados manualmente são sempre analisados mesmo fora do top.
        if extra not in symbols and extra in markets and markets[extra].get("spot", True):
            symbols.append(extra)
        elif extra not in markets:
            log.warning("EXTRA_SYMBOLS: par não disponível nesta exchange: %s", extra)
    return symbols




def agregar_velas_1h(raw_1h: list[list[float]], target_tf: str) -> pd.DataFrame:
    """Constrói TF 2h/6h/12h a partir de 1h, se a exchange não o suportar.

    Exclui a vela 1h atual e quaisquer grupos incompletos. Não inventa candles.
    O calendário é ancorado nas fronteiras UTC.
    """
    factor = {"2h": 2, "6h": 6, "12h": 12}.get(target_tf)
    if factor is None or len(raw_1h) < factor + 1:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw_1h[:-1], columns=["timestamp", "open", "high", "low", "close", "volume"])
    one_hour_ms = 60 * 60 * 1000
    group_ms = factor * one_hour_ms
    df["bucket"] = (df["timestamp"].astype("int64") // group_ms) * group_ms
    grouped = df.groupby("bucket", sort=True)
    counts = grouped["timestamp"].count()
    complete = grouped.agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    complete = complete.loc[counts == factor].reset_index().rename(columns={"bucket": "timestamp"})
    return complete[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def enviar_alerta_telegram(texto: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram não configurado: define TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID nas variáveis do Render.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "disable_web_page_preview": "true"},
            timeout=15,
        )
        response.raise_for_status()
        if not response.json().get("ok"):
            log.error("Telegram recusou o alerta: %s", response.text[:500])
            return False
        return True
    except Exception:
        log.exception("Falha ao enviar alerta para o Telegram.")
        return False


def formatar_alerta(signal: dict[str, Any]) -> str:
    if signal.get("categoria") == "breakout":
        fase = signal["fase"]
        header = "🚀 POSSÍVEL SETUP" if fase == "Possível setup" else "🎯 ENTRADA — RETESTE"
        warning = (
            "Resistência rompida; sustentação confirmada. Aguardar retrace para a zona."
            if fase == "Possível setup" else
            "Reteste em curso (vela aberta): o sinal pode ser invalidado antes do fecho."
            if signal.get("intrabar") else
            "Reteste da zona com recuperação do nível no fecho da vela."
        )
        return "\n".join([
            header,
            f"{signal['exchange']} | {signal['symbol']} | {signal['timeframe']}",
            f"Resistência: {signal['nivel']:.8g} | Zona: {signal['zona_min']:.8g}–{signal['zona_max']:.8g}",
            f"Preço observado: {signal['preco_atual']:.8g}",
            f"Toques anteriores: {signal['toques_resistencia']} | Volume zona: {signal['volume_zona_ratio']:.2f}x",
            f"Volume do rompimento: {signal['volume_rompimento_ratio']:.2f}x a mediana",
            f"Sustentação: {signal['velas_sustentacao']} velas FECHADAS após rompimento",
            f"Rompimento: {signal['breakout_time']} | Evento: {signal['signal_time']}",
            warning,
            "Sinal técnico experimental; não constitui recomendação financeira.",
        ])
    arrow = "🔻" if signal["tipo"] == "bearish" else "🟢"
    lines = [
        f"{arrow} DIVERGÊNCIA RSI {signal['tipo'].upper()} — {signal['pivots_count']} pivots",
        f"Exchange: {EXCHANGE_ID.upper()} | Par: {signal['symbol']} | Timeframe: {signal['timeframe']}",
        f"Padrão: {signal['descricao_padrao']} ({signal['forca_padrao']})",
        f"Preço: {signal['preco_inicial']:.8g} → {signal['preco_final']:.8g} ({signal['variacao_preco_pct']:+.3f}%)",
        f"RSI: {signal['rsi_inicial']:.2f} → {signal['rsi_final']:.2f} ({signal['variacao_rsi']:+.2f})",
        f"Pivot confirmado: {signal['signal_time']} | Idade: {signal['idade_velas']} vela(s)",
        "Swings:",
    ]
    for index, point in enumerate(signal["pivots"], start=1):
        lines.append(
            f"{index}) preço {float(point['price']):.8g} | RSI {float(point['rsi']):.2f} | {point['price_time']}"
        )
    lines.append("Alerta de divergência, não é confirmação de entrada. A lógica de entrada será definida numa fase posterior.")
    return "\n".join(lines)


def executar_scan() -> dict[str, Any]:
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    log.info(
        "Scan iniciado | exchange=%s | timeframes=%s | top=%d pares | candles=%d",
        EXCHANGE_ID, TIMEFRAMES, SCAN_TOP_N, OHLCV_LIMIT,
    )

    if ccxt is None:
        raise RuntimeError("A biblioteca ccxt não está instalada. Executa: pip install -r requirements.txt")
    factory = getattr(ccxt, EXCHANGE_ID, None)
    if factory is None:
        raise ValueError(f"Exchange CCXT desconhecida: {EXCHANGE_ID}")
    exchange = factory({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    })

    signals: list[dict[str, Any]] = []
    errors = 0
    ohlcv_ok = 0
    pair_timeframes_processed = 0
    symbols: list[str] = []
    supported_timeframes: list[str] = []

    try:
        fallback_timeframes: dict[str, str] = {}
        if getattr(exchange, "timeframes", None):
            for tf in TIMEFRAMES:
                if tf in exchange.timeframes:
                    supported_timeframes.append(tf)
                elif tf in ("2h", "6h", "12h") and "1h" in exchange.timeframes:
                    fallback_timeframes[tf] = "1h"
                    supported_timeframes.append(tf)
                else:
                    log.warning("Timeframe %s não suportado por %s (nem agregável); ignorado", tf, EXCHANGE_ID)
            if fallback_timeframes:
                log.info("Agregação de 1h para timeframes não nativos: %s", list(fallback_timeframes))
        else:
            supported_timeframes = TIMEFRAMES[:]
        if not supported_timeframes:
            raise ValueError(f"Nenhum dos timeframes configurados é suportado por {EXCHANGE_ID}.")

        symbols = get_liquid_usdt_symbols(exchange, SCAN_TOP_N)
        log.info("Pares líquidos selecionados: %d", len(symbols))

        for symbol_index, symbol in enumerate(symbols, start=1):
            base_1h = None
            for timeframe in supported_timeframes:
                try:
                    forming = None
                    if timeframe in fallback_timeframes:
                        # Um único pedido por par serve 2h, 6h e 12h agregados.
                        if base_1h is None:
                            base_1h = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=1000)
                        df = agregar_velas_1h(base_1h or [], timeframe)
                    else:
                        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
                        if not raw or len(raw) < max(40, RSI_PERIOD + 10):
                            continue
                        full_df = pd.DataFrame(
                            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                        )
                        # Vela em formação usada apenas no reteste do breakout.
                        forming = full_df.iloc[-1].to_dict()
                        df = full_df.iloc[:-1].reset_index(drop=True)
                    if len(df) < max(40, RSI_PERIOD + 10):
                        if timeframe in fallback_timeframes:
                            log.warning("Histórico 1h insuficiente para agregar %s %s: apenas %d velas fechadas", symbol, timeframe, len(df))
                        else:
                            log.debug("Dados insuficientes para %s %s: %d velas fechadas", symbol, timeframe, len(df))
                        continue
                    ohlcv_ok += 1
                    pair_timeframes_processed += 1
                    for signal in detetar_divergencias_rsi(df, timeframe):
                        signal["symbol"] = symbol
                        signal["exchange"] = EXCHANGE_ID.upper()
                        signal["categoria"] = "rsi"
                        signal["id"] = (
                            f"{EXCHANGE_ID}_{symbol.replace('/', '')}_{timeframe}_"
                            f"{signal['tipo']}_{signal['signal_time_ms']}"
                        )
                        signals.append(signal)
                    if timeframe in HOLD_BARS:
                        for signal in detetar_setups_breakout(df, timeframe, forming_candle=forming):
                            signal["symbol"] = symbol
                            signal["exchange"] = EXCHANGE_ID.upper()
                            signal["id"] = (
                                f"{EXCHANGE_ID}_{symbol.replace('/', '')}_" + signal["id"]
                            )
                            signals.append(signal)
                except Exception as exc:
                    errors += 1
                    log.warning("Falha em %s %s: %s", symbol, timeframe, exc)
            if symbol_index % 10 == 0:
                log.info("Progresso: %d/%d pares; candles OHLCV OK=%d; sinais=%d", symbol_index, len(symbols), ohlcv_ok, len(signals))
    finally:
        try:
            exchange.close()
        except Exception:
            pass

    # Deduplica sinais que partilhem o mesmo pivot, mantendo cada timeframe.
    unique = {signal["id"]: signal for signal in signals}
    signals = sorted(unique.values(), key=lambda item: item["signal_time_ms"], reverse=True)
    payload = {
        "scan_started_at": started_at,
        "scan_completed_at": utc_now_iso(),
        "scan_seconds": round(time.monotonic() - started_monotonic, 1),
        "exchange": EXCHANGE_ID.upper(),
        "symbols_scanned": len(symbols),
        "timeframes": supported_timeframes,
        "timeframes_aggregated": fallback_timeframes,
        "ohlcv_requests_ok": ohlcv_ok,
        "pair_timeframes_processed": pair_timeframes_processed,
        "errors": errors,
        "signals_count": len(signals),
        "rsi_count": sum(s.get("categoria") == "rsi" for s in signals),
        "breakout_possible_count": sum(s.get("fase") == "Possível setup" for s in signals),
        "breakout_entry_count": sum(s.get("fase") == "Entrada" for s in signals),
        "signals": signals,
        "configuration": {
            "rsi_period": RSI_PERIOD,
            "min_rsi_difference": MIN_RSI_DIFFERENCE,
            "min_price_difference_pct": MIN_PRICE_DIFFERENCE_PCT,
            "rsi_pivot_prominence": RSI_PIVOT_PROMINENCE,
            "pivot_match_radius": PIVOT_MATCH_RADIUS,
            "max_pivots_in_pattern": MAX_PIVOTS_IN_PATTERN,
            "max_signal_age_bars": MAX_SIGNAL_AGE_BARS,
            "regra_divergencia": "3+ pivots em todos os TF; 2 pivots apenas 4h e 1d",
            "regra_rompimento": "Só 4h/1d; resistência validada por volume + fechos acima + reteste",
            "velas_confirmacao": HOLD_BARS,
            "entrada_vela_aberta": ENTRY_ON_LIVE_CANDLE,
            "extra_symbols": EXTRA_SYMBOLS,
        },
    }
    write_json_atomic(LATEST_FILE, payload)

    sent_ids = read_sent_ids()
    new_alerts = 0
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        # Ordem cronológica: Possível setup antes da respetiva Entrada.
        for signal in sorted(signals, key=lambda s: s["signal_time_ms"]):
            if signal["id"] in sent_ids:
                continue
            if enviar_alerta_telegram(formatar_alerta(signal)):
                sent_ids.add(signal["id"])
                save_sent_ids(sent_ids)
                new_alerts += 1
    else:
        log.warning("Alertas Telegram desativados: define TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID nas variáveis do Render.")

    log.info(
        "Scan concluído | pares=%d | OHLCV OK=%d | pares/TF=%d | erros=%d | RSI=%d | Breakout provável=%d | Entradas=%d | novos alertas=%d | duração=%.1fs",
        len(symbols), ohlcv_ok, pair_timeframes_processed, errors, payload["rsi_count"],
        payload["breakout_possible_count"], payload["breakout_entry_count"], new_alerts, payload["scan_seconds"],
    )
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Os sinais são gravados, mas os alertas Telegram estão desativados por falta de credenciais.")
    if payload["scan_seconds"] > SCAN_INTERVAL_SECONDS:
        log.warning("A varredura demorou mais do que o intervalo configurado (%ds). Reduz SCAN_TOP_N ou TIMEFRAMES.", SCAN_INTERVAL_SECONDS)
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
