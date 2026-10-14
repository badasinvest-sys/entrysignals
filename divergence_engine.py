"""Divergências estritas: estrutura de preço + RSI na vela do extremo + rejeição.

A divergência multi-pivot tem de usar pivots estruturais CONSECUTIVOS.
É proibido saltar um pivot significativo que invalide qualquer perna.
O RSI é medido no fecho da vela que produziu o HIGH/LOW, nunca numa
outra vela à escolha dentro de uma janela de ±3.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ALLOWED_TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d")
RSI_PERIOD = max(2, int(os.getenv("RSI_PERIOD", "14")))
MIN_RSI_DELTA = max(0.5, float(os.getenv("MIN_RSI_DIFFERENCE", "3.0")))
MIN_PRICE_DELTA_PCT = max(0.001, float(os.getenv("MIN_PRICE_DIFFERENCE_PCT", "0.006")))
MIN_PRICE_DELTA_ATR = max(0.10, float(os.getenv("MIN_PRICE_DELTA_ATR", "0.60")))
PRICE_PIVOT_DISTANCE = max(3, int(os.getenv("PRICE_PIVOT_DISTANCE", "6")))
PRICE_PIVOT_PROMINENCE_ATR = max(0.2, float(os.getenv("PRICE_PIVOT_PROMINENCE_ATR", "0.9")))
RSI_LOCAL_TOLERANCE = max(0.0, float(os.getenv("RSI_LOCAL_TOLERANCE", "1.5")))
MAX_SIGNAL_AGE_BARS = max(2, int(os.getenv("MAX_SIGNAL_AGE_BARS", "6")))
MAX_PIVOT_SEPARATION = max(12, int(os.getenv("MAX_PIVOT_SEPARATION", "90")))
MIN_REJECTION_POINTS = max(1, int(os.getenv("MIN_REJECTION_POINTS", "2")))
REJECT_WICK_FRACTION = max(0.2, float(os.getenv("REJECT_WICK_FRACTION", "0.32")))


def iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def calcular_rsi(closes: np.ndarray | list[float], period: int = RSI_PERIOD) -> np.ndarray:
    prices = np.asarray(closes, dtype=float)
    result = np.full(prices.size, np.nan, dtype=float)
    if prices.size <= period or period < 2:
        return result
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def val(gain: float, loss: float) -> float:
        if gain == 0 and loss == 0:
            return 50.0
        if loss == 0:
            return 100.0
        if gain == 0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    result[period] = val(avg_gain, avg_loss)
    for j in range(period + 1, prices.size):
        avg_gain = (avg_gain * (period - 1) + gains[j - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[j - 1]) / period
        result[j] = val(avg_gain, avg_loss)
    return result


def atr_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.r_[cl[0], cl[:-1]]), np.abs(lo - np.r_[cl[0], cl[:-1]])))
    return pd.Series(tr).rolling(period, min_periods=3).median().bfill().to_numpy(float)


def candle_rejection(df: pd.DataFrame, idx: int, kind: str, atr: float) -> tuple[bool, bool, float]:
    row = df.iloc[idx]
    high, low, op, close = (float(row[k]) for k in ("high", "low", "open", "close"))
    rng = max(high - low, 1e-12)
    upper_wick = high - max(op, close)
    lower_wick = min(op, close) - low
    has_range = rng >= max(atr * 0.55, close * 0.0015)
    prior = df.iloc[max(0, idx - 12):idx]
    if len(prior) < 4:
        return False, False, 0.0
    if kind == "bearish":
        prior_level = float(prior["high"].max())
        wick_fraction = upper_wick / rng
        rejection = has_range and wick_fraction >= REJECT_WICK_FRACTION and (close - low) / rng <= 0.65
        sweep = high >= prior_level + max(atr * 0.08, close * 0.0005) and close < prior_level
    else:
        prior_level = float(prior["low"].min())
        wick_fraction = lower_wick / rng
        rejection = has_range and wick_fraction >= REJECT_WICK_FRACTION and (high - close) / rng <= 0.65
        sweep = low <= prior_level - max(atr * 0.08, close * 0.0005) and close > prior_level
    return bool(rejection or sweep), bool(sweep), float(wick_fraction)


def _pivot_rsi_valid(rsi: np.ndarray, idx: int, kind: str) -> bool:
    window = rsi[max(0, idx - 2): min(len(rsi), idx + 3)]
    if not np.all(np.isfinite(window)) or not np.isfinite(rsi[idx]):
        return False
    if kind == "bearish":
        return rsi[idx] >= float(np.max(window)) - RSI_LOCAL_TOLERANCE
    return rsi[idx] <= float(np.min(window)) + RSI_LOCAL_TOLERANCE


def structural_pivots(df: pd.DataFrame, rsi: np.ndarray, kind: str) -> list[dict[str, Any]]:
    highs, lows = df["high"].to_numpy(float), df["low"].to_numpy(float)
    atrs = atr_series(df)
    # Prominência expressa na escala DO PREÇO, não arbitrariamente em RSI.
    prominence = max(float(np.nanmedian(atrs[-80:])) * PRICE_PIVOT_PROMINENCE_ATR, float(np.nanmedian(df["close"])) * 0.004)
    prices = highs if kind == "bearish" else lows
    peaks, _ = find_peaks(prices if kind == "bearish" else -prices, distance=PRICE_PIVOT_DISTANCE, prominence=prominence)
    out = []
    for idx in peaks.tolist():
        if idx < RSI_PERIOD + 3 or idx >= len(df) - 2:
            continue
        # NÃO eliminar pivots de preço com RSI incompatível: esses pivots
        # têm de permanecer na sequência e INVALIDAR padrões que os ignorem.
        oscillator_pivot = _pivot_rsi_valid(rsi, idx, kind)
        rejection, sweep, wick = candle_rejection(df, idx, kind, float(atrs[idx]))
        out.append({
            "idx": int(idx),
            "price": float(prices[idx]),
            "rsi": float(rsi[idx]),
            "oscillator_pivot": bool(oscillator_pivot),
            "atr": float(atrs[idx]),
            "rejection": bool(rejection),
            "sweep": bool(sweep),
            "wick_pct": float(wick * 100),
            "time_ms": int(df.iloc[idx]["timestamp"]),
        })
    return out


def validate_leg(a: dict[str, Any], b: dict[str, Any], kind: str, price_values: np.ndarray) -> bool:
    i, j = int(a["idx"]), int(b["idx"])
    if j - i < PRICE_PIVOT_DISTANCE or j - i > MAX_PIVOT_SEPARATION:
        return False
    price_delta = (float(b["price"]) - float(a["price"])) * (1 if kind == "bearish" else -1)
    required_move = max(abs(float(a["price"])) * MIN_PRICE_DELTA_PCT, max(float(a["atr"]), float(b["atr"])) * MIN_PRICE_DELTA_ATR)
    if price_delta < required_move:
        return False
    rsi_delta = (float(a["rsi"]) - float(b["rsi"])) * (1 if kind == "bearish" else -1)
    if rsi_delta < MIN_RSI_DELTA:
        return False
    # B tem de ser o HIGH/LOW absoluto de toda a perna em causa.
    segment = price_values[i:j + 1]
    if kind == "bearish" and float(b["price"]) < float(np.max(segment)) - 1e-10:
        return False
    if kind == "bullish" and float(b["price"]) > float(np.min(segment)) + 1e-10:
        return False
    return True


def validate_sequence(seq: list[dict[str, Any]], df: pd.DataFrame, kind: str) -> bool:
    if len(seq) < 2:
        return False
    if not all(p.get("oscillator_pivot", False) for p in seq):
        return False
    prices = df["high"].to_numpy(float) if kind == "bearish" else df["low"].to_numpy(float)
    if not all(validate_leg(a, b, kind, prices) for a, b in zip(seq, seq[1:])):
        return False
    # O último pivot tem SEMPRE de representar rejeição ou sweep.
    if not seq[-1].get("rejection", False):
        return False
    rejection_count = sum(bool(p.get("rejection")) for p in seq)
    if rejection_count < min(len(seq), MIN_REJECTION_POINTS):
        return False
    return True


def detetar_divergencias_rsi(df: pd.DataFrame, timeframe: str, rsi_override: np.ndarray | None = None) -> list[dict[str, Any]]:
    if timeframe not in ALLOWED_TIMEFRAMES:
        return []
    needed = {"timestamp", "open", "high", "low", "close", "volume"}
    if len(df) < max(80, RSI_PERIOD + 20) or not needed.issubset(df.columns):
        return []
    data = df.reset_index(drop=True)
    rsi = np.asarray(rsi_override, dtype=float) if rsi_override is not None else calcular_rsi(data["close"].to_numpy(float))
    if len(rsi) != len(data):
        raise ValueError("RSI override tem de corresponder a uma entrada por vela.")
    result: list[dict[str, Any]] = []
    for kind in ("bearish", "bullish"):
        pivots = structural_pivots(data, rsi, kind)
        counts = (4, 3, 2) if timeframe in ("4h", "1d") else (4, 3)
        # Sem saltar swings: qualquer pivot estrutural intermédio invalida a
        # sequência se interromper a progressão do RSI ou do preço.
        found = None
        for count in counts:
            for end in range(len(pivots) - 1, count - 2, -1):
                seq = pivots[end - count + 1:end + 1]
                last = seq[-1]
                if len(data) - 1 - int(last["idx"]) > MAX_SIGNAL_AGE_BARS:
                    continue
                if not validate_sequence(seq, data, kind):
                    continue
                found = seq
                break
            if found is not None:
                break
        if found is None:
            continue
        first, last = found[0], found[-1]
        pivot_details = [
            {
                "price_idx": int(p["idx"]),
                "rsi_idx": int(p["idx"]),
                "price_time": iso_from_ms(p["time_ms"]),
                "rsi_time": iso_from_ms(p["time_ms"]),
                "price": float(p["price"]),
                "rsi": round(float(p["rsi"]), 2),
                "rejection": bool(p["rejection"]),
                "sweep": bool(p["sweep"]),
                "wick_pct": round(float(p["wick_pct"]), 1),
                "time_ms": int(p["time_ms"]),
            }
            for p in found
        ]
        result.append({
            "categoria": "rsi_divergence",
            "tipo": kind,
            "timeframe": timeframe,
            "signal_time_ms": int(last["time_ms"]),
            "signal_time": iso_from_ms(last["time_ms"]),
            "idade_velas": int(len(data) - 1 - int(last["idx"])),
            "pivots_count": len(found),
            "forca_padrao": "Multi-pivot confirmado" if len(found) >= 3 else "2 pivots",
            "descricao_padrao": "HH preço / LH RSI" if kind == "bearish" else "LL preço / HL RSI",
            "preco_inicial": float(first["price"]),
            "preco_final": float(last["price"]),
            "rsi_inicial": round(float(first["rsi"]), 2),
            "rsi_final": round(float(last["rsi"]), 2),
            "variacao_preco_pct": round((float(last["price"]) / float(first["price"]) - 1) * 100, 3),
            "variacao_rsi": round(float(last["rsi"]) - float(first["rsi"]), 2),
            "rejection_count": sum(int(p["rejection"]) for p in found),
            "sweep_count": sum(int(p["sweep"]) for p in found),
            "pivots": pivot_details,
            "id": f"rsi_{timeframe}_{kind}_{int(last['time_ms'])}_{len(found)}",
            "status": "Divergência RSI",
            "oi": {"status": "Por verificar", "confirmado": None},
        })
    return result
