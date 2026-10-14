from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _env_float(name: str, default: str, minimum: float = 0.0) -> float:
    return max(minimum, float(os.getenv(name, default)))


HOLD_BARS = {"4h": 3, "1d": 2}
HISTORY_BARS = max(90, int(os.getenv("BREAKOUT_HISTORY_BARS", "220")))
BREAKOUT_SEARCH_BARS = max(10, int(os.getenv("BREAKOUT_SEARCH_BARS", "42")))
RETEST_MAX_BARS = max(4, int(os.getenv("RETEST_MAX_BARS", "20")))
PIVOT_DISTANCE = max(3, int(os.getenv("RESISTANCE_PIVOT_DISTANCE", "6")))
MIN_RESISTANCE_TOUCHES = max(2, int(os.getenv("MIN_RESISTANCE_TOUCHES", "3")))
BREAKOUT_VOLUME_RATIO = _env_float("BREAKOUT_VOLUME_RATIO", "1.35", 1.0)
LEVEL_VOLUME_RATIO = _env_float("LEVEL_VOLUME_RATIO", "1.30", 1.0)
BREAKOUT_MAX_AGE_BARS = max(1, int(os.getenv("BREAKOUT_MAX_AGE_BARS", "3")))
ENTRY_ON_LIVE_CANDLE = os.getenv("ENTRY_ON_LIVE_CANDLE", "true").lower() in ("1", "true", "yes", "sim")
STOP_BUFFER_ATR = _env_float("STOP_BUFFER_ATR", "0.35", 0.1)
STOP_BUFFER_PCT = _env_float("STOP_BUFFER_PCT", "0.004", 0.001)


def _time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = (df[col].to_numpy(dtype=float) for col in ("high", "low", "close"))
    prior = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prior), np.abs(l - prior)))
    period = min(period, len(tr))
    return float(np.median(tr[-period:])) if period else 0.0


def _volume_profile_ratio(history: pd.DataFrame, level: float, zone_half: float) -> float:
    lo = history["low"].to_numpy(dtype=float)
    hi = history["high"].to_numpy(dtype=float)
    vol = history["volume"].to_numpy(dtype=float)
    floor, ceiling = float(np.min(lo)), float(np.max(hi))
    if ceiling <= floor or not np.isfinite(vol).all():
        return 0.0
    n_bins = 64
    edges = np.linspace(floor, ceiling, n_bins + 1)
    weights = np.zeros(n_bins, dtype=float)
    size = (ceiling - floor) / n_bins
    for low, high, volume in zip(lo, hi, vol):
        if volume <= 0:
            continue
        candle_range = max(high - low, size / 4.0)
        left = max(0, int((low - floor) / size))
        right = min(n_bins - 1, int((high - floor) / size))
        for bin_idx in range(left, right + 1):
            overlap = max(0.0, min(high, edges[bin_idx + 1]) - max(low, edges[bin_idx]))
            if overlap > 0:
                weights[bin_idx] += volume * overlap / candle_range
    active = weights[weights > 0]
    if len(active) < 8:
        return 0.0
    middle = (edges[:-1] + edges[1:]) / 2.0
    zone_bins = np.flatnonzero(np.abs(middle - level) <= max(zone_half, size))
    if zone_bins.size == 0:
        zone_bins = np.array([int(np.argmin(np.abs(middle - level)))])
    at_level = float(np.max(weights[zone_bins]))
    median = float(np.median(active))
    return at_level / max(median, 1e-12)


def encontrar_resistencias(history: pd.DataFrame) -> list[dict[str, Any]]:
    if len(history) < 80 or history["volume"].median() <= 0:
        return []
    highs = history["high"].to_numpy(dtype=float)
    closes = history["close"].to_numpy(dtype=float)
    vols = history["volume"].to_numpy(dtype=float)
    atr = max(atr_value(history), float(np.median(highs)) * 0.001)
    prominence = max(atr * 0.55, np.median(highs) * 0.0035)
    peak_idx, _ = find_peaks(highs, distance=PIVOT_DISTANCE, prominence=prominence)
    peak_idx = peak_idx[peak_idx <= len(history) - 4]
    if len(peak_idx) < MIN_RESISTANCE_TOUCHES:
        return []

    median_vol = float(np.median(vols))
    out: list[dict[str, Any]] = []
    used_levels: list[float] = []

    for candidate_i in reversed(peak_idx.tolist()):
        p = float(highs[candidate_i])
        half_width = min(p * 0.018, max(p * 0.004, atr * 0.30))
        cluster = [int(i) for i in peak_idx if abs(float(highs[i]) - p) <= half_width]
        if len(cluster) < MIN_RESISTANCE_TOUCHES:
            continue
        rejection_idxs = [i for i in cluster if closes[i] < p + half_width * 0.25]
        if len(rejection_idxs) < MIN_RESISTANCE_TOUCHES:
            continue
        rejection_idxs.sort()
        separated = [rejection_idxs[0]]
        for i in rejection_idxs[1:]:
            if i - separated[-1] >= PIVOT_DISTANCE:
                separated.append(i)
        if len(separated) < MIN_RESISTANCE_TOUCHES:
            continue
        level = float(np.median(highs[separated]))
        if any(abs(level - previous) <= half_width for previous in used_levels):
            continue
        relative_pivot_volume = float(np.mean(vols[separated])) / max(median_vol, 1e-12)
        profile_ratio = _volume_profile_ratio(history, level, half_width)
        if max(relative_pivot_volume, profile_ratio) < LEVEL_VOLUME_RATIO:
            continue
        out.append({
            "level": level,
            "zone_low": level - half_width,
            "zone_high": level + half_width,
            "touches": len(separated),
            "level_volume_ratio": round(max(relative_pivot_volume, profile_ratio), 2),
            "rejection_volume_ratio": round(relative_pivot_volume, 2),
            "volume_profile_ratio": round(profile_ratio, 2),
            "atr": atr,
        })
        used_levels.append(level)
    return out


def _trade_levels(level: float, zone_low: float, atr: float, trigger_price: float) -> tuple[float, float, float]:
    stop = min(zone_low - max(atr * STOP_BUFFER_ATR, level * STOP_BUFFER_PCT), trigger_price * 0.995)
    risk = max(trigger_price - stop, max(level * 0.0035, atr * 0.25))
    entry = float(trigger_price)
    sl = float(entry - risk)
    tp = float(entry + 2.0 * risk)
    return entry, sl, tp


def _as_event(
    kind: str,
    timeframe: str,
    zone: dict[str, Any],
    breakout_ms: int,
    event_ms: int,
    event_idx: int,
    latest_idx: int,
    breakout_volume_ratio: float,
    trigger_price: float,
    intrabar: bool = False,
) -> dict[str, Any]:
    phase = "Possível setup" if kind == "possible" else "Entrada"
    event = {
        "categoria": "breakout",
        "tipo": "bullish",
        "fase": phase,
        "status": phase,
        "timeframe": timeframe,
        "id": f"breakout_{timeframe}_{breakout_ms}_{kind}",
        "setup_id": f"breakout_{timeframe}_{breakout_ms}",
        "breakout_time_ms": int(breakout_ms),
        "signal_time_ms": int(event_ms),
        "signal_time": _time(int(event_ms)),
        "idade_velas": max(0, latest_idx - event_idx),
        "breakout_time": _time(int(breakout_ms)),
        "nivel": round(zone["level"], 12),
        "zona_min": round(zone["zone_low"], 12),
        "zona_max": round(zone["zone_high"], 12),
        "toques_resistencia": zone["touches"],
        "volume_zona_ratio": zone["level_volume_ratio"],
        "volume_rejeicoes_ratio": zone["rejection_volume_ratio"],
        "volume_profile_ratio": zone["volume_profile_ratio"],
        "volume_rompimento_ratio": round(breakout_volume_ratio, 2),
        "velas_sustentacao": HOLD_BARS[timeframe],
        "preco_atual": float(trigger_price),
        "intrabar": bool(intrabar),
        "trade": None,
    }
    if kind == "entry":
        entry, sl, tp = _trade_levels(zone["level"], zone["zone_low"], zone["atr"], float(trigger_price))
        event["trade"] = {
            "entry": round(entry, 12),
            "sl": round(sl, 12),
            "tp": round(tp, 12),
            "rr": 2.0,
        }
    return event


def detetar_setups_breakout(
    closed_df: pd.DataFrame,
    timeframe: str,
    forming_candle: dict[str, float] | list[float] | None = None,
    max_age_bars: int = BREAKOUT_MAX_AGE_BARS,
) -> list[dict[str, Any]]:
    if timeframe not in HOLD_BARS:
        return []
    needed = {"timestamp", "open", "high", "low", "close", "volume"}
    if not needed.issubset(closed_df.columns) or len(closed_df) < 110:
        return []

    df = closed_df.reset_index(drop=True)
    times = df["timestamp"].to_numpy(dtype=np.int64)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    vols = df["volume"].to_numpy(dtype=float)
    n = len(df)
    hold_count = HOLD_BARS[timeframe]
    signals: list[dict[str, Any]] = []

    first_breakout = max(90, n - BREAKOUT_SEARCH_BARS)
    for b in range(first_breakout, n - hold_count):
        previous = df.iloc[max(0, b - HISTORY_BARS):b]
        if len(previous) < 80:
            continue
        median20 = float(np.median(vols[max(0, b - 20):b]))
        if median20 <= 0:
            continue
        ratio_break = float(vols[b] / median20)
        if ratio_break < BREAKOUT_VOLUME_RATIO:
            continue
        zones = encontrar_resistencias(previous)
        eligible = [
            z for z in zones
            if closes[b - 1] <= z["zone_high"]
            and closes[b] > z["zone_high"]
            and highs[b] >= z["zone_high"]
        ]
        if not eligible:
            continue
        z = max(eligible, key=lambda v: (v["level"], v["touches"], v["level_volume_ratio"]))
        hold_end = b + hold_count
        if not np.all(closes[b + 1:hold_end + 1] > z["level"]):
            continue
        possible_age = (n - 1) - hold_end
        still_supported = not np.any(closes[hold_end + 1:] < z["zone_low"])
        if 0 <= possible_age <= max_age_bars and still_supported:
            signals.append(_as_event(
                "possible", timeframe, z, int(times[b]), int(times[hold_end]),
                hold_end, n - 1, ratio_break, float(closes[hold_end]),
            ))

        min_away = max(z["atr"] * 0.5, z["level"] * 0.006)
        last_candidate = min(n - 1, hold_end + RETEST_MAX_BARS)
        highest_since_breakout = float(np.max(highs[b:hold_end + 1]))
        entry_emitted = False
        for i in range(hold_end + 1, last_candidate + 1):
            if closes[i] < z["zone_low"]:
                break
            if highest_since_breakout >= z["zone_high"] + min_away:
                touch = (lows[i] <= z["zone_high"] * 1.0025) and (highs[i] >= z["zone_low"])
                reclaim = closes[i] >= z["level"]
                shallow_retrace = lows[i] >= z["zone_low"] - 0.0035 * z["level"]
                if touch and reclaim and shallow_retrace:
                    if n - 1 - i <= max_age_bars and not np.any(closes[i + 1:] < z["zone_low"]):
                        signals.append(_as_event(
                            "entry", timeframe, z, int(times[b]), int(times[i]),
                            i, n - 1, ratio_break, float(closes[i]),
                        ))
                    entry_emitted = True
                    break
            highest_since_breakout = max(highest_since_breakout, float(highs[i]))

        if entry_emitted or not ENTRY_ON_LIVE_CANDLE or forming_candle is None:
            continue
        if n - 1 >= hold_end + RETEST_MAX_BARS or last_candidate < n - 1:
            continue
        if np.any(closes[hold_end + 1:] < z["zone_low"]):
            continue
        last_candle = dict(zip(("timestamp", "open", "high", "low", "close", "volume"), forming_candle)) if isinstance(forming_candle, (tuple, list)) else forming_candle
        try:
            live_low = float(last_candle["low"])
            live_high = float(last_candle["high"])
            live_price = float(last_candle["close"])
            live_time = int(last_candle["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (live_price >= z["level"] and live_price <= z["zone_high"] * 1.006):
            continue
        if not (z["zone_low"] - 0.0035 * z["level"] <= live_low <= z["zone_high"] * 1.003 and live_high >= z["zone_low"]):
            continue
        if highest_since_breakout < z["zone_high"] + min_away:
            continue
        signals.append(_as_event(
            "entry", timeframe, z, int(times[b]), live_time,
            n, n, ratio_break, live_price, intrabar=True,
        ))

    return sorted({signal["id"]: signal for signal in signals}.values(), key=lambda s: s["signal_time_ms"])
