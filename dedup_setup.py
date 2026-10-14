"""Identidade semântica de setups: tolera pequenas mudanças na zona recalculada.

A mesma rutura/entrada não deve criar trades ou alertas novos quando a
resistência estimada difere alguns ticks. Setup independente por mercado,
par, timeframe e fase. Não elimina setups distantes no tempo.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
MAX_ZONE_RELATIVE_GAP = 0.012  # 1,2% da zona para compensar recalculações pequenas.


def _ms(iso: str) -> int | None:
    try:
        return int(datetime.strptime(iso, "%Y-%m-%d %H:%M UTC").timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _breakout_ms_from_id(value: str) -> int | None:
    try:
        tokens = str(value).split("_")
        idx = tokens.index("breakout")
        return int(tokens[idx + 2])
    except (ValueError, IndexError):
        return None


def _market_value(row: dict[str, Any]) -> str:
    if row.get("market_type"):
        return str(row["market_type"])
    # A V6 não guardava o mercado; o histórico anterior pertence à mesma
    # instalação corrente. Uma troca spot/swap deve limpar os JSON antigos.
    return ""


def _level(row: dict[str, Any]) -> float:
    return float(row.get("nivel") or (float(row.get("zona_min", 0)) + float(row.get("zona_max", 0))) / 2.0)


def same_setup(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first.get("symbol") != second.get("symbol") or first.get("timeframe") != second.get("timeframe"):
        return False
    am, bm = _market_value(first), _market_value(second)
    if am and bm and am != bm:
        return False
    a_ms = first.get("breakout_time_ms") or _breakout_ms_from_id(str(first.get("setup_id") or first.get("trade_id") or first.get("id") or ""))
    b_ms = second.get("breakout_time_ms") or _breakout_ms_from_id(str(second.get("setup_id") or second.get("trade_id") or second.get("id") or ""))
    if a_ms is None:
        a_ms = _ms(str(first.get("breakout_time", "")))
    if b_ms is None:
        b_ms = _ms(str(second.get("breakout_time", "")))
    if a_ms is None or b_ms is None:
        # Sem data do breakout, só IDs idênticos permitem equivalência segura.
        return bool(first.get("id") and first.get("id") == second.get("id"))
    period = TF_MS.get(str(first.get("timeframe")), 0)
    if not period or abs(int(a_ms) - int(b_ms)) > period:
        return False
    try:
        a_level, b_level = _level(first), _level(second)
        if a_level <= 0 or b_level <= 0:
            return False
        return abs(a_level - b_level) <= MAX_ZONE_RELATIVE_GAP * min(a_level, b_level)
    except (TypeError, ValueError):
        return False


def same_phase(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first.get("fase") == second.get("fase") and same_setup(first, second)


def deduplicate_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mantém o sinal mais antigo de cada fase, sem perder RSI."""
    output: list[dict[str, Any]] = []
    for signal in sorted(signals, key=lambda s: s.get("signal_time_ms", 0)):
        if signal.get("categoria") == "breakout" and any(
            other.get("categoria") == "breakout" and same_phase(signal, other)
            for other in output
        ):
            continue
        output.append(signal)
    return sorted(output, key=lambda s: s.get("signal_time_ms", 0), reverse=True)


def deduplicate_trades(trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Repara duplicados V6: prefere registo fechado, senão o mais antigo.

    Mantém o trade plan original, não ajusta SL/TP a cada scan.
    """
    unique: list[dict[str, Any]] = []
    removed = 0
    sorted_trades = sorted(trades, key=lambda t: int(t.get("opened_at_ms") or 0))
    for trade in sorted_trades:
        existing = next((x for x in unique if same_setup(x, trade)), None)
        if existing is None:
            unique.append(trade)
            continue
        removed += 1
        if existing.get("status") == "Em curso" and trade.get("status") in ("Ganha", "Perdida"):
            for key in ("status", "resultado_r", "closed_at_ms", "closed_at", "notas"):
                existing[key] = trade.get(key)
    return unique, removed


def already_traded(signal: dict[str, Any], trades: list[dict[str, Any]]) -> bool:
    return any(same_setup(signal, trade) for trade in trades)


def legacy_sent_breakouts(sent_ids: set[str], current_exchange: str, market_type: str) -> list[dict[str, Any]]:
    """Lê IDs V6 como marcadores históricos, sem reenviar setups antigos."""
    result = []
    prefix = current_exchange + "_"
    for value in sent_ids:
        if not str(value).startswith(prefix) or "_breakout_" not in value:
            continue
        before, tail = str(value)[len(prefix):].split("_breakout_", 1)
        bits = tail.split("_")
        if len(bits) < 3:
            continue
        tf, breakout_ms = bits[:2]
        phase_bit = bits[-1]
        if tf not in TF_MS or phase_bit not in ("entry", "possible"):
            continue
        # Símbolos CCXT com / e : são reduzidos para esta forma na V6.
        # O formato spot pode ser reconstituído sem ambiguidades para /USDT.
        if before.endswith("USDT"):
            sym = before[:-4] + "/USDT"
        elif before.endswith("USDT:USDT"):
            sym = before[:-9] + "/USDT:USDT"
        else:
            continue
        try:
            level = float(bits[2]) if len(bits) >= 4 else None
            ms = int(breakout_ms)
        except (TypeError, ValueError):
            continue
        result.append({
            "symbol": sym, "timeframe": tf, "fase": "Entrada" if phase_bit == "entry" else "Possível setup",
            "breakout_time_ms": ms, "nivel": level, "market_type": market_type,
        })
    return result
