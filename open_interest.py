"""Validação opcional de divergência usando OI HISTÓRICO de perpétuos.

AVISO: spot não tem OI. Usa contratos perpétuos do mesmo subjacente.
Nunca compara a última fotografia de OI com um pivot histórico.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

log = logging.getLogger("rsi-breakout-bot")
OI_MODE = os.getenv("OI_MODE", "confirm_if_available").strip().lower()
OI_SOURCE_EXCHANGE = os.getenv("OI_SOURCE_EXCHANGE", "binance").strip().lower()
OI_MIN_DROP_PCT = max(0.1, float(os.getenv("OI_MIN_DROP_PCT", "2.0")))
TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def oi_confluence(records: list[dict[str, Any]], pivots: list[dict[str, Any]], timeframe: str) -> dict[str, Any]:
    """Retorna 'confirmado', 'contradiz' ou 'indisponível'. Usa amount e tempo.

    OI decrescente num novo extremo de preço é uma evidência possível de
    menor participação alavancada, NÃO uma prova de cansaço por si só.
    """
    tf_ms = TF_MS.get(timeframe)
    if not tf_ms or len(pivots) < 2:
        return {"status": "Indisponível", "confirmado": None, "motivo": "Sem pivots/TF comparáveis"}
    clean = []
    for r in records:
        try:
            timestamp = int(r.get("timestamp"))
            value = float(r.get("openInterestAmount"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            clean.append((timestamp, value))
    clean.sort()
    values = []
    for pivot in pivots:
        # Só observações conhecidas até ao fim da vela que marcou o extremo.
        pivot_time = int(pivot.get("time_ms"))
        last_allowed = pivot_time + tf_ms
        candidates = [(t, v) for t, v in clean if pivot_time - tf_ms <= t <= last_allowed]
        if not candidates:
            return {"status": "Indisponível", "confirmado": None, "motivo": "Histórico OI sem alinhamento temporal"}
        values.append(max(candidates, key=lambda x: x[0])[1])
    previous, final = values[-2], values[-1]
    change_pct = (final / previous - 1.0) * 100.0
    confirmed = change_pct <= -OI_MIN_DROP_PCT
    return {
        "status": "Confirma" if confirmed else "Não confirma",
        "confirmado": confirmed,
        "variacao_pct": round(change_pct, 2),
        "oi_pivots": [round(v, 5) for v in values],
        "motivo": "OI dos perpétuos a contrair" if confirmed else "OI não mostra contração suficiente",
    }


def _swap_symbol(spot_symbol: str) -> str:
    base, remainder = spot_symbol.split("/", 1)
    quote = remainder.split(":", 1)[0]
    return f"{base}/{quote}:{quote}"


def validar_oi(exchange: Any | None, spot_symbol: str, signal: dict[str, Any]) -> bool:
    """Anexa diagnóstico ao sinal e devolve se deve passar o filtro.

    Em confirm_if_available, dados disponíveis e não confirmatórios REJEITAM;
    dados indisponíveis aceitam com etiqueta SEM CONFIRMAÇÃO. Em require,
    rejeita também quando não existe histórico.
    """
    if OI_MODE == "off":
        signal["oi"] = {"status": "Desligado", "confirmado": None}
        return True
    if exchange is None or not exchange.has.get("fetchOpenInterestHistory", False):
        signal["oi"] = {"status": "Indisponível", "confirmado": None, "motivo": "Exchange OI sem histórico unificado"}
        return OI_MODE != "require"
    symbol = _swap_symbol(spot_symbol)
    if symbol not in exchange.markets:
        signal["oi"] = {"status": "Indisponível", "confirmado": None, "motivo": "Sem contrato perpétuo correspondente"}
        return OI_MODE != "require"
    pivots = signal["pivots"]
    timeframe = signal["timeframe"]
    tf_ms = TF_MS[timeframe]
    since = max(0, int(pivots[0]["time_ms"]) - tf_ms * 2)
    try:
        rows = exchange.fetch_open_interest_history(symbol, timeframe=timeframe, since=since, limit=250)
        info = oi_confluence(rows or [], pivots, timeframe)
        info["fonte"] = f"{OI_SOURCE_EXCHANGE.upper()} PERP {symbol}"
        info["nota"] = "Derivados do mesmo ativo; não é OI do mercado spot"
        signal["oi"] = info
        if info["confirmado"] is None:
            return OI_MODE != "require"
        return bool(info["confirmado"])
    except Exception as exc:
        log.warning("Open Interest indisponível para %s %s: %s", spot_symbol, timeframe, str(exc)[:220])
        signal["oi"] = {"status": "Indisponível", "confirmado": None, "motivo": "API OI falhou"}
        return OI_MODE != "require"
