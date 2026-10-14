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
except ImportError:
    ccxt = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
import requests

from breakout_setup import detetar_setups_breakout
from divergence_engine import (detetar_divergencias_rsi, calcular_rsi, RSI_PERIOD, MAX_SIGNAL_AGE_BARS,
                               MIN_RSI_DELTA, MIN_PRICE_DELTA_PCT, PRICE_PIVOT_DISTANCE)
from open_interest import OI_MODE, OI_SOURCE_EXCHANGE, validar_oi
from charting import render_signal_chart, save_trade_chart, CHARTS_DIR, chart_name
from dedup_setup import same_setup, same_phase, deduplicate_signals, deduplicate_trades, already_traded, _breakout_ms_from_id, legacy_sent_breakouts, suppress_stale_possible
from state_store import StateStore, StateUnavailable

BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = Path(os.getenv("LATEST_FILE", str(BASE_DIR / "latest_signals.json")))
SENT_FILE = Path(os.getenv("SENT_FILE", str(BASE_DIR / "sent_alerts.json")))
TRADES_FILE = Path(os.getenv("TRADES_FILE", str(BASE_DIR / "trades_history.json")))
ALERT_HISTORY_FILE = Path(os.getenv("ALERT_HISTORY_FILE", str(BASE_DIR / "alert_history.json")))
BOOTSTRAP_FILE = Path(os.getenv("BOOTSTRAP_FILE", str(BASE_DIR / "scanner_bootstrapped.json")))
STATE = StateStore()

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "mexc").strip().lower()
MARKET_TYPE = os.getenv("MARKET_TYPE", "spot").strip().lower()
if MARKET_TYPE not in ("spot", "swap"):
    raise ValueError("MARKET_TYPE só admite spot ou swap")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SCAN_INTERVAL_SECONDS = max(60, int(os.getenv("SCAN_INTERVAL_SECONDS", "300")))
SCAN_TOP_N = max(1, int(os.getenv("SCAN_TOP_N", "35")))
OHLCV_LIMIT = max(140, min(600, int(os.getenv("OHLCV_LIMIT", "320"))))
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
    data = STATE.load("sent_ids", SENT_FILE, [])
    if isinstance(data, list):
        return {str(x) for x in data}
    return set()


def save_sent_ids(sent_ids: set[str]) -> None:
    STATE.save("sent_ids", SENT_FILE, sorted(sent_ids)[-20000:])


def read_trades() -> list[dict[str, Any]]:
    data = STATE.load("trades", TRADES_FILE, {"trades": []})
    if isinstance(data, dict) and isinstance(data.get("trades"), list):
        return data["trades"]
    return []


def save_trades(trades: list[dict[str, Any]]) -> None:
    payload = {"updated_at": utc_now_iso(), "trades": trades}
    STATE.save("trades", TRADES_FILE, payload)


def get_liquid_usdt_symbols(exchange: ccxt.Exchange, limit: int) -> list[str]:
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    excluded_suffixes = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")
    excluded_bases = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDE"}
    ranked: list[tuple[float, str]] = []
    no_volume_symbols: list[str] = []
    extra_symbols = [s.strip().upper() for s in os.getenv("EXTRA_SYMBOLS", "").split(",") if s.strip()]
    def normalize_extra(s: str) -> str:
        if "/" not in s and s.endswith("USDT"):
            s = f"{s[:-4]}/USDT"
        if MARKET_TYPE == "swap" and s.endswith("/USDT"):
            s += ":USDT"
        return s
    extra_symbols_fmt = {normalize_extra(s) for s in extra_symbols}

    for symbol, market in markets.items():
        if MARKET_TYPE == "swap":
            if not symbol.endswith("/USDT:USDT") or market.get("swap") is not True:
                continue
        else:
            if not symbol.endswith("/USDT") or market.get("spot") is not True or market.get("contract") is True:
                continue
        if market.get("active") is False:
            continue
        base = str(market.get("base") or symbol.split("/")[0]).upper()
        if base in excluded_bases or base.endswith(excluded_suffixes):
            continue
        ticker = tickers.get(symbol) or {}
        try:
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            quote_volume = 0.0
        if quote_volume <= 0:
            try:
                quote_volume = float(ticker.get("baseVolume") or 0.0) * float(ticker.get("last") or 0.0)
            except (TypeError, ValueError):
                quote_volume = 0.0
        if quote_volume > 0:
            ranked.append((quote_volume, symbol))
        else:
            no_volume_symbols.append(symbol)

    ranked.sort(reverse=True)
    symbols = [symbol for _, symbol in ranked[:limit]]
    # Alguns contratos devolvem ticker sem quoteVolume: não excluir o
    # mercado inteiro, mas deixar estes pares para depois dos que têm volume.
    if len(symbols) < limit:
        symbols.extend(no_volume_symbols[:limit - len(symbols)])
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
    oi = signal.get("oi") or {}
    oi_line = f"OI: {oi.get('status', 'Indisponível')}"
    if oi.get("variacao_pct") is not None:
        oi_line += f" ({oi['variacao_pct']:+.1f}%)"
    if oi.get("fonte"):
        oi_line += f" | {oi['fonte']}"
    return "\n".join([
        f"{icon} RSI {signal['tipo'].upper()} | {signal['symbol']} {signal['timeframe']}",
        f"{signal['pivots_count']} pivots | {signal.get('rejection_count', 0)} rejeições | {signal.get('sweep_count', 0)} sweeps",
        f"Preço: {prices}",
        f"RSI: {rsis}",
        oi_line,
        "Padrão de exaustão potencial — não é sinal de entrada.",
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


def upsert_trade(trades: list[dict[str, Any]], signal: dict[str, Any]) -> None:
    trade = signal.get("trade") or {}
    if not trade:
        return
    trade_id = signal["id"]
    if already_traded(signal, trades):
        log.info("Trade equivalente já registada: %s %s", signal["symbol"], signal["timeframe"])
        return
    trades.append({
        "trade_id": trade_id,
        "setup_id": signal.get("setup_id"),
        "breakout_time_ms": signal.get("breakout_time_ms") or _breakout_ms_from_id(str(signal.get("setup_id", ""))),
        "market_type": MARKET_TYPE,
        "symbol": signal["symbol"],
        "timeframe": signal["timeframe"],
        "categoria": signal["categoria"],
        "entrada": float(trade["entry"]),
        "sl": float(trade["sl"]),
        "tp": float(trade["tp"]),
        "rr": 2.0,
        "nivel": float(signal["nivel"]),
        "zona_min": float(signal["zona_min"]),
        "zona_max": float(signal["zona_max"]),
        "volume_zona_ratio": signal.get("volume_zona_ratio"),
        "imagem": chart_name(trade_id),
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
    log.info("Scan iniciado | exchange=%s | mercado=%s | timeframes=%s | top=%d pares | candles=%d", EXCHANGE_ID, MARKET_TYPE, TIMEFRAMES, SCAN_TOP_N, OHLCV_LIMIT)

    if ccxt is None:
        raise RuntimeError("A biblioteca ccxt não está instalada. Executa: pip install -r requirements.txt")
    factory = getattr(ccxt, EXCHANGE_ID, None)
    if factory is None:
        raise ValueError(f"Exchange CCXT desconhecida: {EXCHANGE_ID}")
    exchange = factory({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": MARKET_TYPE}})

    signals: list[dict[str, Any]] = []
    errors = 0
    ohlcv_ok = 0
    pair_timeframes_processed = 0
    symbols: list[str] = []
    supported_timeframes: list[str] = []
    # Falhas no estado remoto param o scan ANTES de enviar alertas.
    sent_ids = read_sent_ids()
    sent_signal_history = [x for x in STATE.load("alert_history", ALERT_HISTORY_FILE, []) if isinstance(x, dict)]
    is_bootstrapped = bool(STATE.load("bootstrapped", BOOTSTRAP_FILE, False))
    sent_signal_history.extend(legacy_sent_breakouts(sent_ids, EXCHANGE_ID, MARKET_TYPE))
    market_cache: dict[tuple[str, str], pd.DataFrame] = {}
    trades, removed_duplicates = deduplicate_trades(read_trades())
    if removed_duplicates:
        log.warning("Histórico: removidas %d trades duplicadas da versão anterior", removed_duplicates)
        save_trades(trades)
    active_trade_keys = {(t["symbol"], t["timeframe"]) for t in trades if t.get("status") == "Em curso"}
    oi_exchange: Any | None = None
    oi_initialized = False
    oi_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    oi_blocked = 0

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
                    if (symbol, timeframe) in active_trade_keys:
                        market_cache[(symbol, timeframe)] = df
                    ohlcv_ok += 1
                    pair_timeframes_processed += 1

                    rsi_signals = detetar_divergencias_rsi(df, timeframe)
                    if rsi_signals and OI_MODE != "off":
                        if not oi_initialized:
                            oi_initialized = True
                            try:
                                source_factory = getattr(ccxt, OI_SOURCE_EXCHANGE)
                                oi_exchange = source_factory({"enableRateLimit": True, "timeout": 16000,
                                                             "options": {"defaultType": "swap"}})
                                oi_exchange.load_markets()
                                log.info("OI: fornecedor de perpétuos = %s", OI_SOURCE_EXCHANGE)
                            except Exception as exc:
                                log.warning("OI: fonte indisponível: %s", str(exc)[:250])
                                oi_exchange = None
                        approved_rsi = []
                        for rsi_sig in rsi_signals:
                            oi_key = (symbol, timeframe, int(rsi_sig["signal_time_ms"]))
                            if oi_key in oi_cache:
                                cache = oi_cache[oi_key]
                                rsi_sig["oi"] = cache["oi"]
                                approved = cache["approved"]
                            else:
                                approved = validar_oi(oi_exchange, symbol, rsi_sig)
                                oi_cache[oi_key] = {"oi": rsi_sig["oi"], "approved": approved}
                            if approved:
                                approved_rsi.append(rsi_sig)
                            else:
                                oi_blocked += 1
                        rsi_signals = approved_rsi
                    elif OI_MODE == "off":
                        for rsi_sig in rsi_signals:
                            rsi_sig["oi"] = {"status": "Desligado", "confirmado": None}
                    breakout_signals = detetar_setups_breakout(df, timeframe, forming_candle=forming)
                    for signal in rsi_signals + breakout_signals:
                        signal["symbol"] = symbol
                        signal["exchange"] = EXCHANGE_ID.upper()
                        signal["id"] = f"{EXCHANGE_ID}_{symbol.replace('/', '')}_{signal['id']}"
                        signals.append(signal)
                        if signal.get("categoria") == "breakout" and signal.get("fase") == "Entrada":
                            market_cache[(symbol, timeframe)] = df
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
        if oi_exchange is not None:
            try:
                oi_exchange.close()
            except Exception:
                pass

    unique = {signal["id"]: signal for signal in signals}
    signals = deduplicate_signals(list(unique.values()))
    # O aviso "aguardando reteste" não deve surgir quando já há entrada.
    signals = suppress_stale_possible(signals, trades, sent_signal_history)

    prior_trade_snapshot = [dict(t) for t in trades]
    for signal in signals:
        if signal.get("categoria") == "breakout" and signal.get("fase") == "Entrada":
            if already_traded(signal, trades):
                continue
            upsert_trade(trades, signal)
    statuses_before = {t["trade_id"]: t.get("status") for t in trades}
    atualizar_estado_trades(trades, market_cache)
    # Uma imagem guardada por trade, atualizada quando fecha em SL/TP.
    max_trade_images = max(0, int(os.getenv("MAX_TRADE_IMAGES_PER_SCAN", "2")))
    generated_trade_images = 0
    for trade in trades:
        if generated_trade_images >= max_trade_images:
            break
        df_trade = market_cache.get((trade["symbol"], trade["timeframe"]))
        if df_trade is None:
            continue
        img_name = trade.get("imagem") or chart_name(str(trade["trade_id"]))
        trade["imagem"] = img_name
        path_img = CHARTS_DIR / img_name
        if (not path_img.is_file()) or (statuses_before.get(trade["trade_id"]) != trade.get("status")):
            try:
                save_trade_chart(trade, df_trade)
                generated_trade_images += 1
            except Exception:
                log.exception("Não foi possível guardar gráfico da trade %s", trade["trade_id"])
    save_trades(trades)
    market_cache.clear()

    payload = {
        "scan_started_at": started_at,
        "scan_completed_at": utc_now_iso(),
        "scan_seconds": round(time.monotonic() - started_monotonic, 1),
        "exchange": EXCHANGE_ID.upper(),
        "market_type": MARKET_TYPE,
        "symbols_scanned": len(symbols),
        "timeframes": supported_timeframes,
        "ohlcv_requests_ok": ohlcv_ok,
        "pair_timeframes_processed": pair_timeframes_processed,
        "errors": errors,
        "oi_blocked": oi_blocked,
        "signals_count": len(signals),
        "signals": signals,
        "configuration": {
            "timeframes": supported_timeframes,
            "rsi_period": RSI_PERIOD,
            "min_rsi_difference": MIN_RSI_DELTA,
            "min_price_difference_pct": MIN_PRICE_DELTA_PCT,
            "price_pivot_distance": PRICE_PIVOT_DISTANCE,
            "max_signal_age_bars": MAX_SIGNAL_AGE_BARS,
            "oi_mode": OI_MODE,
            "oi_source_exchange": OI_SOURCE_EXCHANGE,
            "market_type": MARKET_TYPE,
            "nota": "2h, 6h e 12h foram excluídos por completo.",
        },
    }
    write_json_atomic(LATEST_FILE, payload)

    image_exchange = None
    new_alerts = 0
    # Primeira execução com estado vazio: não reencaminhar rompimentos
    # históricos reaparecidos após um deploy. A partir do segundo scan,
    # apenas eventos novos recebem alerta.
    if not is_bootstrapped:
        old_breakouts = 0
        for signal in signals:
            if signal.get("categoria") != "breakout":
                continue
            sent_ids.add(signal["id"])
            sent_signal_history.append({
                "id": signal["id"], "categoria": "breakout", "fase": signal["fase"],
                "symbol": signal["symbol"], "timeframe": signal["timeframe"],
                "market_type": MARKET_TYPE, "setup_id": signal.get("setup_id"),
                "breakout_time_ms": signal.get("breakout_time_ms"), "nivel": signal.get("nivel"),
                "origin": "bootstrap_no_replay",
            })
            old_breakouts += 1
        STATE.save("alert_history", ALERT_HISTORY_FILE, sent_signal_history[-2500:])
        save_sent_ids(sent_ids)
        STATE.save("bootstrapped", BOOTSTRAP_FILE, True)
        log.warning("Primeiro scan: %d rompimentos históricos registados sem Telegram, para evitar alertas retroativos.", old_breakouts)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        for signal in signals:
            if signal["id"] in sent_ids:
                continue
            if signal.get("categoria") == "breakout":
                # Evitar reenviar entradas antigas mesmo após atualizar da V6 para V7.
                if signal.get("fase") == "Entrada" and already_traded(signal, prior_trade_snapshot):
                    log.info("Entrada já registada antes deste scan: %s", signal["id"])
                    continue
                if any(same_phase(signal, previous) for previous in sent_signal_history):
                    log.info("Alerta equivalente já enviado: %s", signal["id"])
                    continue
                # Possível setup nunca aparece depois de um reteste/entrada.
                if signal["fase"] == "Possível setup" and any(
                    same_setup(signal, previous) and previous.get("fase") == "Entrada"
                    for previous in sent_signal_history
                ):
                    continue
            # Marcar como reservado e persistir ANTES de enviar Telegram.
            # Assim, mesmo um restart após envio não dispara novamente.
            sent_ids.add(signal["id"])
            if signal.get("categoria") == "breakout":
                sent_signal_history.append({
                    "id": signal["id"], "categoria": "breakout", "fase": signal["fase"],
                    "symbol": signal["symbol"], "timeframe": signal["timeframe"],
                    "market_type": MARKET_TYPE, "setup_id": signal.get("setup_id"),
                    "breakout_time_ms": signal.get("breakout_time_ms"),
                    "nivel": signal.get("nivel"), "origin": "reserved_before_send",
                })
                STATE.save("alert_history", ALERT_HISTORY_FILE, sent_signal_history[-2500:])
            save_sent_ids(sent_ids)
            if signal.get("categoria") == "rsi_divergence":
                # Sem imagens RSI: menos RAM e uploads para o Telegram.
                ok = enviar_alerta_telegram(formatar_alerta_rsi(signal), None)
            else:
                chart = None
                try:
                    if image_exchange is None:
                        image_exchange = factory({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": MARKET_TYPE}})
                    raw_chart = image_exchange.fetch_ohlcv(signal["symbol"], timeframe=signal["timeframe"], limit=min(140, OHLCV_LIMIT))
                    if raw_chart and len(raw_chart) >= 20:
                        chart_df = pd.DataFrame(raw_chart, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        chart = render_signal_chart(chart_df, signal)
                except Exception:
                    log.exception("Falha a gerar imagem do sinal %s", signal["id"])
                ok = enviar_alerta_telegram(formatar_alerta_breakout(signal), chart)
            if ok:
                new_alerts += 1
            else:
                log.error("Telegram falhou após reserva de %s; não há repetição automática para evitar duplicação.", signal["id"])
    else:
        log.warning("Alertas Telegram desativados por falta de credenciais.")

    if image_exchange is not None:
        try:
            image_exchange.close()
        except Exception:
            pass
    log.info(
        "Scan concluído | pares=%d | OHLCV OK=%d | pares/timeframes=%d | erros=%d | sinais=%d | novos alertas=%d | duração=%.1fs",
        len(symbols), ohlcv_ok, pair_timeframes_processed, errors, len(signals), new_alerts, payload["scan_seconds"],
    )
    return payload


def main() -> None:
    if "--once" in sys.argv:
        with STATE.exclusive_scan() as allowed:
            if allowed:
                executar_scan()
            else:
                log.warning("Outro scanner está ativo; scan --once omitido.")
        return
    log.info("Scanner iniciado; próxima varredura após cada ciclo de %d segundos.", SCAN_INTERVAL_SECONDS)
    while True:
        cycle_start = time.monotonic()
        try:
            with STATE.exclusive_scan() as allowed:
                if allowed:
                    executar_scan()
                else:
                    log.warning("Outro scanner está ativo; ciclo omitido para prevenir notificações duplicadas.")
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
