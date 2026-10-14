"""Testes V8: caso INJ, fases e persistência após reinício local.

python test_dedup_v8.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dedup_setup import same_setup, same_phase, deduplicate_signals, deduplicate_trades, already_traded, suppress_stale_possible
import rsi_divergence_bot as bot
from state_store import StateStore

BAR_MS = 4 * 60 * 60 * 1000
BASE = 1_760_000_000_000


def signal(bars=0, level=7.824, kind="Entrada", symbol="INJ/USDT", tf="4h"):
    entry = 7.952
    stop = 7.59 if bars == 0 else 7.6004
    return {
        "id": f"mexc_INJUSDT_breakout_{tf}_{BASE + bars*BAR_MS}_{'entry' if kind == 'Entrada' else 'possible'}",
        "setup_id": f"breakout_{tf}_{BASE + bars*BAR_MS}",
        "categoria": "breakout", "tipo": "bullish", "fase": kind,
        "symbol": symbol, "timeframe": tf,
        "breakout_time_ms": BASE + bars*BAR_MS,
        "signal_time_ms": BASE + 20*BAR_MS,
        "signal_time": "2026-10-14 08:00 UTC",
        "nivel": level, "zona_min": level-.108, "zona_max": level+.108,
        "toques_resistencia": 3, "volume_zona_ratio": 4.81,
        "volume_rompimento_ratio": 2.27,
        "trade": {"entry": entry, "sl": stop, "tp": entry + 2 * (entry-stop), "rr": 2.0} if kind=="Entrada" else None,
    }


def test_logic():
    a = signal()
    b = signal(bars=8, level=7.824)
    c = signal(bars=2, level=8.30)
    d = signal(bars=30)
    possible = signal(kind="Possível setup")
    assert same_setup(a,b), "INJ: 8 candles entre rupturas devem contar como mesma oportunidade"
    assert not same_setup(a,c), "Zona materialmente distinta tem de ser independente"
    assert not same_setup(a,d), "Nova campanha 30 candles depois não deve ser bloqueada"
    assert not same_setup(a,signal(symbol="BTC/USDT")), "Símbolo diferente"
    assert not same_setup(a,signal(tf="1d")), "Timeframe diferente"
    assert len(deduplicate_signals([a,b]))==1
    assert len(deduplicate_signals([a,b,possible]))==2
    assert len(suppress_stale_possible([a,possible], [], []))==1
    assert suppress_stale_possible([possible], [], [dict(a, fase="Entrada")]) == []
    trades=[]
    bot.upsert_trade(trades, a)
    bot.upsert_trade(trades, b)
    assert len(trades) == 1 and trades[0]["sl"] == 7.59
    assert already_traded(b, trades)
    older = dict(trades[0], trade_id=b['id'], setup_id=b['setup_id'], breakout_time_ms=b['breakout_time_ms'], sl=7.6004)
    normalized, removed = deduplicate_trades([trades[0], older])
    assert removed==1 and len(normalized)==1 and normalized[0]["sl"]==7.59
    bot.upsert_trade(normalized,d)
    assert len(normalized)==2
    print("PASSOU: mesmo setup INJ (zonas próximas e breakouts distantes), dois níveis distintos, fases e trades")


class FakeExchange:
    timeframes = {"4h": "4h"}
    def __init__(self, *args, **kwargs):
        pass
    def fetch_ohlcv(self, symbol, timeframe, limit):
        base_ms = BASE - 250 * BAR_MS
        return [[base_ms + i*BAR_MS, 7.96, 8.08, 7.82, 7.952, 1200] for i in range(260)]
    def close(self):
        pass


def test_scan_integration():
    with tempfile.TemporaryDirectory(prefix="v8_state_") as folder:
        path = Path(folder)
        orig = {
            "LATEST_FILE": bot.LATEST_FILE,
            "TRADES_FILE": bot.TRADES_FILE,
            "SENT_FILE": bot.SENT_FILE,
            "ALERT_HISTORY_FILE": bot.ALERT_HISTORY_FILE,
            "BOOTSTRAP_FILE": bot.BOOTSTRAP_FILE,
            "STATE": bot.STATE,
            "ccxt": bot.ccxt,
            "TIMEFRAMES": bot.TIMEFRAMES,
            "OI_MODE": bot.OI_MODE,
            "TELEGRAM_BOT_TOKEN": bot.TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": bot.TELEGRAM_CHAT_ID,
            "get_liquid_usdt_symbols": bot.get_liquid_usdt_symbols,
            "detetar_divergencias_rsi": bot.detetar_divergencias_rsi,
            "detetar_setups_breakout": bot.detetar_setups_breakout,
            "enviar_alerta_telegram": bot.enviar_alerta_telegram,
            "save_trade_chart": bot.save_trade_chart,
        }
        try:
            bot.LATEST_FILE = path / 'latest.json'
            bot.TRADES_FILE = path / 'trades.json'
            bot.SENT_FILE = path / 'sent.json'
            bot.ALERT_HISTORY_FILE = path / 'alerts.json'
            bot.BOOTSTRAP_FILE = path / 'boot.json'
            bot.STATE = StateStore()
            bot.ccxt = SimpleNamespace(mexc=FakeExchange)
            bot.TIMEFRAMES = ["4h"]
            bot.OI_MODE = "off"
            bot.TELEGRAM_BOT_TOKEN = "dummy"
            bot.TELEGRAM_CHAT_ID = "dummy"
            bot.get_liquid_usdt_symbols = lambda ex, limit: ["INJ/USDT"]
            bot.detetar_divergencias_rsi = lambda df, tf: []
            current = [signal(), signal(bars=8, level=7.824), signal(kind="Possível setup")]
            bot.detetar_setups_breakout = lambda *args, **kwargs: [dict(x) for x in current]
            bot.save_trade_chart = lambda *args, **kwargs: None
            sends=[]
            bot.enviar_alerta_telegram = lambda msg, photo: sends.append(msg) or True

            # Primeiro scan regista histórico sem fazer backfill de entradas antigas.
            first = bot.executar_scan()
            assert len(sends) == 0 and first['signals_count'] == 1, (sends, first['signals'])
            assert len(bot.read_trades()) == 1
            assert len(bot.STATE.load('alert_history', bot.ALERT_HISTORY_FILE, [])) == 1
            second = bot.executar_scan()
            assert len(sends) == 0 and second['signals_count'] == 1
            # Reinício: perde-se o estado em RAM, mas ficheiros persistem neste teste.
            bot.STATE = StateStore()
            third = bot.executar_scan()
            assert len(sends) == 0
            # Novo símbolo: novo breakout, uma entrada permitida.
            bot.get_liquid_usdt_symbols = lambda ex, limit: ["BTC/USDT"]
            bot.detetar_setups_breakout = lambda *args, **kwargs: [signal(symbol="BTC/USDT")]
            fourth = bot.executar_scan()
            assert len(sends) == 1 and fourth['signals_count'] == 1
            fifth = bot.executar_scan()
            assert len(sends) == 1, "Não enviar repetidamente o mesmo setup BTC"
            # Também não enviar retrospectivamente o possível quando entrada já existe.
            bot.detetar_setups_breakout = lambda *args, **kwargs: [signal(kind="Possível setup",symbol="BTC/USDT")]
            sixth = bot.executar_scan()
            assert len(sends) == 1 and sixth['signals_count'] == 0
            print("PASSOU: 6 scans simulados, sem triplos alertas, sem backfill, sem duplicação após reinício")
        finally:
            for k,v in orig.items():
                setattr(bot,k,v)


def test_remote_store_mock():
    """Simula GET/SET Upstash sem acesso à Internet."""
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'store.json'
        store=StateStore()
        store.remote=True
        values={}
        def fake_command(cmd, key, *args):
            if cmd == 'GET': return values.get(key)
            if cmd == 'SET':
                # SET key token NX EX ttl
                if len(args) > 1 and args[1] == 'NX' and key in values:
                    return None
                values[key] = args[0]
                return 'OK'
            if cmd == 'EVAL':
                # EVAL script 1 key token
                _, _, held_key, token = (key,)+args
                if values.get(held_key) == token:
                    del values[held_key]
                    return 1
                return 0
            raise AssertionError(cmd)
        store._command=fake_command
        assert store.load('sent_ids', path, [])==[]
        store.save('sent_ids', path, ['a','b'])
        path.unlink()
        store2=StateStore()
        store2.remote=True
        store2._command=fake_command
        assert store2.load('sent_ids', path, [])==['a','b']
        with store2.exclusive_scan() as ok1:
            assert ok1
            with store.exclusive_scan() as ok2:
                assert not ok2, 'Não podem correr dois scanners em paralelo com Redis'
        with store.exclusive_scan() as ok3:
            assert ok3, 'O lock deve ser libertado no fim do scan'
        print('PASSOU: estado remoto persistente e lock exclusivo entre dois scanners')


if __name__ == '__main__':
    test_logic()
    test_remote_store_mock()
    test_scan_integration()
