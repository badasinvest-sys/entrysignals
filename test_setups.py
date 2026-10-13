"""Testes offline: divergências, resistência, sustentação, reteste e invalidação.

Executar: python -m unittest -v test_setups.py
Não faz pedidos de API nem contacta o Telegram.
"""
import unittest

import numpy as np
import pandas as pd

from rsi_divergence_bot import detetar_divergencias_rsi
from breakout_setup import detetar_setups_breakout


def fake_rsi(kind, pivots):
    n = 80
    t = 1_755_000_000_000 + np.arange(n) * 3_600_000
    close = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 99.0)
    rsi = np.full(n, 50.0)
    for k, i in enumerate(pivots):
        if kind == "bearish":
            highs[i] = 105 + 5*k
            rsi[i] = 75 - 7*k
        else:
            lows[i] = 95 - 5*k
            rsi[i] = 22 + 7*k
    return pd.DataFrame({"timestamp": t, "open": close, "high": highs, "low": lows,
                         "close": close, "volume": np.full(n, 100.0), "rsi": rsi})


def fake_breakout(n=149, hold=True, retest=True, high_volume=True):
    t = 1_755_000_000_000 + np.arange(155) * 14_400_000
    o = np.full(155, 100.0)
    h = np.full(155, 101.0)
    l = np.full(155, 99.0)
    c = np.full(155, 100.0)
    v = np.full(155, 100.0)
    for i in [75, 92, 110, 124]:
        h[i] = 110.0 + (0.08 if i in (75, 110) else 0.0)
        c[i] = 104.0
        v[i] = 340.0 if high_volume else 100.0
    # Rompimento da resistência na vela 142, com volume forte
    o[142], h[142], l[142], c[142], v[142] = 109.0, 115.2, 108.5, 113.0, 520.0
    for i in (143, 144, 145):
        o[i], h[i], l[i], c[i], v[i] = 113.0, 115.0, 112.0, 113.2, 150.0
    if not hold:
        c[144] = 107.0  # Falhou em sustentar nível
        v[145] = 100.0  # Não existe um novo rompimento com volume relevante
    o[146], h[146], l[146], c[146], v[146] = 114.0, 115.1, 112.5, 114.0, 120.0
    for i in range(148, 155):
        o[i], h[i], l[i], c[i], v[i] = 112.5, 113.5, 111.5, 112.6, 110.0
    if retest:
        o[147], h[147], l[147], c[147], v[147] = 113.5, 114.0, 109.5, 111.0, 170.0
    else:
        o[147], h[147], l[147], c[147], v[147] = 114.0, 116.0, 112.7, 114.5, 120.0
    data = pd.DataFrame({"timestamp": t, "open": o, "high": h, "low": l,
                         "close": c, "volume": v})
    return data.iloc[:n].copy()


class TestSetups(unittest.TestCase):
    def test_three_pivots_all_timeframes(self):
        for tf in ("15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"):
            for kind in ("bullish", "bearish"):
                with self.subTest(tf=tf, kind=kind):
                    out = detetar_divergencias_rsi(fake_rsi(kind, [45, 56, 70]), tf, max_age_bars=12)
                    self.assertTrue(any(s["pivots_count"] >= 3 and s["tipo"] == kind for s in out), out)

    def test_two_pivots_only_4h_1d(self):
        for tf in ("15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"):
            with self.subTest(tf=tf):
                out = detetar_divergencias_rsi(fake_rsi("bearish", [56, 70]), tf, max_age_bars=12)
                self.assertEqual(bool(out), tf in ("4h", "1d"), out)

    def test_only_breakout_4h_and_1d(self):
        d = fake_breakout(147)
        self.assertFalse(detetar_setups_breakout(d, "1h"))
        for tf in ("4h", "1d"):
            with self.subTest(tf=tf):
                o = detetar_setups_breakout(d, tf)
                self.assertTrue(any(s["fase"] == "Possível setup" for s in o), o)
                self.assertFalse(any(s["fase"] == "Entrada" for s in o), o)

    def test_entry_only_after_hold_and_retest(self):
        o = detetar_setups_breakout(fake_breakout(149), "4h")
        self.assertTrue(any(s["fase"] == "Entrada" for s in o), o)
        o = detetar_setups_breakout(fake_breakout(149, retest=False), "4h")
        self.assertFalse(any(s["fase"] == "Entrada" for s in o), o)

    def test_no_hold_no_signal(self):
        self.assertFalse(detetar_setups_breakout(fake_breakout(149, hold=False), "4h"))

    def test_rejected_if_price_later_falls_below_zone(self):
        d = fake_breakout(150)
        d.loc[149, "close"] = 100.0
        self.assertFalse(any(s["fase"] == "Entrada" for s in detetar_setups_breakout(d, "4h")))

    def test_high_volume_required(self):
        self.assertFalse(detetar_setups_breakout(fake_breakout(149, high_volume=False), "4h"))

    def test_aggregate_1h_for_unsupported_timeframes(self):
        from rsi_divergence_bot import agregar_velas_1h
        t0 = 1_728_000_000_000 // (12 * 3_600_000) * (12 * 3_600_000)
        raw = [[t0 + i * 3_600_000, 100.0 + i, 102.0 + i, 99.0 + i,
                101.0 + i, 50.0] for i in range(74)]
        daily_half = agregar_velas_1h(raw, "12h")
        self.assertEqual(len(daily_half), 6)
        self.assertEqual(daily_half.iloc[0]["open"], 100.0)
        self.assertEqual(daily_half.iloc[0]["close"], 112.0)
        self.assertEqual(daily_half.iloc[0]["volume"], 600.0)
        self.assertEqual(len(agregar_velas_1h(raw, "6h")), 12)
        self.assertEqual(len(agregar_velas_1h(raw, "2h")), 36)

    def test_full_scanner_writes_breakout_signal_offline(self):
        import json
        import tempfile
        import types
        from pathlib import Path
        from unittest.mock import patch
        import rsi_divergence_bot as bot

        class FakeExchange:
            timeframes = {"4h": "4h"}
            def __init__(self, *args, **kwargs):
                pass
            def load_markets(self):
                return {"TEST/USDT": {"active": True, "spot": True,
                                       "contract": False, "base": "TEST"}}
            def fetch_tickers(self):
                return {"TEST/USDT": {"quoteVolume": 10_000_000.0}}
            def fetch_ohlcv(self, symbol, timeframe, limit=360):
                return fake_breakout(150).values.tolist()
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(bot, "ccxt", types.SimpleNamespace(mexc=FakeExchange)), \
                 patch.object(bot, "EXCHANGE_ID", "mexc"), \
                 patch.object(bot, "TIMEFRAMES", ["4h"]), \
                 patch.object(bot, "SCAN_TOP_N", 1), \
                 patch.object(bot, "EXTRA_SYMBOLS", []), \
                 patch.object(bot, "LATEST_FILE", Path(folder) / "latest.json"), \
                 patch.object(bot, "SENT_FILE", Path(folder) / "sent.json"), \
                 patch.object(bot, "TELEGRAM_BOT_TOKEN", "TOKEN_FAKE"), \
                 patch.object(bot, "TELEGRAM_CHAT_ID", "CHAT_FAKE"), \
                 patch.object(bot, "enviar_alerta_telegram", return_value=True) as sender:
                payload = bot.executar_scan()
                self.assertGreaterEqual(payload["breakout_entry_count"], 1)
                self.assertTrue(any(s["categoria"] == "breakout" for s in payload["signals"]))
                first_n = sender.call_count
                self.assertGreater(first_n, 0)
                # Segunda varredura idêntica: os mesmos IDs não podem voltar a enviar.
                bot.executar_scan()
                self.assertEqual(sender.call_count, first_n)
                self.assertEqual(json.loads((Path(folder) / "latest.json").read_text())["errors"], 0)

    def test_live_retest_after_hold(self):
        d = fake_breakout(147, retest=False)
        ts = int(d.iloc[-1]["timestamp"]) + 14_400_000
        forming = {"timestamp": ts, "open": 113.1, "high": 114.4,
                   "low": 109.8, "close": 110.7, "volume": 50}
        o = detetar_setups_breakout(d, "4h", forming_candle=forming)
        self.assertTrue(any(s["fase"] == "Entrada" and s["intrabar"] for s in o), o)


if __name__ == "__main__":
    unittest.main()

