"""Testes sintéticos locais. Execução: python test_v5.py"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from divergence_engine import detetar_divergencias_rsi
from open_interest import oi_confluence
from charting import render_signal_chart, save_trade_chart


def sample(kind="bearish", positions=(110, 122, 134), prices=(110.0, 114.0, 118.0),
           osc=(75.0, 68.0, 60.0), last_rejection=True):
    n = 139
    close = np.full(n, 100.0)
    opens = np.full(n, 100.1)
    highs = np.full(n, 100.55)
    lows = np.full(n, 99.45)
    volume = np.full(n, 100.0)
    rsi = np.full(n, 50.0)
    for idx, price, r in zip(positions, prices, osc):
        if kind == "bearish":
            highs[idx] = float(price)
            lows[idx] = 99.4
            opens[idx] = 101.5
            close[idx] = 100.1
        else:
            lows[idx] = float(price)
            highs[idx] = 100.6
            opens[idx] = 98.5
            close[idx] = 99.9
        rsi[idx] = r
    if not last_rejection:
        idx = positions[-1]
        if kind == "bearish":
            opens[idx] = prices[-1] - .1
            close[idx] = prices[-1] - .05
        else:
            opens[idx] = prices[-1] + .1
            close[idx] = prices[-1] + .05
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=np.int64) * 3_600_000,
        "open": opens, "high": highs, "low": lows, "close": close, "volume": volume,
    })
    return df, rsi


def run():
    tests = 0
    def check(condition, msg):
        nonlocal tests
        tests += 1
        assert condition, msg
        print(f"OK {tests:02d} {msg}")

    df, rsi = sample()
    results = detetar_divergencias_rsi(df, "1h", rsi_override=rsi)
    check(len(results) == 1 and results[0]["pivots_count"] == 3 and results[0]["tipo"] == "bearish", "3 picos: RSI desce e preço sobe")
    check([x["rsi"] for x in results[0]["pivots"]] == [75,68,60], "RSI ancorado exatamente à vela do HIGH")

    df, rsi = sample(osc=(75,68,72))
    check(detetar_divergencias_rsi(df, "1h", rsi_override=rsi) == [], "3.º pico do RSI volta a subir: padrão rejeitado")
    check(detetar_divergencias_rsi(df, "4h", rsi_override=rsi) == [], "Nem 2 pivots antigos escapam à idade máxima")

    df, rsi = sample(prices=(110.0, 110.2, 110.4))
    check(detetar_divergencias_rsi(df, "1h", rsi_override=rsi) == [], "Novos highs demasiado residuais: padrão rejeitado")


    df, rsi = sample(positions=(96,108,121,134), prices=(106,110,114,118), osc=(82,76,80,60))
    check(detetar_divergencias_rsi(df, "1h", rsi_override=rsi) == [], "Pivot intermédio com RSI contrário não pode ser ignorado")

    df, rsi = sample(positions=(110,122,134), prices=(110,118,116), osc=(75,68,60))
    check(detetar_divergencias_rsi(df, "1h", rsi_override=rsi) == [], "Último máximo não é absoluto face à perna anterior")
    df, rsi = sample(last_rejection=False)
    check(detetar_divergencias_rsi(df, "1h", rsi_override=rsi) == [], "Sem rejeição/sweep no último pivot: rejeitado")

    df, rsi = sample(positions=(110,134), prices=(110,118), osc=(72,60))
    check(detetar_divergencias_rsi(df, "1h", rsi_override=rsi) == [], "Só 2 pivots em 1h: rejeitado")
    check(len(detetar_divergencias_rsi(df, "4h", rsi_override=rsi)) == 1, "2 pivots em 4h: permitido")
    check(len(detetar_divergencias_rsi(df, "1d", rsi_override=rsi)) == 1, "2 pivots em 1d: permitido")

    df, rsi = sample(kind="bullish", prices=(90,86,82), osc=(25,32,39))
    check(len(detetar_divergencias_rsi(df, "30m", rsi_override=rsi)) == 1, "3 lows absolutos + RSI crescente com rejeição")

    df, rsi = sample()
    check(detetar_divergencias_rsi(df, "2h", rsi_override=rsi) == [], "2h totalmente excluído")
    check(detetar_divergencias_rsi(df, "6h", rsi_override=rsi) == [], "6h totalmente excluído")
    check(detetar_divergencias_rsi(df, "12h", rsi_override=rsi) == [], "12h totalmente excluído")

    sig = detetar_divergencias_rsi(df, "1h", rsi_override=rsi)[0]
    sig["symbol"] = "TEST/USDT"
    sig["oi"] = {"status": "Indisponível"}
    png = render_signal_chart(df, sig)
    check(png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10000, "Imagem candlestick e RSI para alerta Telegram")

    oi_records = [{"timestamp": p["time_ms"], "openInterestAmount": v} for p, v in zip(sig["pivots"], (1000,950,900))]
    oi = oi_confluence(oi_records, sig["pivots"], "1h")
    check(oi["confirmado"] is True and oi["variacao_pct"] < 0, "OI confirma contração por pivots históricos")
    oi_records[-1]["openInterestAmount"] = 1050
    check(oi_confluence(oi_records, sig["pivots"], "1h")["confirmado"] is False, "OI crescente não confirma exaustão")
    check(oi_confluence([], sig["pivots"], "1h")["confirmado"] is None, "OI ausente não é fabricado")

    trade = {
        "trade_id":"test_trade","symbol":"TEST/USDT","timeframe":"4h", "status":"Em curso",
        "nivel":100.5,"zona_min":100.2,"zona_max":100.8,"entrada":101.0,"sl":98.0,"tp":107.0,
        "volume_zona_ratio":1.8,
    }
    with tempfile.TemporaryDirectory() as folder:
        import charting
        original_dir = charting.CHARTS_DIR
        charting.CHARTS_DIR = Path(folder)
        try:
            name = save_trade_chart(trade, df)
            check((Path(folder)/name).is_file(), "Trade em curso tem imagem no histórico")
            trade["status"] = "Ganha"
            save_trade_chart(trade, df)
            check((Path(folder)/name).stat().st_size > 10000, "Trade terminada atualiza imagem")
        finally:
            charting.CHARTS_DIR = original_dir

    print(f"PASSARAM {tests} TESTES LOCAIS.")


if __name__ == "__main__":
    run()


def integration_offline():
    """Smoke-test do scan completo, sem internet ou conta Telegram."""
    import tempfile
    import types
    import rsi_divergence_bot as bot

    class FakeExchange:
        timeframes = {"15m": True, "30m": True, "1h": True, "4h": True, "1d": True}
        def __init__(self, config):
            self.config = config
            self.markets = {"TEST/USDT": {"active": True, "spot": True, "base": "TEST", "quote": "USDT", "contract": False}}
        def load_markets(self):
            return self.markets
        def fetch_tickers(self):
            return {"TEST/USDT": {"quoteVolume": 1000000}}
        def fetch_ohlcv(self, symbol, timeframe, limit):
            arr = []
            for i in range(150):
                close = 100 + float(np.sin(i / 8))
                arr.append([i * 3_600_000, close - .2, close + .7, close - .8, close, 1000])
            return arr
        def close(self):
            pass

    old = (bot.ccxt, bot.LATEST_FILE, bot.TRADES_FILE, bot.SENT_FILE, bot.TELEGRAM_BOT_TOKEN,
           bot.TIMEFRAMES, bot.SCAN_TOP_N, bot.OI_MODE)
    with tempfile.TemporaryDirectory() as folder:
        try:
            bot.ccxt = types.SimpleNamespace(mexc=FakeExchange)
            bot.LATEST_FILE = Path(folder) / "latest_signals.json"
            bot.TRADES_FILE = Path(folder) / "trades_history.json"
            bot.SENT_FILE = Path(folder) / "sent_alerts.json"
            bot.TELEGRAM_BOT_TOKEN = ""
            bot.TIMEFRAMES = ["1h", "4h", "1d"]
            bot.SCAN_TOP_N = 1
            bot.OI_MODE = "off"
            payload = bot.executar_scan()
            assert payload["ohlcv_requests_ok"] == 3, payload
            assert payload["errors"] == 0, payload
            assert (Path(folder) / "latest_signals.json").exists()
            assert (Path(folder) / "trades_history.json").exists()
            print("OK 20 Scan completo com exchange simulada e persistência JSON")
        finally:
            (bot.ccxt, bot.LATEST_FILE, bot.TRADES_FILE, bot.SENT_FILE, bot.TELEGRAM_BOT_TOKEN,
             bot.TIMEFRAMES, bot.SCAN_TOP_N, bot.OI_MODE) = old


if __name__ == "__main__":
    integration_offline()


def integration_oi_offline():
    """Scanner com OI mockado para verificar bloqueio de falsos positivos."""
    import tempfile
    import types
    import copy
    import rsi_divergence_bot as bot

    class FakeMexc:
        timeframes = {"1h": True}
        def __init__(self, _config): pass
        def load_markets(self):
            return {"TEST/USDT": {"active": True, "spot": True, "base": "TEST", "contract": False}}
        def fetch_tickers(self):
            return {"TEST/USDT": {"quoteVolume": 1e8}}
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return [[i * 3600000, 100,101,99,100,1000] for i in range(140)]
        def close(self): pass

    class FakeBinance:
        def __init__(self, _config):
            self.has = {"fetchOpenInterestHistory": True}
            self.markets = {"TEST/USDT:USDT": {"swap": True}}
        def load_markets(self):
            return self.markets
        def fetch_open_interest_history(self, symbol, timeframe, since, limit):
            assert symbol == "TEST/USDT:USDT"
            return [
                {"timestamp": (i * 3600000), "openInterestAmount": value}
                for i, value in ((110, 1000), (122, 950), (134, 900))
            ]
        def close(self): pass

    df, rsi = sample()
    base_signal = detetar_divergencias_rsi(df, "1h", rsi_override=rsi)[0]
    old = (bot.ccxt, bot.LATEST_FILE, bot.TRADES_FILE, bot.SENT_FILE, bot.TELEGRAM_BOT_TOKEN,
           bot.TIMEFRAMES, bot.SCAN_TOP_N, bot.OI_MODE, bot.detetar_divergencias_rsi)
    with tempfile.TemporaryDirectory() as folder:
        try:
            bot.ccxt = types.SimpleNamespace(mexc=FakeMexc, binance=FakeBinance)
            bot.LATEST_FILE = Path(folder) / "latest_signals.json"
            bot.TRADES_FILE = Path(folder) / "trades_history.json"
            bot.SENT_FILE = Path(folder) / "sent_alerts.json"
            bot.TELEGRAM_BOT_TOKEN = ""
            bot.TIMEFRAMES = ["1h"]
            bot.SCAN_TOP_N = 1
            bot.OI_MODE = "confirm_if_available"
            bot.detetar_divergencias_rsi = lambda _df, _tf: [copy.deepcopy(base_signal)]
            payload = bot.executar_scan()
            assert payload["signals_count"] == 1
            assert payload["signals"][0]["oi"]["confirmado"] is True
            print("OK integração OI: divergência + OI decrescente confirmados")
        finally:
            (bot.ccxt, bot.LATEST_FILE, bot.TRADES_FILE, bot.SENT_FILE, bot.TELEGRAM_BOT_TOKEN,
             bot.TIMEFRAMES, bot.SCAN_TOP_N, bot.OI_MODE, bot.detetar_divergencias_rsi) = old


if __name__ == "__main__":
    integration_oi_offline()


def test_swap_market_selection():
    import rsi_divergence_bot as bot
    class FakeExchange:
        def load_markets(self):
            return {
                "ZEC/USDT": {"active": True, "spot": True, "contract": False, "base": "ZEC"},
                "ZEC/USDT:USDT": {"active": True, "swap": True, "spot": False, "contract": True, "base": "ZEC"},
            }
        def fetch_tickers(self):
            return {"ZEC/USDT": {"quoteVolume": 1e7}, "ZEC/USDT:USDT": {"quoteVolume": 2e7}}

    old = bot.MARKET_TYPE
    try:
        bot.MARKET_TYPE = "swap"
        assert bot.get_liquid_usdt_symbols(FakeExchange(), 1) == ["ZEC/USDT:USDT"]
        bot.MARKET_TYPE = "spot"
        assert bot.get_liquid_usdt_symbols(FakeExchange(), 1) == ["ZEC/USDT"]
        print("OK seleção exclusiva spot vs perpétuos MEXC")
    finally:
        bot.MARKET_TYPE = old


if __name__ == "__main__":
    test_swap_market_selection()
