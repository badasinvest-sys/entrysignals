"""Teste sintético rápido da deteção multi-pivot; não usa internet nem credenciais."""
import numpy as np
import pandas as pd

from rsi_divergence_bot import detetar_divergencias_rsi


def make_df(kind: str) -> pd.DataFrame:
    n = 64
    idx = np.arange(n)
    timestamps = (1_750_000_000_000 + idx * 3_600_000).astype(np.int64)
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 99.0)
    rsi = np.full(n, 50.0)
    pivots = [40, 50, 60]

    if kind == "bearish":
        highs = [103.0, 108.0, 114.0]
        rsi_peaks = [72.0, 64.0, 57.0]
        for i, p, r in zip(pivots, highs, rsi_peaks):
            high[i] = p
            rsi[i] = r
    else:
        lows = [97.0, 93.0, 88.0]
        rsi_troughs = [28.0, 36.0, 43.0]
        for i, p, r in zip(pivots, lows, rsi_troughs):
            low[i] = p
            rsi[i] = r

    return pd.DataFrame({
        "timestamp": timestamps, "open": close, "high": high, "low": low,
        "close": close, "volume": np.ones(n), "rsi": rsi,
    })


def main() -> None:
    bearish = detetar_divergencias_rsi(make_df("bearish"), "1h")
    bullish = detetar_divergencias_rsi(make_df("bullish"), "1h")
    assert any(s["tipo"] == "bearish" and s["pivots_count"] >= 3 for s in bearish), bearish
    assert any(s["tipo"] == "bullish" and s["pivots_count"] >= 3 for s in bullish), bullish
    print("OK: divergência bearish multi-pivot detetada")
    print("OK: divergência bullish multi-pivot detetada")


if __name__ == "__main__":
    main()
