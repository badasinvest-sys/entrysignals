"""Gráficos candlestick verdes/vermelhos para Telegram e histórico Streamlit."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from divergence_engine import calcular_rsi

BASE_DIR = Path(__file__).resolve().parent
CHARTS_DIR = Path(os.getenv("CHARTS_DIR", str(BASE_DIR / "trade_charts")))


def chart_name(trade_id: str) -> str:
    return "trade_" + hashlib.sha256(trade_id.encode("utf-8")).hexdigest()[:24] + ".png"


def _render_chart(df: pd.DataFrame, event: dict[str, Any], trade_record: dict[str, Any] | None = None) -> bytes:
    if df.empty or len(df) < 20:
        raise ValueError("Sem velas suficientes para criar o gráfico.")
    original = df.reset_index(drop=True)
    pivots = event.get("pivots", [])
    start = max(0, len(original) - 110)
    if pivots:
        first_idx = int(min(int(p.get("price_idx", len(original))) for p in pivots))
        start = max(0, min(start, first_idx - 7))
    start = max(0, len(original) - 360, start)
    d = original.iloc[start:].reset_index(drop=True)
    n = len(d)

    fig, (ax, arsi) = plt.subplots(2, 1, figsize=(12.5, 7.0), sharex=True,
                                   gridspec_kw={"height_ratios": [3.8, 1.2]})
    fig.patch.set_facecolor("#101827")
    for a in (ax, arsi):
        a.set_facecolor("#111827")
        a.grid(True, color="#374151", linewidth=0.35, alpha=0.6)
        a.tick_params(colors="#e5e7eb", labelsize=9)
        for sp in a.spines.values():
            sp.set_color("#4b5563")
    color_up, color_down = "#22c55e", "#f87171"
    for i, row in enumerate(d.itertuples(index=False)):
        col = color_up if float(row.close) >= float(row.open) else color_down
        ax.plot((i, i), (float(row.low), float(row.high)), linewidth=1.05, color=col)
        low_body, high_body = sorted((float(row.open), float(row.close)))
        body_h = max(high_body - low_body, abs(float(row.close)) * 1e-6)
        ax.add_patch(patches.Rectangle((i - .33, low_body), .66, body_h, facecolor=col, edgecolor=col))

    trade = (trade_record or {}).get("trade") or event.get("trade")
    if trade_record is not None:
        trade = {"entry": trade_record["entrada"], "sl": trade_record["sl"], "tp": trade_record["tp"]}

    if event.get("categoria") == "rsi_divergence":
        kind = event.get("tipo", "bullish")
        point_indices = []
        prices = []
        rsi_points = []
        for p in pivots:
            idx = int(p["price_idx"]) - start
            if 0 <= idx < n:
                point_indices.append(idx)
                prices.append(float(p["price"]))
                rsi_points.append(float(p["rsi"]))
                ax.scatter([idx], [p["price"]], s=85, marker="o", facecolors="none", edgecolors="#fbbf24", linewidth=2, zorder=6)
        if len(point_indices) >= 2:
            ax.plot(point_indices, prices, color="#f59e0b", linewidth=1.9, zorder=6)
            arsi.plot(point_indices, rsi_points, color="#f59e0b", linewidth=1.8, marker="o", markersize=5)
        rsi_full = calcular_rsi(original["close"].to_numpy(float))
        arsi.plot(np.arange(n), rsi_full[start:], color="#d1d5db", linewidth=1.25)
        arsi.axhline(70, linestyle="--", color="#a78bfa", linewidth=.8)
        arsi.axhline(30, linestyle="--", color="#a78bfa", linewidth=.8)
        arsi.set_ylim(10, 90)
        label = "BAIXISTA" if kind == "bearish" else "ALTISTA"
        title = f"RSI {label} | {event['symbol']} | {event['timeframe']} | {len(pivots)} pivots"
        oi = event.get("oi") or {}
        sub = f"Rejeições {event.get('rejection_count', 0)} | Sweeps {event.get('sweep_count', 0)} | OI: {oi.get('status', 'Indisponível')}"
    else:
        lo = float(event.get("zona_min", event.get("nivel", 0)))
        hi = float(event.get("zona_max", event.get("nivel", 0)))
        ax.axhspan(lo, hi, color="#a78bfa", alpha=.25, label="Zona de interesse")
        ax.axhline(float(event.get("nivel", (lo + hi)/2)), color="#c4b5fd", linestyle="--", linewidth=1.2)
        arsi.plot(np.arange(n), calcular_rsi(original["close"].to_numpy(float))[start:], color="#d1d5db", linewidth=1.25)
        arsi.axhline(70, linestyle="--", color="#a78bfa", linewidth=.8)
        arsi.axhline(30, linestyle="--", color="#a78bfa", linewidth=.8)
        arsi.set_ylim(10, 90)
        title = f"{event.get('fase', 'TRADE')} | {event['symbol']} | {event['timeframe']}"
        sub = f"Zona: {lo:.8g} - {hi:.8g} | Volume zona: {event.get('volume_zona_ratio', '?')}x"
        if trade:
            entry, sl, tp = (float(trade[k]) for k in ("entry", "sl", "tp"))
            x0 = max(0, n - 18)
            x1 = n + 9
            ax.add_patch(patches.Rectangle((x0, sl), x1 - x0, entry-sl, facecolor="#ef4444", alpha=.13, edgecolor="none"))
            ax.add_patch(patches.Rectangle((x0, entry), x1 - x0, tp-entry, facecolor="#22c55e", alpha=.13, edgecolor="none"))
            for y, col, label in ((entry, "#fbbf24", "ENTRADA"), (sl, "#f87171", "STOP"), (tp, "#4ade80", "TP 2R")):
                ax.axhline(y, color=col, linewidth=1.15, linestyle="--")
                ax.text(x1 + 0.5, y, f"{label}  {y:.8g}", color=col, va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(-2, n + 32)
            if trade_record and trade_record.get("status") in ("Ganha", "Perdida"):
                outcome = "TP atingido" if trade_record["status"] == "Ganha" else "SL atingido"
                sub += " | " + outcome

    ax.set_title(title, color="#f9fafb", fontsize=16, weight="bold", pad=15, loc="left")
    ax.text(.005, 1.015, sub, transform=ax.transAxes, color="#d1d5db", fontsize=9, va="bottom")
    ax.set_ylabel("Preço", color="#d1d5db")
    arsi.set_ylabel("RSI 14", color="#d1d5db")
    ticks = np.linspace(0, max(0, n - 1), min(7, n), dtype=int)
    arsi.set_xticks(ticks)
    ts = d["timestamp"].to_numpy(dtype=np.int64)
    arsi.set_xticklabels([pd.to_datetime(int(ts[t]), unit="ms", utc=True).strftime("%d %b %Hh") for t in ticks], rotation=0)
    fig.subplots_adjust(hspace=.09, left=.08, right=.86, bottom=.08, top=.88)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def render_signal_chart(df: pd.DataFrame, event: dict[str, Any]) -> bytes:
    return _render_chart(df, event)


def save_trade_chart(trade: dict[str, Any], df: pd.DataFrame) -> str:
    """Persiste PNG local legível pela app. Um redeploy pode apagá-lo."""
    event = {
        "categoria": "breakout", "symbol": trade["symbol"], "timeframe": trade["timeframe"],
        "fase": trade["status"], "nivel": trade["nivel"], "zona_min": trade["zona_min"],
        "zona_max": trade["zona_max"], "volume_zona_ratio": trade.get("volume_zona_ratio", "?"),
    }
    data = _render_chart(df, event, trade)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = chart_name(str(trade["trade_id"]))
    path = CHARTS_DIR / filename
    tmp = path.with_suffix(".png.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return filename
