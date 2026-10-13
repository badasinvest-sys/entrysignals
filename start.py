"""Executa o scanner e o painel no mesmo serviço web gratuito do Render.

O Render só precisa de um Web Service. UptimeRobot mantém o serviço acordado;
este supervisor mantém o scanner e o Streamlit a correr lado a lado.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
children: dict[str, subprocess.Popen] = {}
stop_requested = False


def _terminate_children() -> None:
    for process in list(children.values()):
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
    deadline = time.time() + 10
    for process in list(children.values()):
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()


def _handle_signal(signum, _frame) -> None:
    global stop_requested
    stop_requested = True
    print(f"Recebido sinal {signum}; a terminar processos filhos...", flush=True)
    _terminate_children()


def _start_scanner() -> subprocess.Popen:
    print("A iniciar scanner RSI em background...", flush=True)
    return subprocess.Popen(
        [sys.executable, str(BASE_DIR / "rsi_divergence_bot.py")],
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
    )


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    port = os.getenv("PORT", "8501")
    children["scanner"] = _start_scanner()
    web_command = [
        sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py"),
        "--server.address", "0.0.0.0",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    print(f"A iniciar painel Streamlit na porta {port}...", flush=True)
    children["web"] = subprocess.Popen(web_command, cwd=str(BASE_DIR), env=os.environ.copy())

    try:
        while not stop_requested:
            web_process = children["web"]
            if web_process.poll() is not None:
                print(f"O Streamlit terminou com código {web_process.returncode}.", flush=True)
                return int(web_process.returncode or 0)

            scanner = children["scanner"]
            if scanner.poll() is not None:
                print(f"O scanner terminou com código {scanner.returncode}; a reiniciar em 5 segundos.", flush=True)
                time.sleep(5)
                if not stop_requested:
                    children["scanner"] = _start_scanner()
            time.sleep(1)
    finally:
        _terminate_children()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
