"""Persistência opcional em Redis Upstash REST ($0 tier), com ficheiro local.

No Render Free os ficheiros locais são efémeros. Se a integração Redis for
configurada, o ledger de alertas/trades reside também fora do Render.
Falhas do Redis são *fail closed*: não se enviam alertas duplicáveis.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

URL = os.getenv("UPSTASH_REDIS_REST_URL", "").strip().rstrip("/")
TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
PREFIX = os.getenv("UPSTASH_STATE_PREFIX", "entrysignals:v8:").strip() or "entrysignals:v8:"
ENABLED = bool(URL or TOKEN)


class StateUnavailable(RuntimeError):
    pass


class StateStore:
    def __init__(self):
        self.remote = ENABLED
        if self.remote and not (URL.startswith("https://") and TOKEN):
            raise StateUnavailable("UPSTASH_REDIS_REST_URL e UPSTASH_REDIS_REST_TOKEN têm de estar ambas configuradas")

    def _command(self, *args: Any) -> Any:
        try:
            response = requests.post(
                URL,
                headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
                json=list(args), timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise StateUnavailable(f"Upstash: {str(payload['error'])[:180]}")
            return payload.get("result")
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise StateUnavailable("Falha de acesso ao estado Upstash; envio suspenso para evitar duplicações") from exc

    def load(self, kind: str, local_path: Path, default: Any) -> Any:
        if self.remote:
            result = self._command("GET", PREFIX + kind)
            if result is not None:
                try:
                    value = json.loads(result)
                except (ValueError, TypeError) as exc:
                    raise StateUnavailable(f"Estado remoto corrompido: {kind}") from exc
                self._local_write(local_path, value)
                return value
        try:
            if local_path.exists():
                return json.loads(local_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
        return default

    def save(self, kind: str, local_path: Path, value: Any) -> None:
        if self.remote:
            result = self._command("SET", PREFIX + kind, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            if result != "OK":
                raise StateUnavailable(f"Upstash não confirmou gravação de {kind}")
        self._local_write(local_path, value)

    @staticmethod
    def _local_write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


    @contextmanager
    def exclusive_scan(self, ttl_seconds: int = 1200):
        """Impede dois scanners Render de enviar alertas no mesmo intervalo.

        A operação SET NX é atómica. Se o serviço cair, o bloqueio expira.
        """
        if not self.remote:
            yield True
            return
        key = PREFIX + "scan_lock"
        token = uuid.uuid4().hex
        granted = self._command("SET", key, token, "NX", "EX", max(900, ttl_seconds)) == "OK"
        if not granted:
            yield False
            return
        try:
            yield True
        finally:
            # Apagar somente se o bloqueio ainda for o nosso.
            script = "if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end"
            self._command("EVAL", script, 1, key, token)
