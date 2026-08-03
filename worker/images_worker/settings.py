"""Конфигурация воркера — только os.environ, без pydantic-settings.

Пакет обязан ставиться в чужие venv с нулевыми зависимостями, а
pydantic-settings есть не везде. Значения приходят из
/etc/images-api/worker-<backend>.env через systemd EnvironmentFile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .registry import DEFAULT_WORKER_PORTS


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class WorkerSettings:
    backend: str
    node: str
    host: str
    port: int
    token: str

    # Выгрузка модели после простоя — страховка на случай, если упал сам шлюз.
    # В штатном режиме выгрузкой управляет арбитр шлюза, он знает больше.
    idle_unload_s: int
    # Полный выход процесса после простоя: единственный способ убрать CUDA-контекст
    # в ноль. systemd поднимет обратно за пару секунд (Restart=always).
    idle_exit_s: int

    preload_model: str | None
    log_level: str

    @property
    def unit_name(self) -> str:
        return "images-worker-%s" % self.backend


def load_settings() -> WorkerSettings:
    backend = _str("IMG_WORKER_BACKEND")
    if backend not in DEFAULT_WORKER_PORTS:
        raise SystemExit(
            "IMG_WORKER_BACKEND must be one of %s, got %r"
            % (", ".join(sorted(DEFAULT_WORKER_PORTS)), backend)
        )

    return WorkerSettings(
        backend=backend,
        node=_str("IMG_WORKER_NODE", "br-01"),
        # Только петля: наружу воркеры не смотрят никогда, удалённый доступ —
        # исключительно через ssh-туннель шлюза.
        host=_str("IMG_WORKER_HOST", "127.0.0.1"),
        port=_int("IMG_WORKER_PORT", DEFAULT_WORKER_PORTS[backend]),
        token=_str("IMG_WORKER_TOKEN"),
        idle_unload_s=_int("IMG_WORKER_IDLE_UNLOAD_S", 180),
        idle_exit_s=_int("IMG_WORKER_IDLE_EXIT_S", 600),
        preload_model=_str("IMG_WORKER_PRELOAD_MODEL") or None,
        log_level=_str("IMG_WORKER_LOG_LEVEL", "info"),
    )
