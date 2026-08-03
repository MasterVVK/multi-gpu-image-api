"""Сторож VRAM: следит, чтобы сервис не удерживал память при сбоях и простое.

Философия простая: проверять реальное состояние, а не отвечает ли HTTP,
и уметь эскалировать до пересоздания процесса.

Три ситуации, которые он закрывает:

1. **Простой.** Узел ничего не делает дольше IMG_IDLE_UNLOAD_SECONDS, а модель
   резидентна — выгружаем. Для hidream это возвращает 16.4 ГБ, для остальных
   освобождает системную RAM и делает учёт честным.

2. **Осиротевшая генерация.** Воркер говорит busy=True, а у шлюза аренды на этот
   узел нет. Так выглядит падение шлюза посреди генерации: задача уже никому не
   нужна, а карта занята. Отменяем, выгружаем.

3. **Залипшая память.** Модель осталась резидентной, хотя выгрузку уже просили.
   Эскалация — перезапуск юнита воркера, если это разрешено и узел локальный.

Все вмешательства пишутся в таблицу vram_incidents: без журнала «оно само
починилось» невозможно отличить от «оно и не ломалось».
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time

from ..core.config import Settings
from ..core.logging import get_logger
from ..db.conn import Database
from .node_pool import NodePool, NodeRuntime, _iso, utcnow
from .worker_client import WorkerClient, WorkerRejected, WorkerUnavailable

log = get_logger("vram_guard")


class VramGuard:
    def __init__(
        self,
        *,
        db: Database,
        pool: NodePool,
        client: WorkerClient,
        settings: Settings,
    ) -> None:
        self.db = db
        self.pool = pool
        self.client = client
        self.settings = settings
        self._loop: asyncio.Task | None = None
        self._stopping = False
        # Момент, когда за воркером впервые заметили подозрительно занятую память.
        self._suspect_since: dict[tuple[str, str], float] = {}

    async def start(self) -> None:
        self._loop = asyncio.create_task(self._run(), name="vram-guard")
        log.info("vram_guard_started", interval_s=self.settings.guard_interval_s)

    async def stop(self) -> None:
        self._stopping = True
        if self._loop is not None:
            self._loop.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._loop
            self._loop = None

    async def _run(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.settings.guard_interval_s)
            try:
                for node in list(self.pool.nodes.values()):
                    await self.check_node(node)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("vram_guard_error", error=str(exc))

    async def check_node(self, node: NodeRuntime) -> None:
        # Узел под арендой проверять нечего: там идёт легальная работа.
        if node.busy:
            return

        idle_for = time.monotonic() - node.last_activity

        for backend, url in node.base_urls.items():
            key = (node.name, backend)
            try:
                status = await self.client.status(
                    url, token=node.worker_token, timeout=self.settings.worker_status_timeout_s
                )
            except (WorkerUnavailable, WorkerRejected):
                self._suspect_since.pop(key, None)
                continue

            # (2) Генерация без аренды — сирота после падения шлюза.
            if status.busy and status.current_task_id:
                await self._handle_orphan(node, backend, url, status.current_task_id, status.gpu.used_mb)
                continue

            if status.resident_model is None:
                self._suspect_since.pop(key, None)
                continue

            first_seen = self._suspect_since.setdefault(key, time.monotonic())
            held_for = time.monotonic() - first_seen

            # (1) Штатная выгрузка по простою.
            if idle_for >= self.settings.idle_unload_s:
                await self._unload(
                    node,
                    backend,
                    url,
                    status.resident_model,
                    status.gpu.used_mb,
                    status.gpu.free_mb,
                    action="unload",
                    reason="idle %.0fs" % idle_for,
                )
                self._suspect_since.pop(key, None)
                continue

            # (3) Память держится дольше допустимого, хотя работы нет.
            if held_for >= self.settings.stuck_vram_s and idle_for >= self.settings.stuck_vram_s:
                freed = await self._unload(
                    node,
                    backend,
                    url,
                    status.resident_model,
                    status.gpu.used_mb,
                    status.gpu.free_mb,
                    action="unload",
                    reason="stuck %.0fs" % held_for,
                )
                if not freed:
                    await self._escalate(node, backend, status.resident_model, status.gpu.used_mb)
                self._suspect_since.pop(key, None)

    async def _handle_orphan(
        self, node: NodeRuntime, backend: str, url: str, task_id: str, used_mb: int
    ) -> None:
        log.warning("orphan_generation", node=node.name, backend=backend, task_id=task_id)
        with contextlib.suppress(Exception):
            await self.client.cancel(url, task_id, token=node.worker_token)
        await self._record(
            node.name, used_mb, None, task_id, "cancel_orphan", "cancel sent for %s" % task_id
        )
        # Выгрузка последует на следующем проходе, когда воркер освободится:
        # /unload на занятом воркере намеренно отвечает 409, а не ждёт.

    async def _unload(
        self,
        node: NodeRuntime,
        backend: str,
        url: str,
        resident: str,
        used_mb: int,
        free_mb: int,
        *,
        action: str,
        reason: str,
    ) -> bool:
        log.info("guard_unload", node=node.name, backend=backend, model=resident, reason=reason)
        try:
            result = await self.client.unload(url, token=node.worker_token)
        except (WorkerUnavailable, WorkerRejected) as exc:
            await self._record(node.name, used_mb, free_mb, resident, action, "failed: %s" % exc)
            return False

        if node.resident_backend == backend:
            node.resident_backend = node.resident_model = None

        freed_mb = max(0, result.gpu.free_mb - free_mb)
        await self._record(
            node.name,
            used_mb,
            result.gpu.free_mb,
            resident,
            action,
            "%s; освобождено %d МБ" % (reason, freed_mb),
        )
        return result.gpu.free_mb >= free_mb

    async def _escalate(self, node: NodeRuntime, backend: str, resident: str, used_mb: int) -> None:
        unit = "images-worker-%s" % backend

        if node.kind != "local":
            # На удалённом узле systemctl нам недоступен: там свой watchdog юнита.
            await self._record(
                node.name, used_mb, None, resident, "gave_up", "удалённый узел, нужен рестарт %s вручную" % unit
            )
            log.error("vram_stuck_remote", node=node.name, backend=backend, unit=unit)
            return

        if not self.settings.allow_worker_restart:
            await self._record(
                node.name,
                used_mb,
                None,
                resident,
                "gave_up",
                "рестарт запрещён (IMG_ALLOW_WORKER_RESTART=false), нужен %s вручную" % unit,
            )
            log.error("vram_stuck_no_restart", node=node.name, backend=backend, unit=unit)
            return

        log.warning("restarting_worker", node=node.name, unit=unit)
        outcome = await asyncio.get_running_loop().run_in_executor(None, _systemctl_restart, unit)
        await self._record(node.name, used_mb, None, resident, "restart_worker", outcome)

    async def _record(
        self,
        node: str,
        used_mb: int | None,
        free_mb: int | None,
        resident: str | None,
        action: str,
        outcome: str,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO vram_incidents (node, detected_at, used_mb, free_mb, resident, action, outcome)
            VALUES (?,?,?,?,?,?,?)
            """,
            (node, _iso(utcnow()), used_mb, free_mb, resident, action, outcome[:500]),
        )

    async def recent_incidents(self, limit: int = 20) -> list[dict]:
        rows = await self.db.fetch_all(
            "SELECT * FROM vram_incidents ORDER BY detected_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]


def _systemctl_restart(unit: str) -> str:
    """Перезапуск юнита воркера.

    Требует правила в sudoers:
      user ALL=(root) NOPASSWD: /usr/bin/systemctl restart images-worker-*
    По умолчанию выключено (IMG_ALLOW_WORKER_RESTART=false).
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", unit],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return "restart failed: %s" % exc
    if result.returncode == 0:
        return "restarted"
    return "restart rc=%d: %s" % (result.returncode, (result.stderr or "").strip()[:200])
