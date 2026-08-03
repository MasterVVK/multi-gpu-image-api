"""Репозиторий задач. Источник правды — SQLite, очередь в памяти лишь кэш.

Главное отличие от четырёх старых сервисов: там задачи жили в обычном dict и
исчезали при каждом рестарте вместе с очередью. Здесь перезапуск шлюза — это
восстановление, а не потеря.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from ..core.logging import get_logger
from ..db.conn import Database
from ..domain.schemas import (
    ACTIVE_STATUSES,
    RUNNING_STATUSES,
    GenerateRequest,
    ResolvedParams,
)

log = get_logger("tasks")


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # --- создание ---

    async def create(
        self,
        request: GenerateRequest,
        resolved: ResolvedParams,
        *,
        vram_wait_timeout_s: int,
        source: str = "native",
        group_id: str | None = None,
        api_key_id: str | None = None,
        request_id: str | None = None,
        use_pe: bool | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        now = utcnow()
        # deadline_at НЕ ставится при создании. Десять минут — это лимит ожидания
        # свободной VRAM, а не общий лимит жизни задачи: стоять в очереди за
        # нашими же задачами — нормально, и убивать за это нельзя.
        # Отсчёт начинается в момент, когда планировщик впервые не смог разместить
        # задачу из-за занятой карты (см. Scheduler._start_vram_wait).

        await self.db.execute(
            """
            INSERT INTO tasks (
                task_id, status, source, group_id, idempotency_key,
                model_id, backend, node, priority, request_json,
                prompt, negative_prompt, use_pe,
                width, height, aspect_ratio, num_steps, guidance_scale, shift, scheduler,
                seed_requested, created_at, deadline_at, api_key_id, request_id
            ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?)
            """,
            (
                task_id,
                "queued",
                source,
                group_id,
                request.idempotency_key,
                resolved.model_id,
                resolved.backend,
                request.node,
                request.priority,
                json.dumps(request.model_dump(), ensure_ascii=False),
                request.prompt,
                resolved.negative_prompt or None,
                int(use_pe) if use_pe is not None else None,
                resolved.width,
                resolved.height,
                resolved.aspect_ratio,
                resolved.num_inference_steps,
                resolved.guidance_scale,
                resolved.shift,
                resolved.scheduler,
                request.seed,
                _iso(now),
                None,
                api_key_id,
                request_id,
            ),
        )
        log.info(
            "task_enqueued",
            task_id=task_id,
            model=resolved.model_id,
            size="%dx%d" % (resolved.width, resolved.height),
            steps=resolved.num_inference_steps,
            prompt_len=len(request.prompt),
            prompt_preview=request.prompt[:60],
        )
        return task_id

    # --- чтение ---

    async def get(self, task_id: str) -> aiosqlite.Row | None:
        return await self.db.fetch_one("SELECT * FROM tasks WHERE task_id=?", (task_id,))

    async def by_idempotency_key(self, key: str) -> aiosqlite.Row | None:
        return await self.db.fetch_one("SELECT * FROM tasks WHERE idempotency_key=?", (key,))

    async def by_group(self, group_id: str) -> list[aiosqlite.Row]:
        return await self.db.fetch_all(
            "SELECT * FROM tasks WHERE group_id=? ORDER BY created_at", (group_id,)
        )

    async def pending(self, limit: int = 200) -> list[aiosqlite.Row]:
        """Задачи, ждущие исполнения, в порядке планирования."""
        return await self.db.fetch_all(
            """
            SELECT * FROM tasks
            WHERE status IN ('queued', 'waiting_vram')
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
            """,
            (limit,),
        )

    async def running(self) -> list[aiosqlite.Row]:
        placeholders = ",".join("?" for _ in RUNNING_STATUSES)
        return await self.db.fetch_all(
            "SELECT * FROM tasks WHERE status IN (%s) ORDER BY started_at" % placeholders,
            RUNNING_STATUSES,
        )

    async def active(self) -> list[aiosqlite.Row]:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        return await self.db.fetch_all(
            "SELECT * FROM tasks WHERE status IN (%s) ORDER BY priority, created_at" % placeholders,
            ACTIVE_STATUSES,
        )

    async def recent(self, limit: int = 50, status: str | None = None) -> list[aiosqlite.Row]:
        if status:
            return await self.db.fetch_all(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
            )
        return await self.db.fetch_all(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    async def count_by_status(self) -> dict[str, int]:
        rows = await self.db.fetch_all("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
        return {row["status"]: row["n"] for row in rows}

    async def queue_position(self, task_id: str) -> int | None:
        """Позиция в общей очереди ожидания (1 — следующая на исполнение)."""
        row = await self.db.fetch_one(
            "SELECT priority, created_at, status FROM tasks WHERE task_id=?", (task_id,)
        )
        if row is None or row["status"] not in ("queued", "waiting_vram"):
            return None
        ahead = await self.db.fetch_one(
            """
            SELECT COUNT(*) AS n FROM tasks
            WHERE status IN ('queued', 'waiting_vram')
              AND (priority < ? OR (priority = ? AND created_at < ?))
            """,
            (row["priority"], row["priority"], row["created_at"]),
        )
        return int(ahead["n"]) + 1

    # --- изменение состояния ---

    async def set_fields(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        columns = ", ".join("%s=?" % name for name in fields)
        params: Sequence[Any] = (*fields.values(), task_id)
        await self.db.execute("UPDATE tasks SET %s WHERE task_id=?" % columns, params)

    async def mark_status(self, task_id: str, status: str, **fields: Any) -> None:
        await self.set_fields(task_id, status=status, **fields)

    async def mark_started(self, task_id: str, node: str) -> None:
        now = utcnow()
        row = await self.get(task_id)
        queue_time = None
        if row is not None:
            created = datetime.fromisoformat(row["created_at"])
            queue_time = round((now - created).total_seconds(), 2)
        await self.set_fields(
            task_id,
            status="leased",
            node=node,
            started_at=_iso(now),
            heartbeat_at=_iso(now),
            queue_time_seconds=queue_time,
            attempts=(row["attempts"] + 1) if row is not None else 1,
            deadline_at=None,
        )

    async def start_vram_wait(self, task_id: str, timeout_s: int) -> datetime:
        """Запустить окно ожидания свободной VRAM, если оно ещё не запущено.

        Возвращает момент, после которого задача будет отклонена.
        """
        deadline = utcnow() + timedelta(seconds=timeout_s)
        await self.set_fields(task_id, deadline_at=_iso(deadline))
        return deadline

    async def heartbeat(self, task_id: str) -> None:
        await self.set_fields(task_id, heartbeat_at=_iso(utcnow()))

    async def mark_completed(
        self,
        task_id: str,
        *,
        image_id: str,
        image_path: str,
        image_bytes: int,
        seed_used: int,
        generation_time_seconds: float,
        load_time_seconds: float,
        width: int,
        height: int,
    ) -> None:
        await self.set_fields(
            task_id,
            status="completed",
            completed_at=_iso(utcnow()),
            image_id=image_id,
            image_path=image_path,
            image_bytes=image_bytes,
            seed_used=seed_used,
            # Фактический размер: пайплайн мог его изменить (см. hidream).
            width=width,
            height=height,
            generation_time_seconds=generation_time_seconds,
            load_time_seconds=load_time_seconds,
            error=None,
            error_code=None,
        )
        log.info("task_completed", task_id=task_id, seconds=generation_time_seconds)

    async def mark_failed(self, task_id: str, error: str, error_code: str) -> None:
        await self.set_fields(
            task_id,
            status="failed",
            completed_at=_iso(utcnow()),
            error=error[:2000],
            error_code=error_code,
        )
        log.warning("task_failed", task_id=task_id, error_code=error_code, error=error[:200])

    async def mark_cancelled(self, task_id: str) -> None:
        await self.set_fields(
            task_id,
            status="cancelled",
            completed_at=_iso(utcnow()),
            error_code="cancelled",
        )
        log.info("task_cancelled", task_id=task_id)

    async def request_cancel(self, task_id: str) -> None:
        await self.set_fields(task_id, cancel_requested=1)

    async def requeue(self, task_id: str, status: str = "queued") -> None:
        await self.set_fields(task_id, status=status, node=None, heartbeat_at=None)

    async def set_pe_result(
        self, task_id: str, enhanced_prompt: str | None, pe_status: str, pe_error: str | None
    ) -> None:
        await self.set_fields(
            task_id,
            enhanced_prompt=enhanced_prompt,
            pe_status=pe_status,
            pe_error=pe_error[:500] if pe_error else None,
        )

    # --- восстановление ---

    async def recover_on_startup(self, max_attempts: int = 2) -> tuple[int, int]:
        """Разобрать задачи, застигнутые падением шлюза.

        Что было в работе — вернуть в очередь, если попыток ещё не исчерпано;
        иначе честно завершить как interrupted. Без этого такие задачи висели бы
        в `generating` вечно.
        """
        placeholders = ",".join("?" for _ in RUNNING_STATUSES)

        requeued = await self.db.execute(
            """
            UPDATE tasks SET status='queued', node=NULL, heartbeat_at=NULL
            WHERE status IN (%s) AND attempts < ?
            """ % placeholders,
            (*RUNNING_STATUSES, max_attempts),
        )
        failed = await self.db.execute(
            """
            UPDATE tasks
            SET status='failed', error_code='interrupted',
                error='gateway restarted while the task was running',
                completed_at=?
            WHERE status IN (%s)
            """ % placeholders,
            (_iso(utcnow()), *RUNNING_STATUSES),
        )
        if requeued or failed:
            log.info("startup_recovery", requeued=requeued, failed=failed)
        return requeued, failed

    async def sweep_stale(self, stale_seconds: int, max_attempts: int = 2) -> int:
        """Задачи с протухшим heartbeat — обработчик умер, а строка осталась."""
        cutoff = _iso(utcnow() - timedelta(seconds=stale_seconds))
        placeholders = ",".join("?" for _ in RUNNING_STATUSES)

        rows = await self.db.fetch_all(
            """
            SELECT task_id, attempts FROM tasks
            WHERE status IN (%s) AND (heartbeat_at IS NULL OR heartbeat_at < ?)
            """ % placeholders,
            (*RUNNING_STATUSES, cutoff),
        )
        for row in rows:
            if row["attempts"] < max_attempts:
                await self.requeue(row["task_id"])
                log.warning("stale_lease_requeued", task_id=row["task_id"])
            else:
                await self.mark_failed(
                    row["task_id"], "lease expired without heartbeat", "worker_timeout"
                )
        return len(rows)

    # --- уборка ---

    async def cleanup(self, task_retention_days: int) -> list[str]:
        """Удалить старые завершённые задачи, вернув пути их картинок."""
        cutoff = _iso(utcnow() - timedelta(days=task_retention_days))
        rows = await self.db.fetch_all(
            """
            SELECT task_id, image_path FROM tasks
            WHERE completed_at IS NOT NULL AND completed_at < ?
            """,
            (cutoff,),
        )
        if not rows:
            return []
        await self.db.execute(
            "DELETE FROM tasks WHERE completed_at IS NOT NULL AND completed_at < ?", (cutoff,)
        )
        return [row["image_path"] for row in rows if row["image_path"]]
