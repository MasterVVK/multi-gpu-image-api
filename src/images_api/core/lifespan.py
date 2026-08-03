"""Запуск и остановка: что во что складывается и в каком порядке.

Всё долгоживущее кладётся на app.state, зависимости читают его из
request.app.state, остановка идёт в обратном порядке и каждый шаг —
в своём try/except.

Порядок старта важен: восстановление (recover) обязано отработать ДО того, как
арбитр начнёт раздавать задачи, иначе он поедет по узлам, за которыми ещё висят
модели от прошлой жизни процесса.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from ..db.conn import Database
from ..services.node_pool import NodePool
from ..services.prompt_enhancer import PromptEnhancer
from ..services.scheduler import Scheduler
from ..services.storage import ImageStorage
from ..services.tasks import TaskRepository
from ..services.vram_guard import VramGuard
from ..services.worker_client import WorkerClient
from .config import get_settings
from .logging import configure_logging, get_logger
from .security import check_startup_security

log = get_logger("lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log.info("startup_begin", host=settings.host, port=settings.port)

    check_startup_security(settings)

    storage = ImageStorage(settings)
    storage.prepare()

    db = Database(settings.db_path)
    await db.connect()
    await db.migrate()
    log.info("db_ready", path=str(settings.db_path))

    # Один клиент на всё приложение: пул соединений переиспользуется и для
    # воркеров, и для PE. Лимиты скромные — узлов немного.
    http = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.worker_connect_timeout_s),
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        # Прокси для внешних загрузок здесь не нужен: все адреса локальные
        # либо в LAN, и SOCKS-прокси только мешал бы.
        trust_env=False,
    )

    worker_client = WorkerClient(http, settings.worker_token)
    tasks = TaskRepository(db)
    pool = NodePool(db, worker_client, settings)
    await pool.load()

    enhancer = PromptEnhancer(http, settings)
    scheduler = Scheduler(
        tasks=tasks,
        pool=pool,
        client=worker_client,
        storage=storage,
        enhancer=enhancer,
        settings=settings,
    )
    guard = VramGuard(db=db, pool=pool, client=worker_client, settings=settings)

    app.state.settings = settings
    app.state.db = db
    app.state.http = http
    app.state.storage = storage
    app.state.tasks = tasks
    app.state.pool = pool
    app.state.enhancer = enhancer
    app.state.scheduler = scheduler
    app.state.guard = guard

    # Сначала привести узлы в известное состояние, только потом планировать.
    await scheduler.recover()
    await scheduler.start()
    await guard.start()

    retention = asyncio.create_task(_retention_loop(tasks, storage, settings), name="retention")

    log.info(
        "startup_complete",
        nodes=sorted(pool.nodes),
        nodes_ready=pool.ready_count,
        pe_configured=enhancer.configured,
    )

    try:
        yield
    finally:
        log.info("shutdown_begin")
        for step, coro in (
            ("retention", _cancel(retention)),
            ("guard", guard.stop()),
            ("scheduler", scheduler.stop()),
            ("http", http.aclose()),
            ("db", db.close()),
        ):
            try:
                await coro
            except Exception as exc:
                log.exception("shutdown_step_failed", step=step, error=str(exc))
        log.info("shutdown_complete")


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _retention_loop(tasks: TaskRepository, storage: ImageStorage, settings) -> None:
    """Раз в час чистим старые задачи и осиротевшие картинки."""
    while True:
        await asyncio.sleep(3600)
        try:
            removed_paths = await tasks.cleanup(settings.task_retention_days)
            deleted = sum(1 for path in removed_paths if storage.delete(path))
            if removed_paths:
                log.info("retention_done", tasks=len(removed_paths), images=deleted)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("retention_failed", error=str(exc))
