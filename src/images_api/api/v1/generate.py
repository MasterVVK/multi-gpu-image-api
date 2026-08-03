"""Постановка задач и работа с ними."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, Request

from ...core.errors import ConflictError, NotFoundError
from ...core.security import ApiKeyInfo, require_api_key
from ...domain.resolve import resolve, resolve_spec
from ...domain.schemas import (
    GenerateRequest,
    GenerateResponse,
    TaskInfo,
    TaskListResponse,
    task_row_to_info,
)
from ...services.scheduler import enqueue
from ..deps import get_pool, get_scheduler, get_storage, get_tasks

router = APIRouter(tags=["generate"])


async def _to_info(request: Request, row) -> TaskInfo:
    tasks = get_tasks(request)
    storage = get_storage(request)
    position = await tasks.queue_position(row["task_id"])
    image_url = storage.image_url(row["image_path"]) if row["image_path"] else None
    return task_row_to_info(row, queue_position=position, image_url=image_url)


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: Request,
    payload: GenerateRequest,
    key: ApiKeyInfo = Depends(require_api_key),
) -> GenerateResponse:
    scheduler = get_scheduler(request)
    tasks = get_tasks(request)
    pool = get_pool(request)

    # Валидация модели и параметров идёт до всего остального: клиент должен
    # получить внятную 422, а не задачу, которая молча упадёт через минуту.
    spec = resolve_spec(payload.model)
    resolved = resolve(payload, spec)

    if payload.idempotency_key:
        existing = await tasks.by_idempotency_key(payload.idempotency_key)
        if existing is not None:
            info = await _to_info(request, existing)
            return GenerateResponse(
                task_id=info.task_id,
                status=info.status,
                model=info.model,
                node=info.node,
                queue_position=info.queue_position or 0,
                message="задача с таким idempotency_key уже существует",
            )

    if payload.node and pool.get(payload.node) is None:
        raise NotFoundError("узел %s не зарегистрирован" % payload.node)

    task_id = await enqueue(
        scheduler,
        payload,
        resolved,
        api_key_id=key.key_id,
        request_id=getattr(request.state, "request_id", None),
    )

    position = await tasks.queue_position(task_id) or 1
    serving = pool.nodes_serving(resolved.model_id)
    message = "поставлено в очередь"
    if not serving:
        # Не отказываем сразу: узел может подняться в пределах дедлайна.
        message = "поставлено в очередь, но сейчас ни один узел не имеет весов %s" % resolved.model_id

    return GenerateResponse(
        task_id=task_id,
        status="queued",
        model=resolved.model_id,
        node=payload.node,
        queue_position=position,
        message=message,
    )


@router.get("/tasks/{task_id}", response_model=TaskInfo, dependencies=[Depends(require_api_key)])
async def get_task(request: Request, task_id: str) -> TaskInfo:
    row = await get_tasks(request).get(task_id)
    if row is None:
        raise NotFoundError("задача %s не найдена" % task_id)
    return await _to_info(request, row)


@router.get(
    "/tasks/{task_id}/wait", response_model=TaskInfo, dependencies=[Depends(require_api_key)]
)
async def wait_task(
    request: Request,
    task_id: str,
    timeout: float = Query(default=300.0, ge=1.0, le=1800.0),
    poll: float = Query(default=1.0, ge=0.2, le=10.0),
) -> TaskInfo:
    """Дождаться завершения задачи, не гоняя клиента по циклу поллинга.

    По тайм-ауту возвращается текущее состояние задачи, а не ошибка: работа
    продолжается, и клиент может просто спросить ещё раз.
    """
    tasks = get_tasks(request)
    deadline = asyncio.get_running_loop().time() + timeout

    while True:
        row = await tasks.get(task_id)
        if row is None:
            raise NotFoundError("задача %s не найдена" % task_id)
        if row["status"] in ("completed", "failed", "cancelled"):
            return await _to_info(request, row)
        if asyncio.get_running_loop().time() >= deadline:
            return await _to_info(request, row)
        await asyncio.sleep(poll)


@router.delete("/tasks/{task_id}", response_model=TaskInfo, dependencies=[Depends(require_api_key)])
async def cancel_task(request: Request, task_id: str) -> TaskInfo:
    scheduler = get_scheduler(request)
    tasks = get_tasks(request)

    row = await tasks.get(task_id)
    if row is None:
        raise NotFoundError("задача %s не найдена" % task_id)
    if row["status"] in ("completed", "failed", "cancelled"):
        raise ConflictError("задача уже завершена со статусом %s" % row["status"])

    await scheduler.cancel(task_id)
    row = await tasks.get(task_id)
    return await _to_info(request, row)


@router.get("/tasks", response_model=TaskListResponse, dependencies=[Depends(require_api_key)])
async def list_tasks(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    status: str | None = Query(default=None),
) -> TaskListResponse:
    tasks = get_tasks(request)
    rows = await tasks.recent(limit=limit, status=status)
    counts = await tasks.count_by_status()
    return TaskListResponse(
        tasks=[await _to_info(request, row) for row in rows],
        total=sum(counts.values()),
    )
